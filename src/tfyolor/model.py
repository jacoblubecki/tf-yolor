
from typing import Any, Dict, List, Union
import numpy as np

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as L

from . import dataformat
from .layers import (
    SpaceToDepth,
    ConvBnAct,
    DownsampleInput,
    GenericDownsample,
    FPN,
    PAN,
    ImplicitVariable
)
from .util import check
from .yolo import YoloLayer


def _validate_anchors(anchors):
    def check_valid_list(x, length):
        if not isinstance(x, (tuple, list)) or len(x) != length:
            raise ValueError(f'Expected "{x}" to be tuple of length {length}.')

    def validate_contents(x):
        if not isinstance(x, (int, float)) or x <= 0:
            raise ValueError(f'All sizes must be positive, but found ({x}).')

    check_valid_list(anchors, 4)
    for scale in anchors:
        check_valid_list(scale, 3)
        for anchor in scale:
            check_valid_list(anchor, 2)
            for value in anchor:
                validate_contents(value)


def default_yolor_p6_anchors():
    """ Returns the default YOLOR-P6 anchors.

    They are formatted into a tuple of tuples with an effective shape of
    (4, 3, 2) which means they can be used with the :class:`YolorP6`
    constructor directly.
    """
    return (
        ((19,  27), (44,  40), (38,  94)),
        ((96,  68), (86, 152), (180, 137)),
        ((140, 301), (303, 264), (238, 542)),
        ((436, 615), (739, 380), (925, 792))
    )


def default_yolor_p6_config() -> Dict[str, Dict[str, Any]]:
    """ Returns a config dictionary to match the original YOLOR-P6.

    The dictionary is meant to be used with the :class:`YolorP6` constructor.
    """
    return {
        'downsample': {
            'alphas': (64, 128, 192, 256, 320),
            'resblocks': (3, 7, 7, 3, 3),
        },
        'neck': {
            'alpha': 320,
            'fpn_channels': (256, 192, 128),
            'pan_channels': (192, 256, 320),
        },
        'head': {
            'channels': (256, 384, 512, 640)
        },
    }


class YolorNeckP6(L.Layer):
    """ The neck for a YoloP6 model.

    Contains (in rough order):

    - CSPSPP: Cross-State Partial Connections + Spatial-Pyramid Pooling
        - `CSP arXiv Paper <https://arxiv.org/abs/1911.11929>`_
        - `SPP arXiv Paper <https://arxiv.org/abs/1406.4729>`_
    - FPN (x3): Feature Pyramid Network
        - `FPN arXiv Paper <https://arxiv.org/abs/1612.03144>`_
    - PAN (x3): Path Aggregation Network
        - `PAN arXiv Paper <https://arxiv.org/abs/1803.01534>`_

    :param int alpha: Number of output channels for the CSPSPP section.
    :param fpn_channels: The number of output channels for each FPN.
        Should contain 3 positive int values.
    :type fpn_channels: tuple, list
    :param pan_channels: The number of output channels for each PAN.
        Should contain 3 positive int values.
    :type pan_channels: tuple, list
    """

    def __init__(
            self,
            alpha=320,
            fpn_channels=(256, 192, 128),
            pan_channels=(192, 256, 320),
            data_format='channels_last'
    ):
        super().__init__()

        check._is_positive_int('alpha', alpha)
        check._is_tuple_of_positive_int('fpn_channels', fpn_channels, 3)
        check._is_tuple_of_positive_int('pan_channels', pan_channels, 3)

        self.alpha = alpha
        self.fpn_channels = fpn_channels
        self.pan_channels = pan_channels
        self.data_format = data_format
        self.cat_axis = dataformat.get_channel_axis_keras(data_format)

    def build(self, input_size):
        def conv(out_channels, kernel_size, stride):
            return ConvBnAct(
                out_channels, kernel_size, stride,
                activation='leaky',
                data_format=self.data_format
            )

        def pool(size):
            return L.MaxPooling2D(
                pool_size=(size, size),
                strides=(1, 1),
                padding='same',
                data_format=self.data_format
            )

        self.mp1 = pool(5)
        self.mp2 = pool(9)
        self.mp3 = pool(13)

        a = self.alpha
        self.conv1 = conv(a, 1, 1)
        self.conv2 = conv(a, 1, 1)
        self.conv3 = conv(a, 3, 1)
        self.conv4 = conv(a, 1, 1)
        self.conv5 = conv(a, 1, 1)
        self.conv6 = conv(a, 3, 1)
        self.conv7 = conv(a, 1, 1)
        self.spp_concat = L.Concatenate(axis=self.cat_axis)
        self.cspspp_concat = L.Concatenate(axis=self.cat_axis)

        # FPNs
        fpnc = self.fpn_channels
        self.fpn5 = FPN(channels=fpnc[0], data_format=self.data_format)
        self.fpn4 = FPN(channels=fpnc[1], data_format=self.data_format)
        self.fpn3 = FPN(channels=fpnc[2], data_format=self.data_format)

        # PANs
        panc = self.pan_channels
        self.pan4 = PAN(channels=panc[0], data_format=self.data_format)
        self.pan5 = PAN(channels=panc[1], data_format=self.data_format)
        self.pan6 = PAN(channels=panc[2], data_format=self.data_format)

    def call(self, ds5, ds4, ds3, ds2, training=False):
        """ Returns an output for the inputs at each scale.

        The downsampled inputs to this layer are a sequence of simple, strided
        convolutional downsamples which are progressively applied to the input
        at the beginning of the network. This can be roughly modeled as:

        >>> ds1 = downsample_stage_1(inputs)
        >>> ds2 = downsample_stage_2(ds1)
        >>> ds3 = downsample_stage_3(ds2)
        >>> # etc...

        FPN stages are applied in reverse order from FPN-5 to FPN-3. As the
        stages are applied, decreasingly downsampled data is aggregated into
        the outputs. That is, CSPSPP is applied to the most downsampled output
        from the downsampling stages. The output of the CSPSPP and the second
        most downsampled output are passed into FPN-5. This output is passed to
        FPN-4 alongside the third most downsampled input. Finally, the output
        of FPN-4 is passed to FPN-3 alongside the least downsampled input.

        **Outputs Explained:**

        - FPN-3: The fully upsampled and aggregated output of the FPN layers.
            This represents the smallest output scale.
        - PAN-4: Convolutionally downsamples from the output from FPN-3 and
            aggregates with the output of FPN-4. This is the second smallest
            output scale.
        - PAN-5: Convolutionally downsamples from the output from PAN-4 and
            aggregates with the output of FPN-5. This is the second largest
            output scale.
        - PAN-6: Convolutionally downsamples from the output from PAN-5 and
            aggregates with the output of FPN-6. This is the largest output
            scale.

        :param ds5: The output from the final downsample layer.
        :param ds4: The output from the fourth downsample layer.
        :param ds3: The output from the third downsample layer.
        :param ds2: The output from the second downsample layer.
        :returns: The outputs of FPN-3, PAN-4, PAN-5, and PAN-6 in that order.
        """
        # Inputs: 100, 85, 70, 43
        x1 = self.conv1(ds5, training=training)
        x2 = self.conv2(ds5, training=training)
        x3 = self.conv3(x2, training=training)
        x4 = self.conv4(x3, training=training)

        # SPP: Spatial Pyramidal Pooling
        m1 = self.mp1(x4)
        m2 = self.mp2(x4)
        m3 = self.mp3(x4)
        spp = self.spp_concat([m3, m2, m1, x4])
        # End SPP

        x5 = self.conv5(spp, training=training)
        x6 = self.conv6(x5, training=training)
        x6 = self.cspspp_concat([x6, x1])
        x7 = self.conv7(x6, training=training)  # 115

        # End of CSPSPP

        # FPN Layers
        x8 = self.fpn5(x7, ds4, training=training)  # 131
        x9 = self.fpn4(x8, ds3, training=training)  # 147
        x10 = self.fpn3(x9, ds2, training=training)  # 163

        # PAN Layers
        x11 = self.pan4(x10, x9, training=training)  # 176
        x12 = self.pan5(x11, x8, training=training)  # 189
        x13 = self.pan6(x12, x7, training=training)  # 202

        return x10, x11, x12, x13

    def get_config(self):
        return {
            'alpha': self.alpha,
            'fpn_channels': self.fpn_channels,
            'pan_channels': self.pan_channels,
            'data_format': self.data_format
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class YolorHeadP6(L.Layer):
    """ The final layer of the YOLOR-P6 model.

    :param output_channels int: The number of output channels for each of the
        final convolutional layers.
    :param n_classes int: The number of possible classes.
    :param anchors list: List of shape 4x3x2 meaning 3 anchor boxes consisting
        of a 2 dimensions (width/height) at 4 scales. The scales should be
        ordered from smallest to largest.
    :param channels: The number of output channels in the second to last of
        each of the four output scales. You should supply exactly 4 values with
        the first being the number of channels used at the smallest scale and
        the last being the number of channels used at the largest scale.
    :param data_format str: Must be 'channels_first' or 'channels_last'.
        The 'channels_first' option requires a GPU.
    """

    def __init__(
        self,
        output_channels,
        n_classes,
        anchors,
        channels=(256, 384, 512, 640),
        data_format='channels_last'
    ):
        super().__init__()

        check._is_positive_int('output_channels', output_channels)
        check._is_positive_int('n_classes', n_classes)
        _validate_anchors(anchors)
        check._is_tuple_of_positive_int('channels', channels, 4)

        self.out_channels = output_channels
        self.n_classes = n_classes
        self.data_format = data_format
        self.anchors = anchors
        self.channel_config = channels

    def build(self, input_sizes):

        def conv(
            out_channels,
            kernel_size,
            stride,
            act,
            use_bias=False,
            bias_initializer=None
        ):
            bias_init = bias_initializer if bias_initializer else 'zeros'
            return ConvBnAct(
                out_channels,
                kernel_size,
                stride,
                activation=act,
                use_bias=use_bias,
                use_batch_norm=not use_bias,
                bias_initializer=bias_init,
                data_format=self.data_format,
            )

        c_first = self.data_format == 'channels_first'
        add_sizes = []
        mul_sizes = []
        add_channels = self.channel_config
        for ac in add_channels:
            if c_first:
                add_sizes.append((ac, 1, 1))
                mul_sizes.append((self.out_channels, 1, 1))
            else:
                add_sizes.append((1, 1, ac))
                mul_sizes.append((1, 1, self.out_channels))

        def init_from_yolo(y):
            return YolorHeadP6._create_init(
                y.n_anchors, y.n_outputs, y.n_classes, y.stride)

        self.yolo1 = YoloLayer(
            anchors=self.anchors[0],
            n_classes=self.n_classes,
            stride=8,
            data_format=self.data_format
        )

        init1 = init_from_yolo(self.yolo1)
        self.conv1 = conv(add_channels[0], 3, 1, 'leaky')
        self.conv2 = conv(self.out_channels, 1, 1, 'linear',
                          use_bias=True, bias_initializer=init1)
        self.impl_add1 = ImplicitVariable(add_sizes[0])
        self.impl_mul1 = ImplicitVariable(mul_sizes[0], init_mean=1.0)
        self.add1 = L.Add()
        self.mul1 = L.Multiply()

        self.yolo2 = YoloLayer(
            anchors=self.anchors[1],
            n_classes=self.n_classes,
            stride=16,
            data_format=self.data_format
        )

        init2 = init_from_yolo(self.yolo2)
        self.conv3 = conv(add_channels[1], 3, 1, 'leaky')
        self.conv4 = conv(self.out_channels, 1, 1, 'linear',
                          use_bias=True, bias_initializer=init2)
        self.impl_add2 = ImplicitVariable(add_sizes[1])
        self.impl_mul2 = ImplicitVariable(mul_sizes[1], init_mean=1.0)
        self.add2 = L.Add()
        self.mul2 = L.Multiply()

        self.yolo3 = YoloLayer(
            anchors=self.anchors[2],
            n_classes=self.n_classes,
            stride=32,
            data_format=self.data_format
        )

        init3 = init_from_yolo(self.yolo3)
        self.conv5 = conv(add_channels[2], 3, 1, 'leaky')
        self.conv6 = conv(self.out_channels, 1, 1, 'linear',
                          use_bias=True, bias_initializer=init3)
        self.impl_add3 = ImplicitVariable(add_sizes[2])
        self.impl_mul3 = ImplicitVariable(mul_sizes[2], init_mean=1.0)
        self.add3 = L.Add()
        self.mul3 = L.Multiply()

        self.yolo4 = YoloLayer(
            anchors=self.anchors[3],
            n_classes=self.n_classes,
            stride=64,
            data_format=self.data_format
        )

        init4 = init_from_yolo(self.yolo4)
        self.conv7 = conv(add_channels[3], 3, 1, 'leaky')
        self.conv8 = conv(self.out_channels, 1, 1, 'linear',
                          use_bias=True, bias_initializer=init4)
        self.impl_add4 = ImplicitVariable(add_sizes[3])
        self.impl_mul4 = ImplicitVariable(mul_sizes[3], init_mean=1.0)
        self.add4 = L.Add()
        self.mul4 = L.Multiply()

    def call(self, inputs, training=None):
        # Layer numbers based on original yolor_p6.cfg config file.
        out163, out176, out189, out202 = inputs

        ia1 = self.impl_add1(None)
        im1 = self.impl_mul1(None)
        x1 = self.add1([self.conv1(out163, training=training), ia1])
        x2 = self.mul1([self.conv2(x1, training=training), im1])

        ia2 = self.impl_add2(None)
        im2 = self.impl_mul2(None)
        x3 = self.add2([self.conv3(out176, training=training), ia2])
        x4 = self.mul2([self.conv4(x3, training=training), im2])

        ia3 = self.impl_add3(None)
        im3 = self.impl_mul3(None)
        x5 = self.add3([self.conv5(out189, training=training), ia3])
        x6 = self.mul3([self.conv6(x5, training=training), im3])

        ia4 = self.impl_add4(None)
        im4 = self.impl_mul4(None)
        x7 = self.add4([self.conv7(out202, training=training), ia4])
        x8 = self.mul4([self.conv8(x7, training=training), im4])

        y1 = self.yolo1(x2, training=training)
        y2 = self.yolo2(x4, training=training)
        y3 = self.yolo3(x6, training=training)
        y4 = self.yolo4(x8, training=training)

        return y1, y2, y3, y4

    def get_config(self):
        return {
            'output_channels': self.out_channels,
            'n_classes': self.n_classes,
            'anchors': self.anchors,
            'channels': self.channel_config,
            'data_format': self.data_format
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @staticmethod
    def _create_init(na, no, nc, stride):
        """ Creates a function which is used to do smart bias init."""
        def smart_yolo_bias(shape, dtype=None):
            std = 1 / np.sqrt(shape[0])
            bias = tf.random.uniform(shape, -std, std, dtype=dtype).numpy()
            bias = bias[:no * na].reshape(na, -1)
            # obj (8 objects per 640 image)
            bias[:, 4] += np.log(8 / (640 / stride) ** 2)
            bias[:, 5:] += np.log(0.6 / (nc - 0.99))  # cls (sigmoid(p) = 1/nc)
            return tf.constant(bias.reshape(shape))

        return smart_yolo_bias


class YolorP6(keras.Model):
    """ An implementation of the YOLOR-P6 model.

    Original Model: `yolor_p6.cfg`_

    Configs should contain exactly the following fields and tuple shapes. The
    values may all be adjusted to increase or decrease the size of the
    resulting network.

    .. code-block:: python

        # Default Config
        {
            'downsample': {
                # alpha param for each of 5 downsample layers.
                'alphas': (64, 128, 192, 256, 320),

                # res_blocks param for each of 5 downsample layers.
                'resblocks': (3, 7, 7, 3, 3),
            },
            # See YolorNeckP6 documentation.
            'neck': {
                'alpha': 320,
                'fpn_channels': (256, 192, 128),
                'pan_channels': (192, 256, 320),
            },
            # See YolorHeadP6 documentation.
            'head': {
                'channels': (256, 384, 512, 640)
            },
        }

    :param int classes: The number of classes this model should predict.
    :param anchors: The anchor boxes used for this model. Should be a list or
        tuple of shape (4, 3, 2) containing all positive python primitive
        numerical values. Pass `None` to use the default configuration.
    :type anchors: list, tuple
    :param dict config: See above for default schema. Pass `None` to use the
        default configuration.

    .. _yolor_p6.cfg: https://github.com/WongKinYiu/yolor/blob/
        b168a4dd0fe22068bb6f43724e22013705413afb/cfg/yolor_p6.cfg
    """

    def __init__(
            self,
            classes: int,
            anchors: Union[None, List[List[Union[int, float]]]] = None,
            config: Union[None, Dict[str, Dict[str, Any]]] = None,
            data_format='channels_last'
    ):
        super().__init__()
        self.classes = classes
        self.anchors = anchors
        self.data_format = data_format

        if anchors is None:
            anchors = default_yolor_p6_anchors()

        if config is None:
            config = default_yolor_p6_config()

        output_ch = (5 + classes) * 3

        self.reorg = SpaceToDepth(data_format=data_format)

        # Setup convolutional downsampling layers.
        ds_config = config['downsample']
        ds_alphas = ds_config['alphas']
        ds_resblocks = ds_config['resblocks']
        check._is_tuple_of_positive_int(
            'config.downsample.alphas', ds_alphas, 5)
        check._is_tuple_of_positive_int(
            'config.downsample.resblocks', ds_resblocks, 5)

        a1, a2, a3, a4, a5 = ds_alphas
        r1, r2, r3, r4, r5 = ds_resblocks
        self.ds1 = DownsampleInput(a1, r1, data_format=data_format)
        self.ds2 = GenericDownsample(a2, r2, data_format=data_format)
        self.ds3 = GenericDownsample(a3, r3, data_format=data_format)
        self.ds4 = GenericDownsample(a4, r4, data_format=data_format)
        self.ds5 = GenericDownsample(a5, r5, data_format=data_format)

        # Setup neck layer.
        neck_config = config['neck']
        neck_alpha = neck_config['alpha']
        fpn_channels = neck_config['fpn_channels']
        pan_channels = neck_config['pan_channels']

        self.neck = YolorNeckP6(
            alpha=neck_alpha,
            fpn_channels=fpn_channels,
            pan_channels=pan_channels,
            data_format=data_format
        )

        # Setup head layer.
        head_config = config['head']
        head_channel_config = head_config['channels']

        self.head = YolorHeadP6(
            output_ch,
            classes,
            anchors,
            head_channel_config,
            data_format=data_format
        )

    def call(self, inputs, training=None):
        inputs = self.reorg(inputs)
        ds1 = self.ds1(inputs, training=training)
        ds2 = self.ds2(ds1, training=training)
        ds3 = self.ds3(ds2, training=training)
        ds4 = self.ds4(ds3, training=training)
        ds5 = self.ds5(ds4, training=training)

        neck = self.neck(ds5, ds4, ds3, ds2, training=training)
        output = self.head(neck, training=training)

        if training:
            return output
        else:
            x, p = zip(*output)
            x = tf.concat(x, 1)
            return x, p
