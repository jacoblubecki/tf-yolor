from typing import Optional, Tuple, Union

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
from tensorflow import keras
from tensorflow import nn


from . import dataformat
from .activations import create_activation as act
from .util import check


class ConvBnAct(L.Layer):
    """ Creates a layer containing a `Conv2D`, `BatchNorm`, and `Activation`.

    :param int out_channels: The number of output channels after convolution.
    :param kernel_size: The size of the kernel as a pair of ints (height,
        width), or a single int for a square kernel.
    :type kernel_size: int, tuple
    :param stride: The stride of the kernel as a pair of ints (dy, dx), or a
        single int to use the same stride in both directions.
    :type stride: int, tuple
    :param str activation: A string compatible with
        `tfyolor.activations.create_activation`.
    :param bool use_bias: True if the `Conv2D` should use a bias. Recommended
        to set to :code:`False` if :code:`use_batch_norm == True`.
    :param bool use_batch_norm: True if the layer should use batch norm.
    :param bias_initializer: Any valid bias initializer argument.
    :param str data_format: Must be 'channels_first' or 'channels_last'.
        The 'channels_first' option requires a GPU.
    """

    def __init__(
            self,
            out_channels: int,
            kernel_size: Union[int, Tuple[int]],
            stride: Union[int, Tuple[int]],
            activation: Optional[str] = 'mish',
            use_bias=False,
            use_batch_norm=True,
            bias_initializer='zeros',
            data_format='channels_last'):
        super().__init__()

        check._is_positive_int('out_channels', out_channels)
        check._is_positive_int_or_tuple('kernel_size', kernel_size, 2)
        check._is_positive_int_or_tuple('stride', stride, 2)

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_bias = use_bias
        self.data_format = data_format
        self.bias_init = bias_initializer

        self.activation = act(activation)
        self.act_type = activation

        bn_axis = dataformat.get_channel_axis_keras(data_format)
        self.batch_norm = L.BatchNormalization(
            axis=bn_axis,
            momentum=0.03,
            epsilon=0.0001,
        ) if use_batch_norm else None

    def build(self, input_shape):
        kernel_init = keras.initializers.HeUniform()
        self.conv = L.Conv2D(
            self.out_channels,
            self.kernel_size,
            input_shape=input_shape[1:],
            padding='same',
            strides=self.stride,
            activation=None,
            use_bias=self.use_bias,
            data_format=self.data_format,
            kernel_initializer=kernel_init,
            kernel_regularizer=keras.regularizers.l2(5e-4),
            bias_initializer=self.bias_init,
            bias_regularizer=keras.regularizers.l2(5e-4),
        )

    def call(self, inputs, training=False):
        x = self.conv(inputs)

        if self.batch_norm:
            x = self.batch_norm(x, training=training)

        if self.activation:
            x = self.activation(x)

        return x

    def get_config(self):
        return {
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'activation': self.act_type,
            'use_batch_norm': self.batch_norm is not None,
            'use_bias': self.use_bias,
            'bias_initializer': self.bias_init,
            'data_format': self.data_format,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Upsample(L.Layer):
    """ Upsamples an inputs width/height to a target shape.

    :param str data_format: Must be 'channels_first' or 'channels_last'.
        The 'channels_first' option requires a GPU.
    """

    def __init__(self, data_format='channels_last'):
        super().__init__()
        self.data_format = data_format
        self.is_channels_first = dataformat.is_channels_first_keras(
            data_format)

    def call(self, x, target_size, training=False):
        """ Forward pass of the upsample layer.

        .. note::
            In training, the image will be upsampled via
            :code:`tf.image.resize`.
            The interpolation mode is set to :code:`'nearest'`.

            During evaluation, it will be done by reshaping and broadcasting.

        .. warning::
            Output width/height must evenly divide input width/height.

        :param x: The tensor to upsample.
        :param target_size: A keras shape for the resulting tensor.
            Should be NCHW for 'channels_first' of NHWC for 'channels_last'.
        :returns: The upsampled tensor.
        """
        c_first = self.is_channels_first

        # Current and target dimensions.
        b, c, h, w = dataformat.unpack_shape_to_nchw(K.shape(x), c_first)
        _, _, th, tw = dataformat.unpack_shape_to_nchw(target_size, c_first)

        out_size = (b, c, th, tw) if self.is_channels_first else (b, th, tw, c)
        out_size = tf.stack(out_size)

        if training:
            if c_first:
                resize_size = out_size[2:]
                x = tf.transpose(x, (0, 2, 3, 1))
                x = tf.image.resize(x, resize_size, method='nearest')
                return tf.transpose(x, (0, 3, 1, 2))
            else:
                resize_size = out_size[1:3]
                return tf.image.resize(x, resize_size, method='nearest')
        else:
            if c_first:
                axes_added = (b, c, h, 1, w, 1)
                broadcast_shape = (b, c, h, th // h, w, tw // w)
            else:
                axes_added = (b, h, 1, w, 1, c)
                broadcast_shape = (b, h, th // h, w, tw // w, c)

            out = tf.reshape(x, axes_added)
            out = tf.broadcast_to(out, broadcast_shape)
            return tf.reshape(out, out_size)

    def get_config(self):
        return {'data_format': self.data_format}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class ResBlock(L.Layer):
    """ A layer of sub-blocks which support (optional) residual connections.

    **Sub-Block Structure:**

    Each sub-block consists of a 1x1 convolution followed by a 3x3
    convolution. After each sub-block, the input to that sub-block
    can be added to the output of that sub-block.

    :param int blocks: The number of sub-blocks making up this layer,
    :param bool shortcut: True if each block should use a residual connection.
    :param str data_format: Must be 'channels_first' or 'channels_last'.
        The 'channels_first' option requires a GPU.
    """

    def __init__(
            self,
            blocks: int,
            shortcut=True,
            data_format='channels_last'
    ):
        super().__init__()

        check._is_positive_int('blocks', blocks)

        self.n_blocks = blocks
        self.shortcut = shortcut
        self.data_format = data_format

    def build(self, input_shape):
        channel_axis = dataformat.get_channel_axis_keras(self.data_format)
        channels = input_shape[channel_axis]

        def conv(kernel_size):
            return ConvBnAct(
                channels, kernel_size, stride=1, data_format=self.data_format)

        self.blocks = []
        self.adds = []
        for i in range(self.n_blocks):
            block = [conv(kernel_size=1), conv(kernel_size=3)]
            self.blocks.append(block)
            self.adds.append(L.Add() if self.shortcut else None)

    def call(self, x, training=False):
        for block, add in zip(self.blocks, self.adds):
            h = x
            for conv in block:
                h = conv(h, training=training)

            # This does not work with `x + h`.
            x = add([x, h]) if self.shortcut else h

        return x

    def get_config(self):
        return {
            'blocks': self.n_blocks,
            'shortcut': self.shortcut,
            'data_format': self.data_format
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def _space_to_depth(x, stride: int, channels_first: bool):
    """ Wrapper for `tensorflow.nn.space_to_depth`."""
    tf_format = 'NCHW' if channels_first else 'NHWC'
    return nn.space_to_depth(x, block_size=stride, data_format=tf_format)


class SpaceToDepth(L.Layer):
    """ Similar to YOLOR reorg, but a bit faster with slightly different order.

    :param int stride: Positive int used with `tensorflow.nn.space_to_depth`.
        Leave as default (2) for behavior closest to YOLOR reorg.
    :param str data_format: Must be 'channels_first' or 'channels_last'.
        The 'channels_first' option requires a GPU.
    """

    def __init__(self, stride: int = 2, data_format: str = 'channels_last'):
        super().__init__()

        check._is_positive_int('stride', stride)

        self.data_format = data_format
        self.is_channels_first = dataformat.is_channels_first_keras(
            data_format)
        self.stride = stride

    def call(self, x):
        return _space_to_depth(x, self.stride, self.is_channels_first)

    def get_config(self):
        return {
            'data_format': self.data_format,
            'stride': self.stride
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.function
def _reorg_yolor(x, channels_first: bool):
    """ YOLOR implementation of space-to-depth conversion.

    :param bool channels_first: True if the first non-batch dimension is
        channels. Otherwise false (assumes last index is channels instead).
    :returns: The reorganized tensor.
    """
    if channels_first:
        return tf.concat((
            x[..., ::2, ::2],
            x[..., 1::2, ::2],
            x[..., ::2, 1::2],
            x[..., 1::2, 1::2],
        ), 1)
    else:
        return tf.concat((
            x[:, ::2, ::2, :],
            x[:, 1::2, ::2, :],
            x[:, ::2, 1::2, :],
            x[:, 1::2, 1::2, :]
        ), -1)


class Reorg(L.Layer):
    """ Kersa implementation of original YOLOR reorg.

    :param int stride: Positive int used with `tensorflow.nn.space_to_depth`.
    :param str data_format: Must be 'channels_first' or 'channels_last'.
        The 'channels_first' option requires a GPU.
    """

    def __init__(self, data_format: str = 'channels_last'):
        super().__init__()

        self.data_format = data_format
        self.is_channels_first = dataformat.is_channels_first_keras(
            data_format)

    def call(self, x):
        return _reorg_yolor(x, self.is_channels_first)

    def get_config(self):
        return {'data_format': self.data_format}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class DownsampleInput(L.Layer):
    """ Creates the first downsample of the YOLOR network.

    The output of this layer has its width and height each reduced by a factor
    of 2, and has `2 * alpha` output channels.

    :param int alpha: The number of channels the input should be increased to
        during the first convolution.
    :param int res_blocks: The number of blocks in the residual component of
        this layer.
    :param str data_format: Must be 'channels_first' or 'channels_last'.
        The 'channels_first' option requires a GPU.
    """

    def __init__(
        self,
        alpha: int = 64,
        res_blocks: int = 3,
        data_format='channels_last'
    ):
        super().__init__()

        check._is_positive_int('alpha', alpha)
        check._is_positive_int('res_blocks', res_blocks)

        self.alpha = alpha
        self.res_blocks = res_blocks
        self.data_format = data_format
        self.is_channels_first = dataformat.is_channels_first_keras(
            data_format)
        self.cat_axis = dataformat.get_channel_axis_keras(data_format)

        # We don't care about the input shape, build now.
        self._build_immediately()

    def _build_immediately(self):
        def conv(out_channels, kernel_size, stride):
            return ConvBnAct(
                out_channels, kernel_size, stride, data_format=self.data_format
            )

        alpha = self.alpha
        beta = alpha * 2
        self.conv1 = conv(alpha, 3, 1)
        self.conv2 = conv(beta, 3, 2)
        self.conv3 = conv(alpha, 1, 1)
        self.conv4 = conv(alpha, 1, 1)
        self.res5 = ResBlock(blocks=self.res_blocks,
                             data_format=self.data_format)
        self.conv6 = conv(beta, 1, 1)
        self.concat = L.Concatenate(axis=self.cat_axis)

    def call(self, inputs, training=False):
        x1 = self.conv1(inputs, training)
        x2 = self.conv2(x1, training)

        # split
        x3 = self.conv3(x2, training)
        x4 = self.conv4(x2, training)
        x5 = self.res5(x4, training)

        # merge
        x5 = self.concat([x5, x3])
        x6 = self.conv6(x5, training)
        return x6

    def get_config(self):
        return {
            'alpha': self.alpha,
            'res_blocks': self.res_blocks,
            'data_format': self.data_format
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class GenericDownsample(L.Layer):
    """ The downsampling YOLOR layer used after the initial downsample.

    The output of this layer has its width and height each reduced by a factor
    of 2, and has `2 * alpha` output channels.

    :param int alpha: The number of output channels divided by 2.
    :param int res_blocks: The number of blocks in the residual component of
        this layer.
    :param str data_format: Must be 'channels_first' or 'channels_last'.
        The 'channels_first' option requires a GPU.
    """

    def __init__(
        self,
        alpha: int,
        res_blocks: int,
        data_format='channels_last'
    ):
        super().__init__()

        check._is_positive_int('alpha', alpha)
        check._is_positive_int('res_blocks', res_blocks)

        self.alpha = alpha
        self.res_blocks = res_blocks
        self.data_format = data_format
        self.is_channels_first = dataformat.is_channels_first_keras(
            data_format)
        self.cat_axis = dataformat.get_channel_axis_keras(data_format)

    def build(self, input_size):
        def conv(out_channels, kernel_size, stride):
            return ConvBnAct(
                out_channels, kernel_size, stride,
                data_format=self.data_format)

        alpha = self.alpha
        beta = self.alpha * 2

        self.conv1 = conv(beta, 3, 2)
        self.conv2 = conv(alpha, 1, 1)
        self.conv3 = conv(alpha, 1, 1)
        self.res4 = ResBlock(
            blocks=self.res_blocks,
            data_format=self.data_format)
        self.conv5 = conv(alpha, 1, 1)
        self.concat = L.Concatenate(axis=self.cat_axis)
        self.conv6 = conv(beta, 1, 1)

    def call(self, inputs, training=False):
        x1 = self.conv1(inputs, training)

        # split
        x2 = self.conv2(x1, training)
        x3 = self.conv3(x1, training)
        x4 = self.res4(x3, training)
        x5 = self.conv5(x4, training)

        # merge
        x5 = self.concat([x5, x2])
        x6 = self.conv6(x5, training)
        return x6

    def get_config(self):
        return {
            'alpha': self.alpha,
            'res_blocks': self.res_blocks,
            'data_format': self.data_format
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class ImplicitVariable(L.Layer):
    """ Creates an variable which is used as an implicit input variable to the
    computational graph of a neural network.

    This is can be added or multiplied against outputs of different layers and
    can be applied at any stage of the network.

    This layer basically wraps a weight variable of a fixed shape. Every `call`
    just returns the weight variable directly.

    :param shape: The shape of the variable, excluding the batch dimension. It
        is assumed that batch is always the first dimension.
    :param float init_mean: The mean of the random-normally inited weights.
        Recommended to use `0.0` for addition and `1.0` for multiplication.
    """

    def __init__(self, shape, init_mean=0.0):
        super().__init__()
        self.shape = shape
        self.init_mean = init_mean
        initializer = tf.keras.initializers.RandomNormal(
            mean=init_mean, stddev=0.02)
        self.values = self.add_weight(
            name='values',
            shape=(1, *shape),
            initializer=initializer,
            trainable=True
        )

    def call(self, x):
        """ Returns the implicit weights of this layer.

        :param x: Required for `Layer` subclass. Recommend to just pass `None`.
        :returns: The implicit variable.
        """
        return self.values

    def get_config(self):
        return {
            'shape': tuple(self.values.shape[1:]),
            'init_mean': self.init_mean
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class FPN(L.Layer):
    """ Feature Pyramid Network layer.

    Performs convolution and then naive upsampling of a primary input.
    The upsampling is configured so that the primary input can be converted to
    have the same height and width as the secondary input.

    Afterwards, the upsampled primary input and secondary input are
    concatenated. This variable is forwarded through two parallel convolutional
    paths after which the results are then merged again via concatenation.

    .. warning::
        The input for upsampling must have a height and width that divide
        evenly into the height and width of the secondary input.

    See: `FPN arXiv Paper <https://arxiv.org/abs/1612.03144>`_

    :param int channels: The output channels of this layer.
    :param str data_format: Must be 'channels_first' or 'channels_last'.
        The 'channels_first' option requires a GPU.
    """

    def __init__(self, channels: int, data_format='channels_last'):
        super().__init__()

        check._is_positive_int('channels', channels)

        self.channels = channels
        self.data_format = data_format
        self.cat_axis = dataformat.get_channel_axis_keras(data_format)

    def build(self, input_size):
        def conv(out_channels, kernel_size, stride):
            return ConvBnAct(
                out_channels, kernel_size, stride,
                activation='mish',
                data_format=self.data_format
            )

        self.upsample = Upsample(data_format=self.data_format)

        c = self.channels
        self.conv1 = conv(c, 1, 1)
        self.conv2 = conv(c, 1, 1)
        self.concat2 = L.Concatenate(axis=self.cat_axis)
        self.conv3 = conv(c, 1, 1)
        self.conv4 = conv(c, 1, 1)

        # We don't want blocks configurable since we aren't using a shortcut.
        self.plain5 = ResBlock(
            blocks=3,
            shortcut=False,
            data_format=self.data_format)

        self.concat5 = L.Concatenate(axis=self.cat_axis)
        self.conv6 = conv(c, 1, 1)

    def call(self, inputs, downsample, training=False):
        x1 = self.conv1(inputs, training=training)
        up = self.upsample(x1, K.shape(downsample), training=training)

        x2 = self.conv2(downsample, training=training)
        x2 = self.concat2([x2, up])
        x3 = self.conv3(x2, training=training)

        # split
        x4 = self.conv4(x3, training=training)
        x5 = self.plain5(x3, training=training)

        # Merge
        x5 = self.concat5([x5, x4])
        x6 = self.conv6(x5, training=training)
        return x6

    def get_config(self):
        return {
            'channels': self.channels,
            'data_format': self.data_format
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class PAN(L.Layer):
    """ Path Aggregation Network layer.

    Downsamples the primary input and then concatenates with an output from a
    previous :class:`FPN` layer.

    See: `PAN arXiv Paper <https://arxiv.org/abs/1803.01534>`_

    :param int channels: The number of output channels for this layer.
    :param str data_format: Must be 'channels_first' or 'channels_last'.
        The 'channels_first' option requires a GPU.
    """

    def __init__(self, channels, data_format='channels_last'):
        super().__init__()

        check._is_positive_int('channels', channels)

        self.channels = channels
        self.data_format = data_format
        self.cat_axis = dataformat.get_channel_axis_keras(data_format)

        # We don't care about the input shape, build now.
        self._build_immediately()

    def _build_immediately(self):
        def conv(out_channels, kernel_size, stride):
            return ConvBnAct(
                out_channels, kernel_size, (stride, stride),
                activation='mish',
                data_format=self.data_format
            )

        c = self.channels
        self.conv1 = conv(c, 3, 2)
        self.concat1 = L.Concatenate(axis=self.cat_axis)
        self.conv2 = conv(c, 1, 1)
        self.conv3 = conv(c, 1, 1)

        # We don't want blocks configurable since we aren't using a shortcut.
        self.plain4 = ResBlock(
            blocks=3,
            shortcut=False,
            data_format=self.data_format)

        self.concat4 = L.Concatenate(axis=self.cat_axis)
        self.conv5 = conv(c, 1, 1)

    def call(self, inputs, fpn_in, training=False):
        x1 = self.conv1(inputs, training=training)
        x1 = self.concat1([x1, fpn_in])
        x2 = self.conv2(x1, training=training)

        # split
        x3 = self.conv3(x2, training=training)
        x4 = self.plain4(x2, training=training)

        # merge
        x4 = self.concat4([x4, x3])
        x5 = self.conv5(x4, training=training)
        return x5

    def get_config(self):
        return {
            'channels': self.channels,
            'data_format': self.data_format
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
