import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L

from . import dataformat


class YoloLayer(L.Layer):
    """ Applies YOLO logic to convert the input to bounding boxes.

    .. note::
        When the training param is `True`, this layer short circuits and
        provides outputs in an unprocessed format for use in loss
        calculation. When `False`, the outputs are fully converted to
        bounding boxes.

    :param anchors: A 3-element list where each element is a sublist
        containing an anchor box definition -- (width, height).
    :type anchors: list, tuple
    :param int n_classes: The number of classes for this detector.
    :param int stride: The width of a square in the original image to which a
        single prediction by this layer should map.
    :param str data_format: Must be 'channels_first' or 'channels_last'.
        The 'channels_first' option requires a GPU.
    """

    def __init__(
        self,
        anchors,
        n_classes,
        stride,
        data_format='channels_last'
    ):
        super().__init__()
        self.stride = stride
        self.n_anchors = len(anchors)
        self.n_classes = n_classes
        self.n_outputs = 5 + n_classes
        self.data_format = data_format
        self.is_channels_first = dataformat.is_channels_first_keras(
            data_format)

        self.anchors = tf.constant(anchors, dtype=K.floatx())
        self.anchor_vec = self.anchors / self.stride
        self.anchor_wh = tf.reshape(
            self.anchor_vec, (1, self.n_anchors, 1, 1, 2))

    def call(self, inputs, training=False):
        r""" Reorganizes the inputs to be converted to bounding boxes.

        .. note::
            The final channel of the output will have the format:
            :math:`(x, y, w, h, obj, p_{\text{cls0}}, p_{\text{cls1}}, ...)`

        .. warning::
            In training mode (i.e. :code:`training=True`), the results are not
            fully computed -- this is delegated to the loss calculation
            instead. The training output is a single tensor of shape
            :math:`(batch, n_{\text{anchors}}, height, width, n_{\text{out}})`.

            In evaluation mode, there are two outputs. The first output will be
            a tensor of shape :math:`(batch, n_{\text{pred}}, n_{\text{out}})`.
            The second output will be the same as the training output. This can
            be used for loss calculations during evaluation.

            Where :math:`n_{\text{out}} = n_{\text{classes}} + 5` and the 5
            comes from x, y, width, height, and object-ness.

        :param inputs: The input tensor.
        :param bool training: Whether this layer is training or not.
        :returns: One of the outputs as described above.
        """
        batch_size, _, height, width = dataformat.unpack_shape_to_nchw(
            K.shape(inputs), self.is_channels_first)

        x = tf.reshape(inputs, (batch_size, self.n_anchors,
                       self.n_outputs, height, width))
        x = tf.transpose(x, perm=[0, 1, 3, 4, 2])
        if training:
            return x

        xrange = tf.range(width)
        yrange = tf.range(height)
        YX = tf.meshgrid(yrange, xrange, indexing='ij')
        XY = YX[::-1]
        XY = tf.stack(XY, 2)
        XY = tf.reshape(XY, (1, 1, height, width, 2))

        sig = tf.sigmoid(x)
        grid = tf.cast(XY, dtype=sig.dtype)
        anchor_wh = tf.cast(self.anchor_wh, dtype=sig.dtype)

        xy = self.stride * (sig[..., :2] * 2 - 0.5 + grid)
        wh = self.stride * ((sig[..., 2:4] * 2) ** 2 * anchor_wh)
        p_cls = sig[..., 4:self.n_outputs]
        out = tf.concat((xy, wh, p_cls), axis=-1)
        return tf.reshape(out, (batch_size, -1, self.n_outputs)), x

    def _anchors_as_list(self):
        return self.anchors.numpy().tolist()

    def get_config(self):
        return {
            'stride': self.stride,
            'anchors': self._anchors_as_list(),
            'n_classes': self.n_classes,
            'data_format': self.data_format
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
