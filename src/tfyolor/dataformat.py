
from typing import Tuple


_KERAS_CHANNELS_FIRST = 'channels_first'
_KERAS_CHANNELS_LAST = 'channels_last'
_VALID_KERAS_FORMATS = frozenset((_KERAS_CHANNELS_FIRST, _KERAS_CHANNELS_LAST))


def _validate_format_keras(data_format: str):
    if data_format not in _VALID_KERAS_FORMATS:
        error_message = (
            f'Invalid data format "{data_format}". ' +
            'Expected one of: {_VALID_KERAS_FORMATS}'
        )
        raise ValueError(error_message)


def is_channels_first_keras(data_format: str):
    """ Returns true if the data format is 'channels_first' otherwise False.

    :raises: :code:`ValueError` if `data_format` is invalid.
    """
    _validate_format_keras(data_format)
    return data_format == _KERAS_CHANNELS_FIRST


def get_channel_axis_keras(data_format: str):
    """ Converts 'channels_first' or 'channels_last' to an int axis.

    :raises: :code:`ValueError` if `data_format` is invalid.
    """
    _validate_format_keras(data_format)
    return 1 if data_format == _KERAS_CHANNELS_FIRST else -1


def unpack_shape_to_nchw(
    shape,
    is_channels_first: bool
) -> Tuple[int, int, int, int]:
    """ Unpacks a shape to NCHW format.

    :param shape: The shape to unpack. Must be indexable and contain 4 ints.
    :param bool is_channels_first: True if channels are the first non-batch
        dimension. False if it is the last dimension.
    :returns: The iterable unpacked in NCHW order.
    """
    if is_channels_first:
        batch_size, channels, height, width = (
            shape[0], shape[1], shape[2], shape[3])
    else:
        batch_size, height, width, channels = (
            shape[0], shape[1], shape[2], shape[3])

    return batch_size, channels, height, width
