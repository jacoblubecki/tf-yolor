
from tensorflow_addons.activations import mish as tfa_mish
from tensorflow.keras.layers import Activation, LeakyReLU


def create_activation(string: str):
    """ Converts a string to a Keras activation.

    Supports keras defaults, but adds default-constructed options for:

    - :code:`'mish'` => :code:`tensorflow_addons.activations.mish`
    - :code:`'leaky'` => :code:`LeakyReLU(alpha=0.1)`

    :param str string: A Keras activation string or one of ('mish', 'leaky').
    :returns: The appropriate Keras `Activation`, or `None` if the input is
        `None`.
    """

    if string is None:
        return None

    assert isinstance(string, str), 'Input must be a string (or None).'
    if string == 'mish':
        return tfa_mish
    elif string == 'leaky':
        return LeakyReLU(alpha=0.1)
    else:
        return Activation(string)
