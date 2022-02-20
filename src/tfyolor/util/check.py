
def _is_positive_int(name: str, v):
    if not isinstance(v, int) or v <= 0:
        msg = f'Arg `{name}` expected to be a positive int (was: {v}).'
        raise ValueError(msg)


def _is_tuple_of_positive_int(name, v, required_len):
    if len(v) != required_len:
        raise ValueError(
            f'Int-tuple for ${name} may only have {required_len} values.')
    elif any(map(lambda x: not isinstance(x, int), v)):
        raise ValueError('Tuple must only contain positive ints.')
    elif any(map(lambda x: x <= 0, v)):
        msg = f'Arg `{name}` must only contain positive ints (was {v}).'
        raise ValueError(msg)


def _is_positive_int_or_tuple(name: str, value, length: int):
    if isinstance(value, int):
        _is_positive_int(name, value)
    elif isinstance(value, (tuple, list)):
        _is_tuple_of_positive_int(name, value, 2)
    else:
        raise ValueError('Expected int or 2-tuple of ints.')
