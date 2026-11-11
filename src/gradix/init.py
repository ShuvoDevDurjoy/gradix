import numpy as np

def uniform(shape, left_bound = 0.0, right_bound = 1.0, dtype = np.float32): 
    return np.random.uniform(left_bound, right_bound, size=shape).astype(dtype)

def kaiming_uniform(shape, a = np.sqrt(5.0)):
    fan_in, fan_out = _get_fan_in_and_fan_out(shape)
    gain = np.sqrt(2.0 / (1.0 + a ** 2))

    std = gain / np.sqrt(fan_in)

    weight_bound  = np.sqrt(3.0) * std

    return uniform(shape, -weight_bound, weight_bound)


def _get_fan_in_and_fan_out(shape): 
    assert isinstance(shape, (tuple, list))

    if len(shape) < 2: 
        raise ValueError('Can not implement fan in and fan out for shape less than 2')
    fan_in = shape[1]
    fan_out = shape[0]

    receptive_field_size = 1

    if len(shape) > 2: 
        for s in shape[2:]: 
            receptive_field_size *= s
    fan_in *= receptive_field_size
    fan_out *= receptive_field_size

    return fan_in, fan_out