import numpy as np
from gradix.tensor import Tensor

def _validate_int_tuple(*dims, func_name = ''): 
    dims = tuple(dims)
    if len(dims) == 1 and isinstance(dims, (tuple, list)): 
        dims = tuple(dims[0])

    if any((not isinstance(d, int) or d < 0) for d in dims): 
        raise ValueError('Expected Positive int in : {func_name}')
    return dims


def sigmoid(x: Tensor): 
    return 1 / (1 + (-x).exp())

def softmax(x: Tensor): 
    max_x = x.max()
    x = x - max_x
    ex = x.exp()
    sum_ex = ex.sum()
    return ex / sum_ex

def log_softmax(x: Tensor, axis = 1): 
    max_x = x.max(axis = axis, keepdims = True)
    shifted = x - max_x
    sum_ex = shifted.exp().sum(axis = axis, keepdims = True)
    return shifted - sum_ex.log()


def im2td(x, kernel, stride = 1, padding = 0): 
    if not isinstance(x, Tensor) or x.ndim != 4:
        raise ValueError("unfold2d expects a Tensor shaped (batch, channels, height, width)")
    if not isinstance(kernel, int) or kernel <= 0:
        raise ValueError("kernel_size must be a positive integer")
    if not isinstance(stride, int) or stride <= 0:
        raise ValueError("stride must be a positive integer")
    if not isinstance(padding, int) or padding < 0:
        raise ValueError("padding must be a non-negative integer")

    batch_size, channels, height, width = x.shape
    output_height = (height + 2 * padding - kernel) // stride + 1
    output_width = (width + 2 * padding - kernel) // stride + 1

    if output_height <= 0 or output_width <= 0:
        raise ValueError("kernel_size cannot exceed the padded input dimensions")
    

    batch, out_y, out_x, channel, kernel_y, kernel_x = np.meshgrid(
        np.arange(batch_size), 
        np.arange(output_height), 
        np.arange(output_width), 
        np.arange(channels), 
        np.arange(kernel), 
        np.arange(kernel), 
        indexing='ij'
    )    

    out_y = out_y * stride + kernel_y - padding
    out_x = out_x * stride + kernel_x - padding

    valid = (out_y >= 0) & (out_x >= 0) & (out_y < height) & (out_x < width)

    clipped_x = np.clip(out_x, 0, width - 1)
    clipped_y = np.clip(out_y, 0, height - 1)

    indices = ((batch * channels + channel) * height + clipped_y) * width + clipped_x

    gathered_out = x.gather(indices)

    out = gathered_out * Tensor(valid.astype(np.float32))

    return out.reshape(
        batch_size * output_height * output_width, 
        channels * kernel * kernel
    )

def repeat(x: Tensor, *dims): 
    dims = _validate_int_tuple(dims)
    if len(dims) < x.ndim: 
        raise ValueError(f"repeat expects at least {x.ndim} but got {len(dims)}")
    padded_shape = (1, ) * (len(dims) - x.ndim) + x.shape
    reshape_shape = tuple(
        value 
        for size in padded_shape
        for value in (1, size)
    )
    x = x.reshape(reshape_shape)
    broadcast_shape = tuple(
        value 
        for repeat, shape in zip(dims, padded_shape)
        for value in (repeat, shape)
    )
    x = x.broadcast_to(broadcast_shape)
    output_shape = tuple(repeat * size for repeat, size in zip(dims, padded_shape))
    return x.reshape(output_shape)


def Upsample(x, scale_factor = 2): 
    assert isinstance(scale_factor, int)
    assert isinstance(x, Tensor)
    assert x.ndim == 4

    b, c, h, w = x.shape
    x = x.reshape(b, c, h, 1, w, 1)
    x = repeat(x, 1, 1, 1, scale_factor, 1, scale_factor)
    print(x.shape)
    return x.reshape(b, c, h * scale_factor, w * scale_factor)
