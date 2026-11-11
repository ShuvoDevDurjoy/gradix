from gradix.tensor import Tensor
import numpy as np

from gradix.nn.modules.module import Module
from gradix.init import _get_fan_in_and_fan_out, kaiming_uniform, uniform
from gradix.nn import functional as F


class Conv2D(Module): 
    def __init__(self, in_channels, out_channels, kernel, stride = 1, padding = 0, use_bias = False): 
        super().__init__()

        if not isinstance(in_channels, int) or in_channels <= 0:
            raise ValueError("in_channels must be a positive integer")
        if not isinstance(out_channels, int) or out_channels <= 0:
            raise ValueError("out_channels must be a positive integer")
        if not isinstance(kernel, int) or kernel <= 0:
            raise ValueError("kernel_size must be a positive integer")
        if not isinstance(stride, int) or stride <= 0:
            raise ValueError("stride must be a positive integer")
        if not isinstance(padding, int) or padding < 0:
            raise ValueError("padding must be a non-negative integer")
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.bias = None
        self.weight = Tensor(
            kaiming_uniform((out_channels, in_channels, kernel, kernel), np.sqrt(5.0)), 
            requires_grad = True, 
            dtype = np.float32
        )
        if use_bias: 
            fan_in, _ = _get_fan_in_and_fan_out(self.weight.shape)
            bound = (1.0 / np.sqrt(fan_in)) if fan_in > 0 else 0
            self.bias = Tensor(uniform((self.out_channels, ), -bound, bound), requires_grad=True, dtype=np.float32)

    def parameters(self): 
        return [self.weight] if self.bias is None else [self.weight, self.bias]

    def forward(self, x): 
        if not isinstance(x, Tensor) or x.ndim != 4: 
            raise ValueError(f'Conv2D expects a tensor of shape (batch, channel, height, width) but found one of shape: {x.shape}')
        
        batch_size, in_channels, height, width = x.shape

        if in_channels != self.in_channels: 
            raise ValueError(f'Expected {self.in_channels} input channels but received: {in_channels}')


        output_height = (height + 2 * self.padding - self.kernel) // self.stride + 1
        output_width = (width + 2 * self.padding - self.kernel) // self.stride + 1

        flattened_weight = self.weight.reshape(self.out_channels, -1)
        patches = F.im2td(x, self.kernel, self.stride, self.padding)

        product = patches @ flattened_weight.T

        output = product.reshape(batch_size, output_height, output_width, self.out_channels).transpose(0, 3, 1, 2)

        if self.bias is not None: 
            output = output + self.bias.reshape(1, self.out_channels, 1, 1)
        
        return output