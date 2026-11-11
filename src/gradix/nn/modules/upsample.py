
from gradix.nn.modules.module import Module


class Upsample(Module): 
    def __init__(self, scale_factor: int = 1, mode = 'nearest'): 
        self.scale_factor = scale_factor
        self.mode = mode

    def _nearest_interpolate(self, x): 
        pass

    def _uniform_interpolate(self, x): 
        pass

    def parameters(self): 
        return []

    def forward(self, x): 
        return x