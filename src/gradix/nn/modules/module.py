from gradix.tensor import Tensor
class Module: 
    def zero_grad(self): 
        for p in self.parameters(): 
            p.grad = p.xp.zeros_like(p.data)

    def __call__(self, *args, **kwargs): 
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs): 
        raise NotImplementedError

    def parameters(self): 
        return []