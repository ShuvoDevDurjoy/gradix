import math
import numpy as np
from . import *

class Tensor:
    float32 = float32
    float64 = float64
    float16 = float16
    int32 = int32
    int64 = int64
    bool_ = bool_
    def __init__(self, data: np.ndarray, _childs: tuple = (), requires_grad = False, _op = None, device = DEVICE, dtype = None, name = None): 
        if device == 'cuda' and cp is None:
            raise RuntimeError("CUDA support requires installing the 'gpu' extra")
        self.device = DEVICE if device not in ['cpu', 'cuda'] else device
        self.xp = cp if self.device == 'cuda' else np
        self.dtype = dtype
        self.requires_grad = requires_grad
        self._init_data(data)
        self._backward = lambda: None
        self.grad = self.xp.zeros_like(self.data) if requires_grad else None
        self._prevs = set(_childs)
        self._op = _op
        self._name = None
        self.name = name

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = None if value is None else str(value)

    def _init_data(self, data):
        if self.dtype is None: 
            self.dtype = getattr(data, 'dtype', self.float32)
            if self.requires_grad and np.dtype(self.dtype).kind != 'f':
                self.dtype = self.float32
        if isinstance(data, self.xp.ndarray):
            self.data = data.astype(self.dtype, copy = False)
        else: 
            self.data = self.xp.asarray(data, dtype = self.dtype) 

    def _scatter_add(self, target, key, values):
        if self.device == 'cuda':
            if cupyx is None:
                raise RuntimeError("CUDA support requires installing the 'gpu' extra")
            cupyx.scatter_add(target, key, values)
            return
        np.add.at(target, key, values)

    def _const(self, other):
        if isinstance(other, Tensor):
            return other
        return Tensor(other, device = self.device, dtype = self.dtype)

    def _norm_axes(self, axis):
        if axis is None:
            return tuple(range(self.ndim))
        axes = axis if isinstance(axis, tuple) else (axis,)
        return tuple(ax % self.ndim for ax in axes)

    @property
    def T(self): 
        return self.transpose()
    
    def transpose(self, *axes):
        if not axes:
            axes = tuple(reversed(range(self.data.ndim)))
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        else:
            axes = tuple(axes)
        axes = tuple(ax % self.ndim for ax in axes)
        out = Tensor(self.xp.transpose(self.data, axes), (self,), self.requires_grad, _op='transpose', device = self.device)
        def _backward(): 
            inverse_axes = tuple(int(i) for i in np.argsort(axes))
            self._accumulate_grad(self.xp.transpose(out.grad, inverse_axes))
        out._backward = _backward
        return out

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = Tensor(self.data.reshape(*shape), (self,), self.requires_grad, _op='reshape', device = self.device)
        def _backward():
            self._accumulate_grad(out.grad.reshape(self.shape))
        out._backward = _backward
        return out

    def gather(self, indices):
        indices = self.xp.asarray(indices)
        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError("gather indices must be integers")
        indices = indices.astype(self.xp.intp, copy=False)
        if indices.size and (int(indices.min()) < 0 or int(indices.max()) >= self.data.size):
            raise IndexError("gather index is outside the flattened Tensor")

        out = Tensor(self.data.reshape(-1)[indices], (self,), self.requires_grad, _op='gather', device = self.device)

        def _backward():
            self._scatter_add(self.grad.reshape(-1), indices.reshape(-1), out.grad.reshape(-1))

        out._backward = _backward
        return out

    def broadcast_to(self, shape: tuple): 
        if len(shape) < self.ndim: 
            raise ValueError(f"Can not broadcast to fewer dimension from {self.ndim} to {len(shape)}")

        for source, target in zip((1,) * (len(shape) - self.ndim) + self.shape, shape): 
            if source != target and source != 1: 
                raise ValueError(f"Can not Broadcast shape: from {self.shape} to {shape}")

        out = Tensor(self.xp.broadcast_to(self.data, shape), (self, ), requires_grad= self.requires_grad, _op ='broadcast_to', device = self.device)
        def _backward(): 
            self._accumulate_grad(out.grad)
        out._backward = _backward
        return out

    def __getitem__(self, key):
        out = Tensor(self.data[key], (self, ), requires_grad= self.requires_grad, _op='getitem', device = self.device)
        def _backward(): 
            self._scatter_add(self.grad, key, out.grad)
        out._backward = _backward
        return out

    def requires_grad_(self, requires_grad: bool):
        self.requires_grad = requires_grad
        self.grad = self.xp.zeros_like(self.data) if self.requires_grad else None
        return self

    def __add__(self, other):
        other = self._const(other)
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data + other.data, (self, other), requires_grad=requires_grad, _op='+', device = self.device)
        def _backward(): 
            self._accumulate_grad(out.grad)
            other._accumulate_grad(out.grad)
        out._backward = _backward
        return out

    def __mul__(self, other): 
        other = self._const(other)
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data * other.data, (self, other), requires_grad=requires_grad, _op='*', device = self.device)
        def _backward(): 
            self._accumulate_grad(other.data * out.grad)
            other._accumulate_grad(self.data * out.grad)
        out._backward = _backward
        return out

    def __matmul__(self, other): 
        other = self._const(other)
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data @ other.data, (self, other), requires_grad=requires_grad, _op='@', device = self.device)
        def _backward(): 
            a, b, grad = self.data, other.data, out.grad
            if b.ndim == 1:
                b, grad = b[:, None], self.xp.expand_dims(grad, -1)
            if a.ndim == 1:
                a, grad = a[None, :], self.xp.expand_dims(grad, -2)
            grad_a = grad @ b.swapaxes(-1, -2)
            grad_b = a.swapaxes(-1, -2) @ grad
            if self.ndim == 1:
                grad_a = grad_a[..., 0, :]
            if other.ndim == 1:
                grad_b = grad_b[..., 0]
            self._accumulate_grad(grad_a)
            other._accumulate_grad(grad_b)
        out._backward = _backward
        return out

    def __pow__(self, value): 
        assert isinstance(value, (int, float))
        out = Tensor(self.xp.power(self.data, value), (self, ), requires_grad = self.requires_grad, _op=f'^{value}', device = self.device)
        def _backward(): 
            self._accumulate_grad(value * self.xp.power(self.data, value - 1) * out.grad)
        out._backward = _backward
        return out

    def reciprocal(self): 
        out = Tensor(1 / self.data, (self, ), requires_grad=self.requires_grad, _op='reciprocal', device = self.device)
        def _backward(): 
            self._accumulate_grad(-1 * out.grad / (self.data ** 2))
        out._backward = _backward
        return out 

    def __div__(self, other): 
        other = self._const(other)
        return self * other.reciprocal()

    def __truediv__(self, other):
        return self.__div__(other)

    def leaky_relu(self, lvalue = 0.1):
        cdata = self.data.copy()
        lindexes = self.data <= 0
        cdata[lindexes] *= lvalue
        out = Tensor(cdata, (self, ), requires_grad=self.requires_grad, _op=f'leaky_relu: {lvalue}', device = self.device)
        def _backward():
            grad = self.xp.ones_like(self.data)
            grad[self.data <= 0] = lvalue
            self._accumulate_grad(out.grad * grad)
        out._backward = _backward
        return out

    def relu(self): 
        return self.leaky_relu(0)

    def exp(self):
        out = Tensor(self.xp.exp(self.data), (self,), requires_grad=self.requires_grad, _op='exp', device = self.device)
        def _backward():
            self._accumulate_grad(out.grad * out.data)
        out._backward = _backward
        return out

    def log(self): 
        out = Tensor(self.xp.log(self.data), (self, ), requires_grad = self.requires_grad, _op='log', device = self.device)
        def _backward(): 
            self._accumulate_grad(out.grad / self.data)
        out._backward = _backward
        return out

    def tanh(self): 
        out = Tensor(self.xp.tanh(self.data), (self, ), requires_grad= self.requires_grad, _op='tanh', device = self.device)
        def _backward(): 
            self._accumulate_grad(out.grad * (1 - out.data ** 2))
        out._backward = _backward
        return out

    def sigmoid(self): 
        out = Tensor(0.5 * (1 + self.xp.tanh(self.data * 0.5)), (self, ), requires_grad= self.requires_grad, _op='sigmoid', device = self.device)
        def _backward(): 
            self._accumulate_grad(out.grad * out.data * (1 - out.data))
        out._backward = _backward
        return out

    def _extremum(self, fn, axis, keepdims, _op):
        out = Tensor(fn(self.data, axis = axis, keepdims = keepdims), (self, ), requires_grad=self.requires_grad, _op=_op, device = self.device)
        def _backward(): 
            kept = fn(self.data, axis = axis, keepdims = True)
            grad = out.grad.reshape(kept.shape)
            mask = (self.data == kept).astype(self.data.dtype)
            mask = mask / mask.sum(axis = axis, keepdims = True)
            self._accumulate_grad(grad * mask)
        out._backward = _backward
        return out

    def max(self, axis = None, keepdims = False): 
        return self._extremum(self.xp.max, axis, keepdims, 'max')

    def min(self, axis = None, keepdims = False): 
        return self._extremum(self.xp.min, axis, keepdims, 'min')

    def sum(self, axis = None, keepdims = False):
        out = Tensor(self.xp.sum( self.data, axis = axis, keepdims = keepdims),(self,),self.requires_grad, _op='sum', device = self.device)

        def _backward():
            axes = self._norm_axes(axis)
            kept_shape = tuple(1 if i in axes else s for i, s in enumerate(self.shape))
            grad = out.grad.reshape(kept_shape)
            self._accumulate_grad(self.xp.broadcast_to(grad, self.shape))

        out._backward = _backward
        return out

    def mean(self, axis = None, keepdims = False):
        count = math.prod(self.shape[ax] for ax in self._norm_axes(axis))
        return self.sum(axis = axis, keepdims = keepdims) / count

    def __neg__(self): 
        return self * -1
    def __sub__(self, other): 
        return self + (-other)
    def __radd__(self, other): 
        return self + other
    def __rsub__(self, other): 
        return (-self) + other
    def __rmul__(self, other): 
        return self * other
    def __rtruediv__(self, other):
        return other * self.reciprocal()

    def zero_grad(self):
        for node in self._build_topo():
            if node.requires_grad:
                node.grad = self.xp.zeros_like(node.data)
    
    def backward(self): 
        if not self.requires_grad:
            raise RuntimeError("backward() called on a Tensor that does not require grad")
        nodes = list(self._build_topo())
        for node in nodes:
            if node._prevs:
                node.grad = self.xp.zeros_like(node.data)
        self.grad = self.xp.ones_like(self.data)
        for node in nodes:
            node._backward() 

    def _build_topo(self): 
        nodes, visited = [], set()
        def build_topo(node): 
            if (node not in visited) and (node.requires_grad): 
                visited.add(node)
                for child in node._prevs: 
                    build_topo(child)
                nodes.append(node)
        build_topo(self)
        return reversed(nodes)
    
    def _accumulate_grad(self, grad):
        if not self.requires_grad: 
            return
        while grad.ndim > self.grad.ndim:
            grad = grad.sum(axis=0)
        axes = tuple(
            i
            for i, (gdim, tdim) in enumerate(zip(grad.shape, self.grad.shape))
            if tdim == 1 and gdim != 1
        )
        if axes:
            grad = grad.sum(axis=axes, keepdims=True)
        self.grad += grad

    @property
    def shape(self): 
        return self.data.shape

    @property
    def ndim(self): 
        return self.data.ndim

    def __repr__(self): 
        return f'Tensor({self.data}, dtype: {self.data.dtype}, Shape: {list(self.data.shape)})'