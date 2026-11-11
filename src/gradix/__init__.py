import numpy as np

try:
	import cupy as cp
	import cupyx
except ImportError:
	cp = None
	cupyx = None

GPU_AVAILABLE = cp is not None and cp.cuda.is_available() and cp.cuda.runtime.getDeviceCount() > 0
DEVICE = 'cuda' if GPU_AVAILABLE else 'cpu'

#dtypes
float32 = np.float32
float64 = np.float64
float16 = np.float16
int32   = np.int32
int64   = np.int64
bool_   = np.bool_

from .tensor import Tensor
from . import nn

__all__ = [
	"Tensor",
	"nn",
	"GPU_AVAILABLE",
	"DEVICE",
	"float32",
	"float64",
	"float16",
	"int32",
	"int64",
	"bool_",
]