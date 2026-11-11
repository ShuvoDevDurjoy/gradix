import numpy as np
from gradix.nn.functional import log_softmax
from gradix.tensor import Tensor

def cross_entropy(logits, targets): 
    assert isinstance(logits, Tensor)
    assert isinstance(targets, Tensor)
    if logits.ndim != 2: 
        raise ValueError(f'cross_entropy expects logits of shape (batch, classes) but found: {logits.shape}')
    batch_size, num_classes = logits.shape
    class_indices = targets.data.astype(int).reshape(-1)
    if class_indices.size != batch_size: 
        raise ValueError(f'Expected {batch_size} targets but found: {class_indices.size}')
    if (class_indices < 0).any() or (class_indices >= num_classes).any(): 
        raise IndexError(f'target class out of range [0, {num_classes})')
    log_probs = log_softmax(logits)
    row_offsets = num_classes * logits.xp.arange(batch_size)
    corrected_log_probs = log_probs.gather(row_offsets + class_indices)
    return -corrected_log_probs.mean()