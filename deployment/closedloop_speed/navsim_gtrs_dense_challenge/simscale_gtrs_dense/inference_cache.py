"""Bounded caches for read-only vocabulary tensors during inference."""
from collections import OrderedDict

import torch


class InferenceTensorCache:
    """Keep derived tensors out of checkpoints and invalidate on tensor changes."""

    def __init__(self, max_entries=4):
        self.max_entries = max_entries
        self._signature = None
        self._values = OrderedDict()

    def clear(self):
        self._signature = None
        self._values.clear()

    def get(self, module, tensors, key, compute):
        # Training, gradients and autocast must retain their original execution.
        if (torch.is_grad_enabled() or any(child.training for child in module.modules())
                or torch.is_autocast_enabled() or torch.is_autocast_enabled('cpu')):
            self.clear()
            return compute()
        try:
            signature = tuple(
                (id(tensor), tensor.data_ptr(), tensor._version, tensor.device,
                 tensor.dtype, tuple(tensor.shape))
                for tensor in tensors
            )
        except RuntimeError:
            # Inference-created parameters have no version counter.
            self.clear()
            return compute()
        if signature != self._signature:
            self.clear()
            self._signature = signature
        if key not in self._values:
            self._values[key] = compute()
            if len(self._values) > self.max_entries:
                self._values.popitem(last=False)
        else:
            self._values.move_to_end(key)
        return self._values[key]

