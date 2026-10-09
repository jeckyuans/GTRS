import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
CACHES = [ROOT / 'deployment/closedloop_speed/navsim_gtrs_dense_challenge/simscale_gtrs_dense/inference_cache.py']


@pytest.fixture(params=CACHES, ids=['gtrs'])
def cache(request):
    spec = importlib.util.spec_from_file_location('tested_inference_cache', request.param)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.InferenceTensorCache(max_entries=2)


def test_weight_updates_load_and_dtype_changes_invalidate(cache):
    module = torch.nn.Linear(3, 2).eval()
    original_keys = set(module.state_dict())
    inputs = torch.ones(2, 3)
    calls = []

    def get():
        def compute():
            calls.append(True)
            return module(inputs.to(module.weight))
        return cache.get(module, list(module.parameters()), 2, compute)

    with torch.inference_mode():
        first = get()
        assert get() is first and len(calls) == 1
        module.weight.add_(1)
        updated = get()
        assert not torch.equal(first, updated) and len(calls) == 2
        state = {key: value.clone() for key, value in module.state_dict().items()}
        state['bias'].add_(2)
        module.load_state_dict(state)
        loaded = get()
        torch.testing.assert_close(loaded, updated + 2, rtol=0, atol=1e-6)
        module.double()
        assert get().dtype == torch.float64 and len(calls) == 4
    assert set(module.state_dict()) == original_keys


def test_gradients_training_and_autocast_bypass(cache):
    module = torch.nn.Linear(3, 2).eval()
    inputs = torch.ones(2, 3)
    tensors = list(module.parameters())
    compute = lambda: module(inputs)
    with torch.no_grad():
        assert cache.get(module, tensors, 2, compute) is cache.get(module, tensors, 2, compute)
    output = cache.get(module, tensors, 2, compute)
    output.sum().backward()
    assert module.weight.grad is not None
    module.train()
    with torch.no_grad():
        assert cache.get(module, tensors, 2, compute) is not cache.get(module, tensors, 2, compute)
    module.eval()
    with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
        assert cache.get(module, tensors, 2, compute).dtype == torch.bfloat16
        assert not cache._values


def test_cache_is_bounded_and_tracks_batch_shape(cache):
    module = torch.nn.Identity().eval()
    tensor = torch.nn.Parameter(torch.arange(3.0), requires_grad=False)
    with torch.no_grad():
        for batch_size in (1, 2, 1, 4):
            result = cache.get(module, [tensor], batch_size, lambda: tensor.repeat(batch_size, 1))
            assert result.shape == (batch_size, 3)
    assert list(cache._values) == [1, 4]
    cache.clear()
    assert not cache._values and cache._signature is None
