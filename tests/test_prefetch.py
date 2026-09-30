"""CPU-only checks of the bounded prefetch queue with fake slow samples: python -m pytest tests/test_prefetch.py"""

import threading
import time

import pytest
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from navsim.planning.training.prefetch import (
    ENV_ENABLE,
    PrefetchDataLoader,
    ThreadedFetchDataset,
    build_train_dataloader,
)


class SlowDataset(Dataset):
    def __init__(self, n: int, delay: float, numel: int = 256, fail_at: int = -1):
        self.n, self.delay, self.numel, self.fail_at = n, delay, numel, fail_at

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        time.sleep(self.delay)
        if idx == self.fail_at:
            raise ValueError(f"bad sample {idx}")
        return {"x": torch.full((self.numel,), float(idx))}, idx


def _loader(dataset, depth, **kwargs):
    return PrefetchDataLoader(dataset, queue_depth=depth, ram_budget_gb=0, batch_size=1, num_workers=0, **kwargs)


def test_queue_fills_before_first_step():
    depth, delay = 4, 0.05
    it = iter(_loader(SlowDataset(10, delay), depth))
    start = time.monotonic()
    first = next(it)
    assert time.monotonic() - start >= 0.9 * depth * delay
    assert it.stats.filled_at_start == depth
    assert first[1].item() == 0


def test_fast_consumer_eventually_blocks():
    depth, n = 3, 12
    it = iter(_loader(SlowDataset(n, 0.03), depth))
    got = [batch[1].item() for batch in it]
    assert got == list(range(n))
    assert it.stats.filled_at_start == depth
    assert it.stats.consumer_waits > 0
    assert it.stats.peak_reserved <= depth


def test_slow_consumer_stays_capped():
    depth, n = 3, 15
    it = iter(_loader(SlowDataset(n, 0.001), depth))
    for batch in it:
        state = it._state
        with state.cond:
            assert state.reserved() <= depth
            assert state.stats.produced - state.stats.consumed <= depth
        time.sleep(0.02)
    assert it.stats.peak_reserved <= depth
    assert it.stats.consumer_waits == 0


def test_ram_budget_shrinks_depth():
    numel = 256 * 1024  # 1 MiB of float32 per batch
    budget_gb = 2.5 * 1024**2 / 1024**3
    loader = PrefetchDataLoader(
        SlowDataset(8, 0.001, numel=numel), queue_depth=8, ram_budget_gb=budget_gb, batch_size=1, num_workers=0
    )
    it = iter(loader)
    assert len(list(it)) == 8
    assert it.stats.capacity == 2
    assert it.stats.peak_reserved <= 2


def test_error_reaches_consumer():
    it = iter(_loader(SlowDataset(10, 0.0, fail_at=5), 2))
    with pytest.raises(ValueError, match="bad sample 5"):
        for _ in it:
            pass


def test_early_exit_stops_producer():
    it = iter(_loader(SlowDataset(100, 0.001), 2))
    next(it)
    thread = it._thread
    del it
    thread.join(timeout=2.0)
    assert not thread.is_alive()


def test_workers_and_decode_threads_keep_order():
    n = 24
    loader = PrefetchDataLoader(
        SlowDataset(n, 0.01), queue_depth=4, ram_budget_gb=0, decode_threads=4,
        batch_size=4, num_workers=2, prefetch_factor=2,
    )
    assert loader.prefetch_factor == 1
    assert isinstance(loader.dataset, ThreadedFetchDataset)
    got = torch.cat([batch[1] for batch in loader]).tolist()
    assert got == list(range(n))


def test_lightning_injects_distributed_sampler():
    from pytorch_lightning.trainer.states import RunningStage
    from pytorch_lightning.utilities.data import _update_dataloader

    loader = PrefetchDataLoader(SlowDataset(16, 0.0), queue_depth=5, ram_budget_gb=3.0, decode_threads=2,
                                batch_size=2, num_workers=0)
    sampler = DistributedSampler(loader.dataset, num_replicas=2, rank=1, shuffle=False)
    rebuilt = _update_dataloader(loader, sampler, mode=RunningStage.TRAINING)
    assert isinstance(rebuilt, PrefetchDataLoader)
    assert (rebuilt.queue_depth, rebuilt.ram_budget_gb, rebuilt.decode_threads) == (5, 3.0, 2)
    assert not isinstance(rebuilt.dataset.dataset, ThreadedFetchDataset)
    assert torch.cat([batch[1] for batch in rebuilt]).tolist() == list(range(1, 16, 2))


def test_env_toggle(monkeypatch):
    monkeypatch.delenv(ENV_ENABLE, raising=False)
    assert type(build_train_dataloader(SlowDataset(4, 0.0), batch_size=2)) is DataLoader
    monkeypatch.setenv(ENV_ENABLE, "1")
    assert isinstance(build_train_dataloader(SlowDataset(4, 0.0), batch_size=2), PrefetchDataLoader)
    assert type(build_train_dataloader(SlowDataset(4, 0.0), enabled=False, batch_size=2)) is DataLoader


def test_no_leaked_threads():
    before = {t for t in threading.enumerate() if t.name == "navsim-prefetch"}
    for _ in _loader(SlowDataset(6, 0.0), 2):
        pass
    time.sleep(0.1)
    after = {t for t in threading.enumerate() if t.name == "navsim-prefetch"}
    assert after <= before
