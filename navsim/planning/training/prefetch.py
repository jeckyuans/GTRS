"""Bounded sliding-window batch prefetch for training.

`PrefetchDataLoader` is a `torch.utils.data.DataLoader` whose iterator keeps a bounded queue of already
collated batches filled by a background thread. The first `next()` of every epoch blocks until the queue
is full, so the GPU only waits while the window fills; afterwards the decoders stay ahead as long as their
throughput exceeds the step rate.

Host memory per rank is bounded by
    queue_depth * batch_bytes            (the queue, capped by ram_budget_gb)
  + num_workers * batch_bytes            (one batch in flight per worker process; prefetch_factor is forced to 1)
  + 1 batch held by Lightning's own one-batch lookahead.
The queue depth shrinks, never grows, if the largest batch seen so far times the depth would exceed
`ram_budget_gb`.

`decode_threads > 1` decodes the samples of a batch concurrently inside each worker process (gzip / file
reads release the GIL), which raises decoder throughput without spawning more processes.

Environment overrides (explicit constructor arguments win):
    NAVSIM_PREFETCH_QUEUE           1 to enable in `build_train_dataloader` (default 0)
    NAVSIM_PREFETCH_DEPTH           batches per rank (default 8)
    NAVSIM_PREFETCH_RAM_GB          host-RAM budget for the queue per rank in GB (default 20; <= 0 disables the cap)
    NAVSIM_PREFETCH_DECODE_THREADS  threads per worker process decoding one batch's samples (default 1)
"""

import logging
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

ENV_ENABLE = "NAVSIM_PREFETCH_QUEUE"
ENV_DEPTH = "NAVSIM_PREFETCH_DEPTH"
ENV_RAM_GB = "NAVSIM_PREFETCH_RAM_GB"
ENV_DECODE_THREADS = "NAVSIM_PREFETCH_DECODE_THREADS"

DEFAULT_DEPTH = 8
DEFAULT_RAM_GB = 20.0
_GB = 1024**3


def _env(name: str, cast, default):
    value = os.environ.get(name, "")
    return cast(value) if value.strip() else default


def prefetch_enabled(explicit: Optional[bool] = None) -> bool:
    if explicit is not None:
        return bool(explicit)
    return _env(ENV_ENABLE, lambda v: v.strip().lower() in ("1", "true", "yes", "on"), False)


def batch_nbytes(obj: Any) -> int:
    """Bytes held by the tensors / arrays of a (nested) batch."""
    if isinstance(obj, torch.Tensor):
        return obj.element_size() * obj.nelement()
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    if isinstance(obj, dict):
        return sum(batch_nbytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(batch_nbytes(v) for v in obj)
    return 0


class ThreadedFetchDataset(Dataset):
    """Wraps a map-style dataset so that the DataLoader fetches a batch's samples on a thread pool."""

    def __init__(self, dataset: Dataset, threads: int):
        self.dataset = dataset
        self.threads = threads
        self._pool: Optional[ThreadPoolExecutor] = None

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __getitems__(self, indices):
        if self._pool is None:
            self._pool = ThreadPoolExecutor(self.threads, thread_name_prefix="navsim-decode")
        return list(self._pool.map(self.dataset.__getitem__, indices))

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state["_pool"] = None
        return state

    def __getattr__(self, name: str):
        if name == "dataset":
            raise AttributeError(name)
        return getattr(self.dataset, name)


@dataclass
class PrefetchStats:
    depth: int = 0
    capacity: int = 0
    max_batch_bytes: int = 0
    startup_wait_s: float = 0.0
    filled_at_start: int = 0
    produced: int = 0
    consumed: int = 0
    peak_reserved: int = 0
    consumer_waits: int = 0
    consumer_wait_s: float = 0.0


class _State:
    """Shared between the consumer iterator and the producer thread; holds no reference to the iterator."""

    def __init__(self, depth: int, budget_bytes: int):
        self.cond = threading.Condition()
        self.buffer: deque = deque()
        self.fetching = False
        self.done = False
        self.stop = False
        self.error: Optional[BaseException] = None
        self.budget_bytes = budget_bytes
        self.stats = PrefetchStats(depth=depth, capacity=depth)

    def reserved(self) -> int:
        return len(self.buffer) + int(self.fetching)

    def fit_capacity(self, nbytes: int) -> None:
        stats = self.stats
        if nbytes <= stats.max_batch_bytes:
            return
        stats.max_batch_bytes = nbytes
        if self.budget_bytes <= 0:
            return
        capacity = max(1, min(stats.depth, self.budget_bytes // nbytes))
        if capacity < stats.capacity:
            logger.warning(
                "prefetch queue: %d batches of %.2f GB exceed the %.1f GB budget, shrinking depth %d -> %d",
                stats.capacity, nbytes / _GB, self.budget_bytes / _GB, stats.capacity, capacity,
            )
            stats.capacity = capacity


def _produce(state: _State, source: Iterator) -> None:
    try:
        while True:
            with state.cond:
                while not state.stop and state.reserved() >= state.stats.capacity:
                    state.cond.wait()
                if state.stop:
                    return
                state.fetching = True
                state.stats.peak_reserved = max(state.stats.peak_reserved, state.reserved())
            try:
                batch = next(source)
            except StopIteration:
                with state.cond:
                    state.fetching = False
                    state.done = True
                    state.cond.notify_all()
                return
            nbytes = batch_nbytes(batch)
            with state.cond:
                state.fetching = False
                state.buffer.append(batch)
                state.stats.produced += 1
                state.fit_capacity(nbytes)
                state.cond.notify_all()
    except BaseException as error:  # re-raised in the consumer
        with state.cond:
            state.fetching = False
            state.error = error
            state.done = True
            state.cond.notify_all()


class _PrefetchIterator:
    def __init__(self, source: Iterator, depth: int, budget_bytes: int, length: Optional[int], log_every: int):
        self._state = _State(depth, budget_bytes)
        self._length = length
        self._log_every = log_every
        self._started = False
        self._thread = threading.Thread(
            target=_produce, args=(self._state, source), name="navsim-prefetch", daemon=True
        )
        self._thread.start()

    @property
    def stats(self) -> PrefetchStats:
        return self._state.stats

    def __iter__(self) -> "_PrefetchIterator":
        return self

    def __len__(self) -> int:
        if self._length is None:
            raise TypeError("underlying loader has no length")
        return self._length

    def __next__(self):
        state, stats = self._state, self._state.stats
        with state.cond:
            if not self._started:
                start = time.monotonic()
                while not state.done and len(state.buffer) < stats.capacity:
                    state.cond.wait()
                stats.startup_wait_s = time.monotonic() - start
                stats.filled_at_start = len(state.buffer)
                self._started = True
                logger.info(
                    "prefetch queue filled %d/%d batches (%.2f GB each) in %.1f s",
                    stats.filled_at_start, stats.capacity, stats.max_batch_bytes / _GB, stats.startup_wait_s,
                )
            if not state.buffer and not state.done:
                start = time.monotonic()
                while not state.buffer and not state.done:
                    state.cond.wait()
                stats.consumer_waits += 1
                stats.consumer_wait_s += time.monotonic() - start
            if state.buffer:
                batch = state.buffer.popleft()
                stats.consumed += 1
                state.cond.notify_all()
                if self._log_every > 0 and stats.consumed % self._log_every == 0:
                    logger.info(
                        "prefetch queue: %d consumed, %d/%d queued, %d waits totalling %.1f s",
                        stats.consumed, len(state.buffer), stats.capacity, stats.consumer_waits, stats.consumer_wait_s,
                    )
                return batch
            error, state.error = state.error, None
        self.close()
        if error is not None:
            raise error
        raise StopIteration

    def close(self) -> None:
        state = self._state
        with state.cond:
            state.stop = True
            state.buffer.clear()
            state.cond.notify_all()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=1.0)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class PrefetchDataLoader(DataLoader):
    """DataLoader whose iterator serves batches from a bounded, pre-filled queue (see module docstring).

    Lightning re-instantiates DataLoader subclasses from their public attributes to inject a
    DistributedSampler, so the extra arguments are stored under their own names.
    """

    def __init__(
        self,
        dataset: Dataset,
        queue_depth: Optional[int] = None,
        ram_budget_gb: Optional[float] = None,
        decode_threads: Optional[int] = None,
        log_every: int = 200,
        **kwargs,
    ):
        queue_depth = _env(ENV_DEPTH, int, DEFAULT_DEPTH) if queue_depth is None else int(queue_depth)
        ram_budget_gb = _env(ENV_RAM_GB, float, DEFAULT_RAM_GB) if ram_budget_gb is None else float(ram_budget_gb)
        decode_threads = _env(ENV_DECODE_THREADS, int, 1) if decode_threads is None else int(decode_threads)
        if queue_depth < 1:
            raise ValueError(f"queue_depth must be >= 1, got {queue_depth}")
        if decode_threads > 1 and not isinstance(dataset, ThreadedFetchDataset):
            dataset = ThreadedFetchDataset(dataset, decode_threads)
        if kwargs.get("num_workers", 0) > 0 and kwargs.get("prefetch_factor") not in (None, 1):
            logger.info("prefetch queue: prefetch_factor %s -> 1, the queue does the prefetching",
                        kwargs["prefetch_factor"])
            kwargs["prefetch_factor"] = 1
        super().__init__(dataset, **kwargs)
        self.queue_depth = queue_depth
        self.ram_budget_gb = ram_budget_gb
        self.decode_threads = decode_threads
        self.log_every = log_every
        self.last_iterator: Optional[_PrefetchIterator] = None

    def __iter__(self) -> _PrefetchIterator:
        # The source iterator is created on the calling thread: worker start-up installs signal handlers
        # and the pin-memory thread binds to the current CUDA device.
        source = super().__iter__()
        try:
            length = len(self)
        except TypeError:
            length = None
        iterator = _PrefetchIterator(
            source, self.queue_depth, int(self.ram_budget_gb * _GB), length, self.log_every
        )
        self.last_iterator = iterator
        return iterator


def build_train_dataloader(
    dataset: Dataset,
    enabled: Optional[bool] = None,
    queue_depth: Optional[int] = None,
    ram_budget_gb: Optional[float] = None,
    decode_threads: Optional[int] = None,
    **kwargs,
) -> DataLoader:
    """Returns a PrefetchDataLoader if enabled (argument, else NAVSIM_PREFETCH_QUEUE), else a plain DataLoader."""
    if not prefetch_enabled(enabled):
        return DataLoader(dataset, **kwargs)
    loader = PrefetchDataLoader(
        dataset, queue_depth=queue_depth, ram_budget_gb=ram_budget_gb, decode_threads=decode_threads, **kwargs
    )
    logger.info(
        "prefetch queue on: depth %d batches/rank, RAM budget %.1f GB/rank, %d workers x %d decode threads",
        loader.queue_depth, loader.ram_budget_gb, loader.num_workers, loader.decode_threads,
    )
    return loader
