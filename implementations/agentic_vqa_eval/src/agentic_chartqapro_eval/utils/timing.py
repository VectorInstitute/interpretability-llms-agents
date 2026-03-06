"""Timing context manager."""

import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TimingResult:
    start_ts: str = ""
    end_ts: str = ""
    elapsed_ms: float = 0.0


@contextmanager
def timed():
    result = TimingResult()
    result.start_ts = iso_now()
    t0 = time.time()
    yield result
    result.elapsed_ms = (time.time() - t0) * 1000.0
    result.end_ts = iso_now()
