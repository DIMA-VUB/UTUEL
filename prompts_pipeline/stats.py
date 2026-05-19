"""
stats.py
RunStats — collects per-run counters and timings.
"""

import time
from dataclasses import dataclass, field


@dataclass
class RunStats:
    run_id: str
    started_at: float = field(default_factory=time.time)

    total_requests: int = 0
    skipped_resumed: int = 0
    dispatched: int = 0
    succeeded: int = 0
    failed: int = 0
    total_batches: int = 0

    _finished_at: float | None = None

    # --- mutators ---

    def record_batch(self, batch_size: int, successes: int) -> None:
        self.total_batches += 1
        self.dispatched += batch_size
        self.succeeded += successes
        self.failed += batch_size - successes

    def finish(self) -> None:
        self._finished_at = time.time()

    # --- derived ---

    @property
    def elapsed_seconds(self) -> float:
        end = self._finished_at or time.time()
        return round(end - self.started_at, 2)

    @property
    def throughput(self) -> float:
        """Successful responses per second."""
        elapsed = self.elapsed_seconds
        return round(self.succeeded / elapsed, 2) if elapsed > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "run_id":           self.run_id,
            "elapsed_seconds":  self.elapsed_seconds,
            "total_requests":   self.total_requests,
            "skipped_resumed":  self.skipped_resumed,
            "dispatched":       self.dispatched,
            "succeeded":        self.succeeded,
            "failed":           self.failed,
            "total_batches":    self.total_batches,
            "throughput_rps":   self.throughput,
        }

    def __str__(self) -> str:
        d = self.to_dict()
        lines = [
            "┌─── Run Stats ──────────────────────",
            f"│  run_id          {d['run_id']}",
            f"│  elapsed         {d['elapsed_seconds']}s",
            f"│  total requests  {d['total_requests']}",
            f"│  skipped/resumed {d['skipped_resumed']}",
            f"│  dispatched      {d['dispatched']}",
            f"│  succeeded       {d['succeeded']}",
            f"│  failed          {d['failed']}",
            f"│  batches sent    {d['total_batches']}",
            f"│  throughput      {d['throughput_rps']} resp/s",
            "└────────────────────────────────────",
        ]
        return "\n".join(lines)
