from __future__ import annotations

import os
import resource


GIB = 1024**3


class MemoryLimitReached(RuntimeError):
    pass


class MemoryWatchdog:
    def __init__(
        self,
        stop_rss_gib: float = 3,
        hard_rss_gib: float = 4,
        min_available_gib: float = 2,
    ):
        self.stop_rss = int(stop_rss_gib * GIB)
        self.hard_rss = int(hard_rss_gib * GIB)
        self.min_available = int(min_available_gib * GIB)
        self.peak_rss = 0

    @staticmethod
    def rss_bytes() -> int:
        try:
            with open("/proc/self/statm", encoding="ascii") as handle:
                pages = int(handle.read().split()[1])
            return pages * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError):
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024

    @staticmethod
    def available_bytes() -> int:
        try:
            with open("/proc/meminfo", encoding="ascii") as handle:
                for line in handle:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            pass
        return GIB * 1024

    def check(self) -> tuple[int, int]:
        rss = self.rss_bytes()
        available = self.available_bytes()
        self.peak_rss = max(self.peak_rss, rss)
        if rss >= self.hard_rss:
            raise MemoryLimitReached(f"RSS reached hard limit: {rss / GIB:.2f} GiB")
        if rss >= self.stop_rss:
            raise MemoryLimitReached(f"RSS reached checkpoint limit: {rss / GIB:.2f} GiB")
        if available <= self.min_available:
            raise MemoryLimitReached(
                f"System available RAM fell to {available / GIB:.2f} GiB"
            )
        return rss, available
