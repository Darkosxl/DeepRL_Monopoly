from __future__ import annotations

import os
import sys

try:
    import psutil
except ImportError:  # pragma: no cover - exercised only in minimal installs
    psutil = None

try:
    import resource
except ImportError:  # pragma: no cover - unavailable on Windows
    resource = None


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
        if psutil is not None:
            return psutil.Process().memory_info().rss
        try:
            with open("/proc/self/statm", encoding="ascii") as handle:
                pages = int(handle.read().split()[1])
            return pages * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError):
            if resource is None:
                raise MemoryLimitReached("Process RSS is unavailable")
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return peak if sys.platform == "darwin" else peak * 1024

    @staticmethod
    def available_bytes() -> int:
        if psutil is not None:
            return psutil.virtual_memory().available
        try:
            with open("/proc/meminfo", encoding="ascii") as handle:
                for line in handle:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            pass
        try:
            return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, AttributeError):
            raise MemoryLimitReached("System available RAM is unavailable")

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
