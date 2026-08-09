"""Laptop resource guards used by local training and remote handoff."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil

import torch

from .config import ResourceConfig

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


GIB = 1024**3


class ResourceLimitReached(RuntimeError):
    def __init__(self, resource: str, message: str):
        super().__init__(message)
        self.resource = resource


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    rss_bytes: int
    available_ram_bytes: int
    free_disk_bytes: int
    free_vram_bytes: int | None


class ResourceWatchdog:
    def __init__(self, root: str | Path, config: ResourceConfig | None = None):
        self.root = Path(root)
        self.config = config or ResourceConfig()
        self.peak_rss = 0

    @staticmethod
    def _rss() -> int:
        if psutil is not None:
            return psutil.Process().memory_info().rss
        with open("/proc/self/statm", encoding="ascii") as handle:
            return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")

    @staticmethod
    def _available() -> int:
        if psutil is not None:
            return psutil.virtual_memory().available
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
        raise ResourceLimitReached("ram", "System available RAM is unavailable")

    def snapshot(self) -> ResourceSnapshot:
        rss = self._rss()
        self.peak_rss = max(self.peak_rss, rss)
        free_vram = torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else None
        return ResourceSnapshot(
            rss_bytes=rss,
            available_ram_bytes=self._available(),
            free_disk_bytes=shutil.disk_usage(self.root).free,
            free_vram_bytes=free_vram,
        )

    def check(self, *, raise_soft: bool = True) -> ResourceSnapshot:
        value = self.snapshot()
        limits = self.config
        if value.rss_bytes >= limits.hard_rss_gib * GIB:
            raise ResourceLimitReached("hard_rss", f"RSS reached {value.rss_bytes / GIB:.2f} GiB")
        if raise_soft and value.rss_bytes >= limits.soft_rss_gib * GIB:
            raise ResourceLimitReached("soft_rss", f"RSS reached checkpoint limit {value.rss_bytes / GIB:.2f} GiB")
        if value.available_ram_bytes <= limits.min_available_ram_gib * GIB:
            raise ResourceLimitReached("ram", f"Available RAM fell to {value.available_ram_bytes / GIB:.2f} GiB")
        if value.free_disk_bytes <= limits.min_free_disk_gib * GIB:
            raise ResourceLimitReached("disk", f"Free disk fell to {value.free_disk_bytes / GIB:.2f} GiB")
        if value.free_vram_bytes is not None and value.free_vram_bytes <= limits.reserved_vram_gib * GIB:
            raise ResourceLimitReached("vram", f"Free VRAM fell to {value.free_vram_bytes / GIB:.2f} GiB")
        return value
