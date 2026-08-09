from __future__ import annotations

from pathlib import Path
import shutil
import sys
import tempfile

import pytest


sys.dont_write_bytecode = True


@pytest.fixture
def bench_tmp():
    root = Path(__file__).resolve().parents[1] / ".test-runtime"
    root.mkdir(exist_ok=True)
    directory = Path(tempfile.mkdtemp(dir=root))
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)

