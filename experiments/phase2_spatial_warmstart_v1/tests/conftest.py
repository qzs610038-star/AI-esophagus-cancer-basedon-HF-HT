from __future__ import annotations

import os
from pathlib import Path
import sys


PACKAGE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_DIR.parents[1]

if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep test runs read-only outside pytest's own temporary directories.
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
