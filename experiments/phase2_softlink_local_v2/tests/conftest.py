from __future__ import annotations

import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
SRC = PACKAGE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
