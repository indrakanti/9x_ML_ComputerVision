from __future__ import annotations

import sys
from pathlib import Path

MODULE_PYTHON = Path(__file__).resolve().parents[1] / "python"
sys.path.insert(0, str(MODULE_PYTHON))
