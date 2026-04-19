"""[ACTIVE] Config / environment variable loader for beat_docile.

Status: ACTIVE — used in current best (v2_ensemble).
See KNOWLEDGE_BASE.md §3 for the architecture map.

Reads from .env.local if present. Ref: EVAL_SPEC §7.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

_root = Path(__file__).parent.parent.parent
_env = _root / ".env.local"
if _env.exists():
    load_dotenv(_env)

VERTEX_PROJECT_ID: str = os.environ.get("VERTEX_PROJECT_ID", "")
VERTEX_LOCATION: str = os.environ.get("VERTEX_LOCATION", "us-east5")

_data_default = Path.home() / "beat_docile" / "data"
DATA_ROOT: Path = Path(os.environ.get("DATA_ROOT", str(_data_default)))

DEFAULT_MODEL: str = os.environ.get("DEFAULT_MODEL", "claude-sonnet-4-6")
