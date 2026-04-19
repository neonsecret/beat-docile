"""[SHIM] Backward-compatibility re-export — new code should import from llm_client directly."""

from .llm_client import complete, get_client

__all__ = ["complete", "get_client"]
