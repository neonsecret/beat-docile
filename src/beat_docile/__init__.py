"""beat_docile — Few-shot Claude pipeline for the DocILE KILE/LIR benchmark.

Current best: v2_ensemble = 46.48% KILE AP / 50.77% LIR F1 (500 val docs).
Target: GraphDoc SOTA at 71.25% KILE / 75.93% LIR.

See KNOWLEDGE_BASE.md for full architecture notes, experiment verdicts, and
operational lessons. See README.md for quick-start commands and module map.

Public API:
    extract_documents   — async; runs Claude extraction over a sequence of docs
    merge_predictions   — merge multiple prediction dicts (ensemble)
    load_predictions    — load a predictions JSON from disk
    save_predictions    — save a predictions dict to disk
    run_eval            — run DocILE benchmark evaluation
    print_scores        — format and print eval scores; returns score dict
    iter_pages          — yield PageContext for each page of a Document
    Field               — docile Field dataclass (bbox, page, fieldtype, score, ...)
"""

from docile.dataset import Field

from .data import iter_pages
from .ensemble import load_predictions, merge_predictions, save_predictions
from .eval import print_scores, run_eval
from .extract import extract_documents

__all__ = [
    "Field",
    "extract_documents",
    "iter_pages",
    "load_predictions",
    "merge_predictions",
    "print_scores",
    "run_eval",
    "save_predictions",
]
