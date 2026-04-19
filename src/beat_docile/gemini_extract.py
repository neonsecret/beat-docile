"""[RESEARCH-BURIED] Gemini 3 Flash extraction for DocILE — 4th voice for v2 ensemble.

Status: RESEARCH-BURIED — 41.56% KILE solo vs Sonnet 44.95%; 4-way ensemble regressed
3-way by -0.65pp. thinking_budget=0 required to avoid output truncation, but disabling
thinking makes Flash strictly weaker than Sonnet at affordable budgets.
See KNOWLEDGE_BASE.md §6.4 for full results and budget analysis.

Preserved for retry with thinking-enabled Flash or a cheaper-thinking Gemini variant.
Uses google-genai SDK with Vertex AI backend (project set via VERTEX_PROJECT_ID env var).
env: BD_GEMINI_MODEL=gemini-3-flash-preview (default), BD_GEMINI_WORKERS=6
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
from collections.abc import Sequence

from docile.dataset import Field

from .data import PageContext, iter_pages
from .extract import (
    _SYSTEM,
    _SYSTEM_TARGETED,
    _TARGETED_FIELDS,
    _parse_response,
    _words_to_prompt,
)

_GEMINI_MODEL = os.environ.get("BD_GEMINI_MODEL", "gemini-3-flash-preview")
_TEMPERATURE = float(os.environ.get("BD_TEMPERATURE", "1.0"))
_MAX_WORKERS = int(os.environ.get("BD_GEMINI_WORKERS", "6"))

_VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT_ID", "")
_VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "global")

_client = None
_semaphore: asyncio.Semaphore | None = None


def _get_client():
    global _client
    if _client is None:
        import google.genai

        _client = google.genai.Client(
            vertexai=True,
            project=_VERTEX_PROJECT,
            location=_VERTEX_LOCATION,
        )
    return _client


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(_MAX_WORKERS)
    return _semaphore


def _page_to_bytes(page: PageContext) -> bytes:
    buf = io.BytesIO()
    page.image.save(buf, format="PNG")
    return buf.getvalue()


def _build_contents(
    page: PageContext,
    few_shot_examples=None,
    task_text: str = "Extract all fields from this invoice page. Return JSON only.",
) -> list:
    """Build Gemini contents list: optional few-shot history + query user turn."""
    from google.genai import types

    contents = []

    if few_shot_examples:
        for ex in few_shot_examples:
            img_bytes = base64.standard_b64decode(ex.image_b64)
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                        types.Part.from_text(
                            text=(
                                f"[Document words]\n{ex.words_layout}\n\n"
                                "[Task]\nExtract all fields from this invoice page. Return JSON only."
                            )
                        ),
                    ],
                )
            )
            contents.append(
                types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=ex.gold_json)],
                )
            )

    img_bytes = _page_to_bytes(page)
    words_layout = _words_to_prompt(page.words)
    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                types.Part.from_text(
                    text=f"[Document words]\n{words_layout}\n\n[Task]\n{task_text}"
                ),
            ],
        )
    )
    return contents


async def _gemini_complete(
    contents: list,
    system: str,
    model: str = _GEMINI_MODEL,
    max_tokens: int = 8192,
    temperature: float = _TEMPERATURE,
) -> str:
    """Call Gemini via run_in_executor; retry on transient errors.

    thinking_budget=0: disables extended thinking to avoid burning token budget.
    Gemini 3 Flash is a reasoning model; without this, thinking tokens (~8-13K/call)
    consume max_output_tokens and truncate the actual JSON output.
    """
    import google.api_core.exceptions as gexc
    from google.genai import types

    client = _get_client()
    config = types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=max_tokens,
        temperature=temperature,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )

    max_attempts = 5
    delay = 2.0
    last_exc = None
    for attempt in range(max_attempts):
        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                ),
            )
            return response.text or ""
        except (gexc.ResourceExhausted, gexc.ServiceUnavailable, gexc.InternalServerError) as e:
            last_exc = e
            await asyncio.sleep(min(delay * (2**attempt), 60))
        except Exception:
            raise

    raise RuntimeError(f"Gemini {model} failed after {max_attempts} attempts: {last_exc}")


async def extract_page_gemini(
    page: PageContext,
    model: str = _GEMINI_MODEL,
    few_shot_examples=None,
) -> tuple[list[Field], list[Field]]:
    """Extract KILE + LIR from one page via Gemini. Returns (kile_fields, lir_fields)."""
    if not page.words:
        return [], []

    contents = _build_contents(page, few_shot_examples=few_shot_examples)
    sem = _get_semaphore()
    async with sem:
        raw = await _gemini_complete(contents, system=_SYSTEM, model=model)

    return _parse_response(raw, page.words, page.page_index)


async def extract_page_targeted_gemini(
    page: PageContext,
    model: str = _GEMINI_MODEL,
) -> list[Field]:
    """Targeted second pass for financial/registration fields only."""
    if not page.words:
        return []

    task = (
        "Look carefully for any banking, registration, or tax-rate detail fields. Return JSON only."
    )
    contents = _build_contents(page, task_text=task)
    sem = _get_semaphore()
    async with sem:
        raw = await _gemini_complete(
            contents, system=_SYSTEM_TARGETED, model=model, max_tokens=1024
        )

    kile, lir = _parse_response(raw, page.words, page.page_index)
    return [f for f in kile + lir if f.fieldtype in _TARGETED_FIELDS]


async def extract_documents(
    docs: Sequence,
    model: str = _GEMINI_MODEL,
    train_index: dict[int, list[str]] | None = None,
    targeted_pass: bool = True,
    cluster_override: dict[str, int] | None = None,
) -> tuple[dict[str, list[Field]], dict[str, list[Field]]]:
    """Extract fields for a sequence of Document objects via Gemini.

    Returns (kile_preds, lir_preds) both as {docid: [Field, ...]}.
    Mirrors extract.extract_documents() interface.
    """
    from .fewshot import FewShotExample, load_few_shot_examples

    def _cluster_id(doc) -> int | None:
        if cluster_override and doc.docid in cluster_override:
            return cluster_override[doc.docid]
        try:
            return doc.annotation.cluster_id
        except Exception:
            return None

    kile_preds: dict[str, list[Field]] = {}
    lir_preds: dict[str, list[Field]] = {}

    # Build few-shot cache keyed by cluster_id
    few_shot_cache: dict[int, list[FewShotExample]] = {}
    if train_index is not None:
        cluster_ids = [cid for doc in docs if (cid := _cluster_id(doc)) is not None]
        unique_cids = list(set(cluster_ids))
        if unique_cids:
            examples_by_cluster = load_few_shot_examples(
                unique_cids, train_index, max_per_cluster=1
            )
            few_shot_cache.update(examples_by_cluster)

    async def process_doc(doc) -> None:
        kile_preds[doc.docid] = []
        lir_preds[doc.docid] = []

        cid = _cluster_id(doc)
        fs_examples = few_shot_cache.get(cid) if cid is not None else None

        pages = list(iter_pages(doc))

        main_tasks = [
            extract_page_gemini(page, model=model, few_shot_examples=fs_examples) for page in pages
        ]
        targeted_tasks = (
            [extract_page_targeted_gemini(page, model=model) for page in pages]
            if targeted_pass
            else []
        )

        all_results = await asyncio.gather(*main_tasks, *targeted_tasks)
        n = len(pages)
        for kile, lir in all_results[:n]:
            kile_preds[doc.docid].extend(kile)
            lir_preds[doc.docid].extend(lir)
        for fields in all_results[n:]:
            for f in fields:
                if f.line_item_id is not None:
                    lir_preds[doc.docid].append(f)
                else:
                    kile_preds[doc.docid].append(f)

    await asyncio.gather(*[process_doc(doc) for doc in docs])
    return kile_preds, lir_preds
