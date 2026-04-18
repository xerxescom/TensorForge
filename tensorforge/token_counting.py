"""
Token counting helpers with clear precision/source metadata.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


def heuristic_token_count(text: str) -> int:
    """Fallback heuristic used only when exact counting is unavailable."""
    return int(len(text.split()) * 1.3) if text else 0


@lru_cache(maxsize=32)
def _load_tiktoken_encoding(model_name: str):
    import tiktoken  # type: ignore[import-not-found]

    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


@lru_cache(maxsize=16)
def _load_hf_tokenizer(model_id_or_path: str):
    from transformers import AutoTokenizer  # type: ignore[import-not-found]

    return AutoTokenizer.from_pretrained(model_id_or_path, local_files_only=True)


def local_exact_token_count(
    text: str,
    *,
    model_name: str = "",
    local_model_path: str | None = None,
) -> tuple[int | None, str | None]:
    """
    Best-effort local exact token count.

    Returns:
      (token_count, source) where source is one of:
      - "local_tokenizer:tiktoken"
      - "local_tokenizer:transformers"
      - None (if exact local counting is unavailable)
    """
    if not text:
        return 0, "local_tokenizer:tiktoken"

    try:
        enc = _load_tiktoken_encoding(model_name or "cl100k_base")
        return len(enc.encode(text)), "local_tokenizer:tiktoken"
    except Exception:
        pass

    hf_candidates: list[str] = []
    if local_model_path:
        p = Path(local_model_path)
        if p.exists():
            hf_candidates.append(str(p))
    if model_name:
        hf_candidates.append(model_name)

    for candidate in hf_candidates:
        try:
            tok = _load_hf_tokenizer(candidate)
            return len(tok.encode(text, add_special_tokens=False)), "local_tokenizer:transformers"
        except Exception:
            continue

    return None, None
