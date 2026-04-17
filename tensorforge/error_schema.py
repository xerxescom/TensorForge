"""
Structured error schema helpers for subprocess-heavy benchmarks.
"""

from __future__ import annotations

from typing import Any


def classify_error_type(message: str) -> str:
    msg = (message or "").lower()

    if any(k in msg for k in ("timeout", "timed out", "timeoutexpired")):
        return "timeout"
    if any(
        k in msg
        for k in (
            "out of memory",
            "cuda out of memory",
            "cudnn_status_alloc_failed",
            "std::bad_alloc",
            " oom",
            "killed",
        )
    ):
        return "oom"
    if any(
        k in msg
        for k in (
            "no module named",
            "modulenotfounderror",
            "importerror",
            "filenotfounderror",
            "command not found",
            "not installed",
            "dependency",
        )
    ):
        return "dependency_missing"
    if any(
        k in msg
        for k in (
            "download",
            "snapshot_download",
            "connection",
            "network",
            "dns",
            "proxy",
            "ssl",
            "http error",
            "502",
            "503",
            "504",
        )
    ):
        return "download_failed"
    return "runtime_error"


def short_trace(error_text: str, limit: int = 240) -> str:
    text = (error_text or "").strip()
    if not text:
        return ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    best = lines[-1] if lines else text
    best = " ".join(best.split())
    return best[:limit]


def structured_error(
    *,
    error: Any,
    error_stage: str,
    trace_text: str = "",
    error_type: str | None = None,
) -> dict[str, str]:
    error_str = str(error)
    merged_text = "\n".join(part for part in [error_str, trace_text] if part)
    resolved_type = error_type or classify_error_type(merged_text)
    return {
        "error": error_str,
        "error_type": resolved_type,
        "error_stage": error_stage,
        "short_trace": short_trace(trace_text or error_str),
    }


def ensure_structured_error(
    payload: dict[str, Any],
    *,
    error_stage: str,
    trace_key: str = "details",
) -> dict[str, Any]:
    if not payload.get("error"):
        return payload
    if payload.get("error_type") and payload.get("error_stage") and payload.get("short_trace"):
        return payload

    payload.update(
        structured_error(
            error=payload.get("error", "unknown_error"),
            error_stage=payload.get("error_stage", error_stage),
            trace_text=str(payload.get(trace_key, "")),
            error_type=payload.get("error_type"),
        )
    )
    return payload

