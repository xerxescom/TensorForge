"""Subprocess worker for ASRBenchmark."""

from __future__ import annotations

import argparse
import json
import time
from difflib import SequenceMatcher

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TensorForge ASR benchmark worker")
    parser.add_argument("--model", required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--audio-files-json", default="[]")
    parser.add_argument("--ground-truths-json", default="[]")
    return parser.parse_args()


def _levenshtein_wer(ref_words: list[str], hyp_words: list[str]) -> tuple[float, int, int, int]:
    n = len(ref_words)
    m = len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    op = [[""] * (m + 1) for _ in range(n + 1)]

    for r in range(1, n + 1):
        dp[r][0] = r
        op[r][0] = "D"
    for c in range(1, m + 1):
        dp[0][c] = c
        op[0][c] = "I"

    for r in range(1, n + 1):
        for c in range(1, m + 1):
            if ref_words[r - 1] == hyp_words[c - 1]:
                dp[r][c] = dp[r - 1][c - 1]
                op[r][c] = "E"
            else:
                sub_cost = dp[r - 1][c - 1] + 1
                del_cost = dp[r - 1][c] + 1
                ins_cost = dp[r][c - 1] + 1
                best = min(sub_cost, del_cost, ins_cost)
                dp[r][c] = best
                if best == sub_cost:
                    op[r][c] = "S"
                elif best == del_cost:
                    op[r][c] = "D"
                else:
                    op[r][c] = "I"

    s = d = ins = 0
    r, c = n, m
    while r > 0 or c > 0:
        move = op[r][c] if r >= 0 and c >= 0 else ""
        if move in ("E", "S"):
            if move == "S":
                s += 1
            r -= 1
            c -= 1
        elif move == "D":
            d += 1
            r -= 1
        elif move == "I":
            ins += 1
            c -= 1
        else:
            if r > 0:
                d += 1
                r -= 1
            elif c > 0:
                ins += 1
                c -= 1

    denom = max(n, 1)
    wer = round((s + d + ins) / denom, 4)
    return wer, s, d, ins


def _run_synthetic(model: WhisperModel) -> dict:
    audio = np.random.randn(5 * 16000).astype(np.float32)
    latencies = []
    for _ in range(5):
        t0 = time.perf_counter()
        segs, _ = model.transcribe(audio, language="en")
        list(segs)
        latencies.append(time.perf_counter() - t0)

    mean_lat = sum(latencies) / len(latencies)
    return {
        "rtf": round(mean_lat / 5.0, 4),
        "latency_mean_s": round(mean_lat, 3),
        "audio_duration_s": 5.0,
        "note": "synthetic_audio_benchmark",
    }


def _run_files(model: WhisperModel, audio_files: list[str], ground_truths: list[str]) -> dict:
    results: list[dict] = []

    for i, fpath in enumerate(audio_files):
        audio, sr = sf.read(fpath)
        duration_s = len(audio) / sr

        t0 = time.perf_counter()
        segs, _info = model.transcribe(fpath, language="en")
        transcript = " ".join(s.text for s in segs)
        elapsed = time.perf_counter() - t0

        rtf = round(elapsed / duration_s, 4) if duration_s > 0 else 0
        wer = None
        wer_approx = None
        wer_s = wer_d = wer_i = None
        if i < len(ground_truths):
            ref = ground_truths[i].lower().split()
            hyp = transcript.lower().split()
            wer, wer_s, wer_d, wer_i = _levenshtein_wer(ref, hyp)
            sm = SequenceMatcher(None, ref, hyp)
            matches = sum(block.size for block in sm.get_matching_blocks())
            wer_approx = round(1 - matches / max(len(ref), 1), 4)

        results.append(
            {
                "file": fpath,
                "rtf": rtf,
                "duration_s": round(duration_s, 2),
                "elapsed_s": round(elapsed, 3),
                "wer": wer,
                "wer_approx": wer_approx,
                "wer_s": wer_s,
                "wer_d": wer_d,
                "wer_i": wer_i,
            }
        )

    rtf_list = [r["rtf"] for r in results]
    wer_list = [r["wer"] for r in results if r["wer"] is not None]
    wer_approx_list = [r["wer_approx"] for r in results if r["wer_approx"] is not None]
    return {
        "n_files": len(results),
        "rtf_mean": round(sum(rtf_list) / len(rtf_list), 4) if rtf_list else 0,
        "rtf_min": round(min(rtf_list), 4) if rtf_list else 0,
        "wer_mean": round(sum(wer_list) / len(wer_list), 4) if wer_list else None,
        "wer_approx_mean": round(sum(wer_approx_list) / len(wer_approx_list), 4)
        if wer_approx_list
        else None,
        "per_file_detail": results,
    }


def main() -> int:
    args = _parse_args()
    model = WhisperModel(args.model, device=args.device, compute_type=args.precision)

    if args.synthetic:
        print(json.dumps(_run_synthetic(model)))
        return 0

    audio_files = json.loads(args.audio_files_json)
    ground_truths = json.loads(args.ground_truths_json)
    if not isinstance(audio_files, list):
        raise ValueError("audio-files-json must be a JSON list")
    if not isinstance(ground_truths, list):
        raise ValueError("ground-truths-json must be a JSON list")

    print(json.dumps(_run_files(model, [str(v) for v in audio_files], [str(v) for v in ground_truths])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
