"""Score aggregation and reporting for VideoArgus eval outputs.

Each per-video report JSON contains a `criteria[]` list, one entry per rubric criterion with a
0-10 integer `score`, plus `importance`, `type`, and `hard_cap`. The final per-video score is a
SOFT-CAP blend of the importance-weighted base score and the hard-capped score:

    base       = importance-weighted mean of criterion scores  (IMP_W high:3 / medium:2 / low:1)
    cap_val    = min hard_cap over FIRED hard criteria (a hard criterion "fires" when score < 5)
    hard_final = min(base, cap_val)
    final(a)   = base*(1-a) + hard_final*a          (alpha default 0.5)

This module re-aggregates from criteria[] and never mutates stored files.
"""
from __future__ import annotations

import glob
import json
import os
import statistics

IMP_W = {"high": 3, "medium": 2, "low": 1}
FAIL_THR = 5
DEFAULT_ALPHA = 0.5


def softcap_final(criteria, alpha: float = DEFAULT_ALPHA):
    """Recompute a single video's final score from its criteria[] at soft-cap `alpha`.

    Returns None if no criterion has an integer score.
    """
    scored = [c for c in criteria if isinstance(c.get("score"), int)]
    if not scored:
        return None
    num = sum(IMP_W.get(c.get("importance"), 1) * c["score"] for c in scored)
    den = sum(IMP_W.get(c.get("importance"), 1) for c in scored)
    base = num / den
    cap_val = 10.0
    for c in scored:
        if c.get("type") == "hard" and c["score"] < FAIL_THR and c.get("hard_cap") is not None:
            cap_val = min(cap_val, float(c["hard_cap"]))
    hard_final = min(base, cap_val)
    return base * (1 - alpha) + hard_final * alpha


def score_report(path: str, alpha: float = DEFAULT_ALPHA):
    """Load one report JSON and return its soft-cap final score (or None)."""
    try:
        obj = json.load(open(path))
    except Exception:
        return None
    return softcap_final(obj.get("criteria", []), alpha)


def aggregate_eval_root(eval_root: str, alpha: float = DEFAULT_ALPHA) -> dict:
    """Walk `<eval_root>/<task>/<model>/*.json` and return per (task, model) mean + count.

    Result: {task: {model: {"mean": float, "n": int}}}
    """
    out: dict = {}
    for task_dir in sorted(glob.glob(os.path.join(eval_root, "*"))):
        if not os.path.isdir(task_dir):
            continue
        task = os.path.basename(task_dir.rstrip("/"))
        for model_dir in sorted(glob.glob(os.path.join(task_dir, "*"))):
            if not os.path.isdir(model_dir):
                continue
            model = os.path.basename(model_dir.rstrip("/"))
            scores = []
            for f in glob.glob(os.path.join(model_dir, "*.json")):
                s = score_report(f, alpha)
                if s is not None:
                    scores.append(float(s))
            if scores:
                out.setdefault(task, {})[model] = {"mean": statistics.mean(scores), "n": len(scores)}
    return out


def format_report(agg: dict, alpha: float = DEFAULT_ALPHA) -> str:
    lines = [f"(soft-cap alpha={alpha})"]
    for task in sorted(agg):
        for model in sorted(agg[task]):
            r = agg[task][model]
            lines.append(f"{task}/{model}: mean={r['mean']:.2f} n={r['n']}")
    return "\n".join(lines)
