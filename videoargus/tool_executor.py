"""VideoArgus tool executor — orchestration, judge, aggregation, output.

For each (task, model, id):
  1. Build ExecutionContext (resolve output video + input bundle).
  2. For each criterion: repair its evidence_plan (te_repair), execute the normalized
     tool_sequence (te_tools), resolve 'reference' just-in-time, collect evidence,
     then ask the judge VLM for a 0-10 score.
  3. Aggregate: weighted mean (high/med/low = 3/2/1); a HARD criterion scoring < 5
     caps the final score at its hard_cap (soft-cap blend controlled by alpha).
  4. Write per-video JSON to <eval-root>/<run>/<TASK>/<MODEL>/<ID>.json.

The judge VLM is served over an OpenAI-compatible local endpoint pool (see te_tools). No hosted
API credentials are used anywhere on the evaluation path.

This module is normally driven by the top-level `evaluate.py`, which sets the VIDARGUS_* env
knobs (rubric root/subdir, video root, eval root, judge endpoint/model, cap alpha) before calling
`eval_video`. It can also be run directly as a CLI (see `main`).
"""
from __future__ import annotations

import os
import sys
import json
import glob
import argparse
import traceback
from collections import Counter

from .paths import root as _root

# Benchmark root: contains <task>/rubrics/<id>.json + <task>/manifest.jsonl + media.
RUBRIC_ROOT = _root("VIDARGUS_RUBRIC_ROOT", "VideoArgusBench")
# Optional sub-directory under <task>/rubrics/ to read rubrics from. Empty (default) = rubrics live
# directly under <task>/rubrics/. Set VIDARGUS_RUBRIC_SUBDIR to score an alternative rubric set kept
# in a subfolder (e.g. rubrics from a different generator you produced with generate_rubrics.py).
RUBRIC_SUBDIR = os.environ.get("VIDARGUS_RUBRIC_SUBDIR", "")
# where per-video eval JSON is written
EVAL_ROOT = _root("VIDARGUS_EVAL_ROOT", "eval_runs")

# the five task families this pipeline evaluates
TASKS = ["T2V", "TI2V", "TS2V", "TV2V", "TSV2V"]

IMP_WEIGHT = {"high": 3, "medium": 2, "low": 1}
FAIL_THR = 5  # a hard criterion scoring < FAIL_THR is a failure -> hard_cap applies
# Softening of the hard-cap correction. 1.0 = hard min(base, cap); a<1.0 blends toward base as
# final = base*(1-a) + min(base,cap)*a. a=0.5 keeps the cap auditable while tracking human
# consensus better. Override via env (evaluate.py sets it from --alpha).
CAP_ALPHA = float(os.environ.get("VIDARGUS_CAP_ALPHA", "0.5"))
# Criteria-within-a-video concurrency. 1 = serial. Set to ~N (endpoint count) so vlm_qa/judge
# calls fan out across the local endpoint pool.
CONCURRENCY = int(os.environ.get("VIDARGUS_EVAL_CONCURRENCY", "1"))

from . import te_repair
from . import te_resolve


# --------------------------------------------------------------------------- #
# rubric / id discovery
# --------------------------------------------------------------------------- #
def _rubric_dir(task: str) -> str:
    base = os.path.join(RUBRIC_ROOT, task, "rubrics")
    return os.path.join(base, RUBRIC_SUBDIR) if RUBRIC_SUBDIR else base


def load_rubric(task: str, vid: str) -> dict:
    with open(os.path.join(_rubric_dir(task), f"{vid}.json")) as f:
        return json.load(f)


def list_ids(task: str) -> list:
    paths = sorted(glob.glob(os.path.join(_rubric_dir(task), "*.json")))
    return [os.path.splitext(os.path.basename(p))[0] for p in paths]


def web_enabled() -> bool:
    return bool(os.environ.get("SERPER_API_KEY"))


# --------------------------------------------------------------------------- #
# reference resolution (just-in-time, per criterion)
# --------------------------------------------------------------------------- #
def resolve_reference(ctx, reference_source: str):
    """Register 'reference' in ctx.media_registry from a non-web source.
    web_search registers it itself during execution."""
    if reference_source == "subject_image" and ctx.has_media("subject_image"):
        ctx.register("reference", {"kind": "images",
                                   "paths": ctx.media_registry["subject_image"]["paths"]})
    elif reference_source == "first_frame" and ctx.has_media("first_frame"):
        ctx.register("reference", dict(ctx.media_registry["first_frame"]))
    elif reference_source == "source_video" and ctx.has_media("source_video"):
        # representative middle frame of source video as the reference image
        img = ctx.representative_image("source_video")
        if img is not None:
            ctx.register("reference", {"kind": "pil", "image": img})


# --------------------------------------------------------------------------- #
# per-criterion execution + judge
# --------------------------------------------------------------------------- #
def execute_criterion(ctx, criterion: dict, dry: bool = False) -> dict:
    base_media = ctx.available_media()
    plan = te_repair.repair(criterion, base_media, web_enabled(), use_llm=not dry)
    # --- VLM-ONLY ablation ------------------------------------------------- #
    # When VIDARGUS_VLM_ONLY=1, disable every CV/web tool and route the criterion
    # straight to a single vlm_qa on the output video (the VLM answers everything).
    # This measures the contribution of the tool layer vs a bare-VLM judge.
    if not dry and os.environ.get("VIDARGUS_VLM_ONLY") == "1":
        from . import te_repair as _tr
        plan = {"tool_sequence": [_tr._step("vlm_qa",
                    question=_tr._criterion_question(criterion), on="output_video")],
                "reference_source": "none", "repairs": ["VLM_ONLY: tools disabled -> single vlm_qa"],
                "degraded": True, "fallback": True}
    out = {"id": criterion.get("id"), "dimension": criterion.get("dimension"),
           "type": criterion.get("type"), "importance": criterion.get("importance"),
           "hard_cap": criterion.get("hard_cap"),
           "repairs": plan["repairs"], "degraded": plan["degraded"],
           "normalized_sequence": [s["tool"] for s in plan["tool_sequence"]]}
    if dry:
        return out

    from . import te_tools
    # pre-resolve non-web reference
    if plan["reference_source"] not in ("none", None) and plan["reference_source"] != "web_search":
        resolve_reference(ctx, plan["reference_source"])

    step_state: dict = {}
    evidence = []
    # GPU-bound tools (and their crop->dino chain + ctx.media_registry mutations) must run
    # serialized under TOOL_LOCK when criteria are evaluated concurrently. vlm_qa/web_search
    # are pure HTTP — if the sequence is ONLY those, skip the lock so they parallelize freely.
    GPU_TOOLS = {"sam_track", "crop_compare", "dino_sim", "ocr", "aesthetic",
                 "depth_probe", "spatial_relation", "flicker_probe"}
    needs_lock = any(s["tool"] in GPU_TOOLS for s in plan["tool_sequence"])
    if needs_lock:
        with te_tools.TOOL_LOCK:
            for step in plan["tool_sequence"]:
                evidence.append(te_tools.run_tool(step["tool"], ctx, step.get("args", {}), step_state))
    else:
        for step in plan["tool_sequence"]:
            evidence.append(te_tools.run_tool(step["tool"], ctx, step.get("args", {}), step_state))
    out["evidence"] = evidence

    score, rationale = judge_criterion(ctx, criterion, evidence)
    out["score"] = score
    out["rationale"] = rationale
    return out


def judge_criterion(ctx, criterion: dict, evidence: list):
    """Ask the judge VLM for an integer 0-10 + rationale, given criterion + evidence."""
    from . import te_tools

    ep = criterion.get("evidence_plan", {}) or {}
    # textual evidence summary
    ev_txt = json.dumps([{k: v for k, v in e.items() if k != "trace"} for e in evidence],
                        indent=1)[:6000]

    content = [{"type": "text", "text":
        "You are a strict evaluator. Score how well the GENERATED VIDEO meets ONE criterion.\n"
        f"CRITERION: {criterion.get('criterion','')}\n"
        f"SCORING RULE (0-10): {criterion.get('scoring_rule','')}\n"
        f"FAILURE MODES: {criterion.get('failure_modes')}\n"
        f"AGGREGATION HINT (informational, do not execute): {ep.get('aggregation','')}\n\n"
        f"TOOL EVIDENCE (results of running the evidence plan):\n{ev_txt}\n\n"
        "Frames of the output video follow. Weigh tool evidence (DINO cosine, OCR text, "
        "aesthetic 0-100, detection coverage) together with what you see. "
        "Return ONLY JSON: {\"score\": <int 0-10>, \"rationale\": \"<one or two sentences>\"}."}]
    # attach a few output frames
    for fr in ctx.frames("output_video")[:6]:
        content.append({"type": "image_url",
                        "image_url": {"url": te_tools._pil_to_data_url(fr)}})
    # Retry until we get a parseable score — a judge occasionally returns an EMPTY body, which
    # json.loads can't parse. We must not leave a null score.
    msgs = [{"role": "user", "content": content}]
    last_err = None
    for attempt in range(8):
        try:
            resp = te_tools.chat_with_retry(msgs, model=te_tools.QA_MODEL,
                                            response_format={"type": "json_object"}, max_tokens=400)
            raw = resp.choices[0].message.content
            if not raw or not raw.strip():
                last_err = "empty response"
                # nudge with a plain-text follow-up next round
                msgs = [{"role": "user", "content": content},
                        {"role": "user", "content": 'Return ONLY a JSON object: {"score": <int 0-10>, "rationale": "<short>"}'}]
                import time as _t; _t.sleep(3 + attempt * 2); continue
            data = _lenient_json(raw)
            score = max(0, min(10, int(round(float(data.get("score", 0))))))
            return score, data.get("rationale", "")
        except Exception as e:
            last_err = str(e)[:120]
            import time as _t; _t.sleep(3 + attempt * 2)
    # exhausted retries — try ONCE more (plain, no response_format) rather than fabricate a score.
    try:
        resp = te_tools.chat_with_retry(msgs, model=te_tools.QA_MODEL, max_tokens=400)
        raw = resp.choices[0].message.content
        if raw and raw.strip():
            data = _lenient_json(raw)
            return max(0, min(10, int(round(float(data.get("score", 0)))))), "[retry] " + data.get("rationale", "")
    except Exception as e:
        last_err = str(e)[:120]
    return None, f"judge error (8 retries + alt failed): {last_err}"


def _lenient_json(txt: str) -> dict:
    """Parse a JSON object from possibly-messy model output (a local model may wrap in
    ```json fences or add prose). Strip fences, extract the first {...}, then json.loads."""
    t = (txt or "").strip()
    if "```" in t:
        seg = t.split("```")
        # take the fenced block; drop a leading 'json' language tag
        t = seg[1][4:].strip() if seg[1].lower().startswith("json") else seg[1].strip()
    if "{" in t and "}" in t:
        t = t[t.index("{"): t.rindex("}") + 1]
    return json.loads(t)


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def aggregate(criteria_results: list, cap_alpha=None) -> dict:
    scored = [c for c in criteria_results if isinstance(c.get("score"), int)]
    if not scored:
        return {"final_score": None, "base_score": None, "n_scored": 0, "caps_applied": []}
    num = sum(IMP_WEIGHT.get(c["importance"], 1) * c["score"] for c in scored)
    den = sum(IMP_WEIGHT.get(c["importance"], 1) for c in scored)
    base = num / den
    caps = []
    cap_val = 10.0
    for c in scored:
        if c.get("type") == "hard" and c["score"] < FAIL_THR and c.get("hard_cap") is not None:
            caps.append({"id": c["id"], "score": c["score"], "cap": c["hard_cap"]})
            cap_val = min(cap_val, float(c["hard_cap"]))
    hard_final = min(base, cap_val)
    # cap softening: a=1.0 reproduces the hard cap exactly.
    a = CAP_ALPHA if cap_alpha is None else cap_alpha
    final = hard_final if a >= 1.0 else base * (1 - a) + hard_final * a
    return {"final_score": round(final, 2), "base_score": round(base, 2),
            "n_scored": len(scored), "caps_applied": caps, "cap_alpha": a}


# --------------------------------------------------------------------------- #
# per-video driver
# --------------------------------------------------------------------------- #
def eval_video(task: str, model: str, vid: str, run_name: str, dry: bool = False,
               overwrite: bool = False) -> dict:
    out_path = os.path.join(EVAL_ROOT, run_name, task, model, f"{vid}.json")
    if os.path.exists(out_path) and not overwrite and not dry:
        return {"id": vid, "skipped": True}

    rubric = load_rubric(task, vid)
    ctx = te_resolve.ExecutionContext(task, model, vid)

    result = {"id": vid, "task": task, "model": model,
              "output_video": ctx.output_video,
              "output_exists": ctx.output_exists(),
              "bundle": {"subject_images": ctx.bundle.subject_images,
                         "first_frame": ctx.bundle.first_frame,
                         "source_video": ctx.bundle.source_video},
              "criteria": []}
    if not dry and not ctx.output_exists():
        result["error"] = "output video missing"
        return result

    crits = rubric.get("criteria", [])

    def _run_one(crit):
        try:
            return execute_criterion(ctx, crit, dry=dry)
        except Exception as e:
            return {"id": crit.get("id"), "error": str(e)[:300],
                    "trace": traceback.format_exc()[-500:], "score": None}

    if CONCURRENCY > 1 and not dry and crits:
        # Pre-warm the frame cache once (avoids concurrent frame-decode races); then fan
        # criteria across threads. GPU tools serialize on te_tools.TOOL_LOCK; vlm_qa/judge
        # (HTTP) hit the local endpoint pool in parallel.
        ctx.frames("output_video")
        if ctx.bundle.source_video:
            ctx.frames("source_video")
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
            result["criteria"] = list(ex.map(_run_one, crits))  # map preserves order
    else:
        result["criteria"] = [_run_one(c) for c in crits]

    if not dry:
        result["aggregate"] = aggregate(result["criteria"])
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=1)
        result["written_to"] = out_path
    return result


# --------------------------------------------------------------------------- #
# dry-run-repair over all rubrics (no GPU / no judge; just the repair layer)
# --------------------------------------------------------------------------- #
def dry_run_repair(tasks: list, model: str = "model"):
    rule_counter = Counter()
    total = 0; crit_total = 0; degraded = 0; fallback = 0; crashes = 0
    per_task = {}
    we = web_enabled()
    for task in tasks:
        ids = list_ids(task)
        t_crit = 0; t_deg = 0
        for vid in ids:
            total += 1
            try:
                rubric = load_rubric(task, vid)
                ctx = te_resolve.ExecutionContext(task, model, vid)
                base_media = ctx.available_media()
            except Exception as e:
                crashes += 1
                print(f"  CRASH building ctx {task}/{vid}: {e}")
                continue
            for crit in rubric.get("criteria", []):
                crit_total += 1; t_crit += 1
                try:
                    plan = te_repair.repair(crit, base_media, we, use_llm=False)
                except Exception as e:
                    crashes += 1
                    print(f"  CRASH repair {task}/{vid}/{crit.get('id')}: {e}")
                    continue
                if plan["degraded"]:
                    degraded += 1; t_deg += 1
                if plan["fallback"]:
                    fallback += 1
                for r in plan["repairs"]:
                    key = r.split(":")[0].split("(")[0].strip()[:48]
                    rule_counter[key] += 1
        per_task[task] = {"videos": len(ids), "criteria": t_crit, "degraded_criteria": t_deg}
    print("\n========== DRY-RUN REPAIR SUMMARY ==========")
    print(f"web_search enabled: {we}")
    print(f"videos: {total} | criteria: {crit_total} | degraded: {degraded} "
          f"| fallback: {fallback} | CRASHES: {crashes}")
    print("per task:", json.dumps(per_task, indent=1))
    print("\nrepair-rule histogram (top 30):")
    for k, n in rule_counter.most_common(30):
        print(f"  {n:5d}  {k}")
    return crashes


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=TASKS)
    ap.add_argument("--model", default="model")
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--run-name", default="run1")
    ap.add_argument("--rubric-subdir", default=None, help="optional sub-directory under <task>/rubrics/ (default: read rubrics directly under rubrics/)")
    ap.add_argument("--dry-run-repair", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.rubric_subdir:
        global RUBRIC_SUBDIR
        RUBRIC_SUBDIR = args.rubric_subdir

    if args.dry_run_repair:
        tasks = [args.task] if args.task else TASKS
        crashes = dry_run_repair(tasks, model=args.model)
        sys.exit(1 if crashes else 0)

    assert args.task, "--task required for eval"
    ids = args.ids or list_ids(args.task)
    if args.num_shards > 1:
        ids = [v for i, v in enumerate(ids) if i % args.num_shards == args.shard]

    print(f"[executor] task={args.task} model={args.model} n_ids={len(ids)} "
          f"run={args.run_name} web={web_enabled()}", flush=True)
    for vid in ids:
        try:
            r = eval_video(args.task, args.model, vid, args.run_name, overwrite=args.overwrite)
            if r.get("skipped"):
                print(f"  {vid}: skip (exists)", flush=True)
            else:
                agg = r.get("aggregate", {})
                print(f"  {vid}: final={agg.get('final_score')} base={agg.get('base_score')} "
                      f"caps={len(agg.get('caps_applied', []))} "
                      f"crit={len(r.get('criteria', []))}", flush=True)
        except Exception as e:
            print(f"  {vid}: ERROR {e}", flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
