#!/usr/bin/env python
"""Evaluate generated videos against sample-specific rubrics with a LOCAL vLLM judge.

For each rubric criterion, an agent repairs and runs its evidence plan (CV tools + optional web
search), then a judge VLM scores the criterion 0-10 given the tool evidence and the output-video
frames. Scores aggregate to a per-video final via an importance-weighted mean with a soft hard-cap.

The judge VLM is served by YOUR OWN local vLLM server (OpenAI-compatible). No hosted-API
credentials are used on the evaluation path. Start a server, e.g.:

    python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen3.6-27B --port 8000 --tensor-parallel-size 2

then point --judge-endpoint at it. Multiple endpoints (comma-separated) fan criteria out in parallel.

Layout expected:
    <bench>/<task>/rubrics/<id>.json                   rubrics (one per input)
    <bench>/<task>/manifest.jsonl                      input bundle spec (media under <task>/images,videos)
    <videos-dir>/<task>/<model>/<id>.mp4               the generated videos to score

Usage:
    python evaluate.py \
        --bench VideoArgusBench --task TSV2V --model my_model \
        --videos-dir ./videos_under_eval \
        --judge-endpoint http://localhost:8000/v1 --judge-model Qwen/Qwen3.6-27B \
        --output ./eval_runs --run-name run1 --report
"""
from __future__ import annotations

import os
import sys
import json
import argparse


def main():
    ap = argparse.ArgumentParser(description="Score generated videos with a local vLLM judge.")
    ap.add_argument("--bench", default="VideoArgusBench", help="benchmark root (contains <task>/...)")
    ap.add_argument("--task", required=True, choices=["T2V", "TI2V", "TS2V", "TV2V", "TSV2V"])
    ap.add_argument("--model", required=True, help="label of the model whose videos are scored (<videos-dir>/<task>/<model>/)")
    ap.add_argument("--videos-dir", required=True, help="root of generated videos: <task>/<model>/<id>.mp4")
    ap.add_argument("--judge-endpoint", required=True,
                    help="OpenAI-compatible base URL(s) of your local vLLM server (comma-separated for a pool)")
    ap.add_argument("--judge-model", required=True, help="model name the vLLM server serves (the judge VLM)")
    ap.add_argument("--rubric-subdir", default="",
                    help="optional sub-directory under <task>/rubrics/ to score against "
                         "(default: rubrics directly under <task>/rubrics/)")
    ap.add_argument("--output", default="eval_runs", help="eval output root")
    ap.add_argument("--run-name", default="run1")
    ap.add_argument("--alpha", type=float, default=0.5, help="soft hard-cap blend (1.0 = hard cap; default 0.5)")
    ap.add_argument("--concurrency", type=int, default=1, help="criteria evaluated in parallel per video")
    ap.add_argument("--slots-per-endpoint", type=int, default=1, help="concurrent requests allowed per endpoint")
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--report", action="store_true", help="print per-model mean at --alpha after scoring")
    ap.add_argument("--vlm-only", action="store_true",
                    help="ablation: disable CV/web tools, route each criterion to a single vlm_qa")
    args = ap.parse_args()

    # --- wire the pipeline via env, BEFORE importing tool_executor (its submodules read these
    #     at import time). Local vLLM only — no hosted-API creds anywhere on this path. ---------
    bench_abs = os.path.abspath(args.bench)
    os.environ["VIDARGUS_RUBRIC_ROOT"] = bench_abs
    os.environ["VIDARGUS_RUBRIC_SUBDIR"] = args.rubric_subdir
    os.environ["VIDARGUS_VIDEO_ROOT"] = os.path.abspath(args.videos_dir)
    os.environ["VIDARGUS_EVAL_ROOT"] = os.path.abspath(args.output)
    os.environ["VIDARGUS_VLM_ENDPOINTS"] = args.judge_endpoint
    os.environ["VIDARGUS_VLM_MODEL"] = args.judge_model
    os.environ["VIDARGUS_QA_MODEL"] = args.judge_model
    os.environ["VIDARGUS_CAP_ALPHA"] = str(args.alpha)
    os.environ["VIDARGUS_EVAL_CONCURRENCY"] = str(args.concurrency)
    os.environ["VIDARGUS_VLM_SLOTS_PER_ENDPOINT"] = str(args.slots_per_endpoint)
    if args.vlm_only:
        os.environ["VIDARGUS_VLM_ONLY"] = "1"

    from videoargus import tool_executor as ex

    ids = args.ids or ex.list_ids(args.task)
    if args.num_shards > 1:
        ids = [v for i, v in enumerate(ids) if i % args.num_shards == args.shard]

    print(f"[evaluate] task={args.task} model={args.model} rubric-subdir={args.rubric_subdir or '(root)'} "
          f"n={len(ids)} judge={args.judge_model} @ {args.judge_endpoint} "
          f"alpha={args.alpha} web={ex.web_enabled()}", flush=True)

    done = skip = err = 0
    for vid in ids:
        try:
            r = ex.eval_video(args.task, args.model, vid, args.run_name, overwrite=args.overwrite)
            if r.get("skipped"):
                skip += 1
            elif r.get("error"):
                err += 1
                print(f"  {vid}: {r['error']}", flush=True)
            else:
                done += 1
                agg = r.get("aggregate", {})
                print(f"  {vid}: final={agg.get('final_score')} base={agg.get('base_score')} "
                      f"caps={len(agg.get('caps_applied', []))} crit={len(r.get('criteria', []))}",
                      flush=True)
        except Exception as e:
            err += 1
            print(f"  {vid}: ERROR {type(e).__name__}: {str(e)[:200]}", flush=True)

    print(f"[evaluate] scored={done} skipped={skip} errors={err}", flush=True)

    if args.report:
        from utils import report as rp
        # report over just this run's eval root: <output>/<run-name>
        run_root = os.path.join(os.path.abspath(args.output), args.run_name)
        agg = rp.aggregate_eval_root(run_root, alpha=args.alpha)
        print("\n" + rp.format_report(agg, alpha=args.alpha))


if __name__ == "__main__":
    main()
