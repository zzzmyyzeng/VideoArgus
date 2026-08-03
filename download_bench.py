#!/usr/bin/env python
"""Download VideoArgusBench (inputs + rubrics) from HuggingFace into ./VideoArgusBench.

The benchmark is published as a HF dataset rather than bundled in this repo. This script pulls it
into the local ./VideoArgusBench directory, which is the default --bench path evaluate.py expects.

Usage:
    python download_bench.py                                   # -> ./VideoArgusBench
    python download_bench.py --dest /data/VideoArgusBench
    python download_bench.py --repo-id someone/MyBench
"""
from __future__ import annotations

import os
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DEST = REPO_ROOT / "VideoArgusBench"


def main():
    ap = argparse.ArgumentParser(description="Download VideoArgusBench from HuggingFace.")
    ap.add_argument("--repo-id", default="zengziyun/VideoArgusBench")
    ap.add_argument("--dest", default=str(DEFAULT_DEST),
                    help="local directory to download the benchmark into (default: ./VideoArgusBench)")
    ap.add_argument("--revision", default=None, help="optional dataset revision / tag")
    args = ap.parse_args()

    # load repo .env so HF_TOKEN can live there (needed only if the dataset is private)
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except Exception:
        pass
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    from huggingface_hub import snapshot_download

    dest = os.path.abspath(args.dest)
    print(f"[download_bench] {args.repo_id} (dataset) -> {dest}", flush=True)
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=dest,
        token=token,
    )
    print(f"[download_bench] done. Use it with:  python evaluate.py --bench {args.dest} ...", flush=True)


if __name__ == "__main__":
    main()
