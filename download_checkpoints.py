#!/usr/bin/env python
"""Download the CV-tool checkpoints VideoArgus's evaluation tools use, into tool_models/checkpoints.

The evaluation tools load these models from the HuggingFace cache on first use. This script
pre-fetches them so the first eval run doesn't stall on downloads (and so an air-gapped machine
can be primed ahead of time). It sets HF_HOME to tool_models/checkpoints (unless HF_HOME is already
set) so everything lands in one relocatable directory inside the repo.

Checkpoints:
  * facebook/dinov3-vitb16-pretrain-lvd1689m   DINOv3 ViT-B/16   (dino_sim identity/similarity)
  * depth-anything/Depth-Anything-V2-Small-hf  Depth-Anything-V2 (depth_probe / front-back geometry)
  * facebook/sam3                              SAM3 (sam3.pt + config.json)  (sam_track / crop / spatial)
  * Thunderbolt215215/UniPercept               UniPercept        (aesthetic / imaging-quality tool)

PaddleOCR is different: its detector/recognizer weights are NOT on the HuggingFace Hub — Paddle
fetches them from its own servers on the first `PaddleOCR(...)` call. To prime them ahead of time
(e.g. for an air-gapped box), pass `--models ... paddleocr`, which instantiates PaddleOCR once so it
downloads into its cache. `paddleocr` is NOT in the default set (it needs the paddleocr package and,
depending on the build, a GPU); request it explicitly.

Not downloaded here:
  * The judge VLM — you serve it yourself with vLLM (see evaluate.py); pull it separately, e.g.
        huggingface-cli download Qwen/Qwen3.6-27B

Gated/large repos honor HF_TOKEN (from the environment or the repo .env).

Usage:
    python download_checkpoints.py                 # all HF checkpoints, into tool_models/checkpoints
    python download_checkpoints.py --models dinov3 sam3
    python download_checkpoints.py --models dinov3 depth sam3 unipercept paddleocr   # + prime OCR
    python download_checkpoints.py --dest /path/to/hf_cache
"""
from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DEST = REPO_ROOT / "tool_models" / "checkpoints"

# name -> (repo_id, [specific files] or None for full snapshot, extra kwargs)
CHECKPOINTS = {
    "dinov3": ("facebook/dinov3-vitb16-pretrain-lvd1689m", None, {}),
    "depth":  ("depth-anything/Depth-Anything-V2-Small-hf", None, {}),
    "sam3":   ("facebook/sam3", ["sam3.pt", "config.json"], {}),
    "unipercept": ("Thunderbolt215215/UniPercept", None, {}),
}
# PaddleOCR weights don't come from the HF Hub; priming = instantiate PaddleOCR once. Kept out
# of the default HF-snapshot loop and out of the choices below (handled separately in main()).
PADDLE = "paddleocr"


def prime_paddleocr():
    """Instantiate PaddleOCR once so it downloads its detector/recognizer weights into its cache."""
    from paddleocr import PaddleOCR
    PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False,
              use_textline_orientation=False)


def main():
    ap = argparse.ArgumentParser(description="Pre-fetch VideoArgus CV-tool checkpoints.")
    ap.add_argument("--models", nargs="*", choices=list(CHECKPOINTS) + [PADDLE],
                    default=list(CHECKPOINTS),
                    help="subset to fetch (default: all HF checkpoints; add 'paddleocr' to also "
                         "prime PaddleOCR's non-HF weights)")
    ap.add_argument("--dest", default=str(DEFAULT_DEST),
                    help="HF cache directory to download into (default: tool_models/checkpoints)")
    args = ap.parse_args()

    dest = os.path.abspath(args.dest)
    os.makedirs(dest, exist_ok=True)
    # route the HF cache into --dest unless the user already pinned HF_HOME
    os.environ.setdefault("HF_HOME", dest)

    # load repo .env so HF_TOKEN can live there
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except Exception:
        pass
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    from huggingface_hub import snapshot_download, hf_hub_download

    print(f"[download_checkpoints] HF_HOME={os.environ['HF_HOME']}  models={args.models}", flush=True)
    for name in args.models:
        if name == PADDLE:
            print(f"  {name}: priming PaddleOCR (non-HF weights) ...", flush=True)
            try:
                prime_paddleocr()
                print(f"  {name}: OK", flush=True)
            except Exception as e:
                print(f"  {name}: FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
            continue
        repo_id, files, kw = CHECKPOINTS[name]
        print(f"  {name}: {repo_id} ...", flush=True)
        try:
            if files is None:
                snapshot_download(repo_id=repo_id, token=token, **kw)
            else:
                for fn in files:
                    hf_hub_download(repo_id=repo_id, filename=fn, token=token, **kw)
            print(f"  {name}: OK", flush=True)
        except Exception as e:
            print(f"  {name}: FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)

    if PADDLE not in args.models:
        print("[download_checkpoints] note: PaddleOCR weights auto-download on first `ocr` use "
              "(add 'paddleocr' to --models to prime them now).", flush=True)
    print("[download_checkpoints] done. Serve the judge VLM yourself (see evaluate.py --judge-endpoint).",
          flush=True)


if __name__ == "__main__":
    main()
