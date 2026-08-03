#!/usr/bin/env bash
# VideoArgus one-shot setup: fetch everything the repo does NOT bundle.
#
#   1. clone CV-tool source (SAM3, UniPercept)      -> tool_models/{sam3,UniPercept}
#   2. download CV-tool checkpoints                 -> tool_models/checkpoints
#   3. download the benchmark (inputs + rubrics)    -> VideoArgusBench/
#
# The judge VLM is served by you (see evaluate.py / README) and is NOT fetched here.
#
# Usage:
#   bash setup.sh                # everything
#   bash setup.sh --no-bench     # skip the benchmark download
#   bash setup.sh --no-ckpt      # skip the checkpoint download (e.g. prime tools only)
#
# Run inside your activated environment (conda activate videoargus). Honors HF_TOKEN from the
# environment or a repo-root .env for gated/ private downloads.
set -euo pipefail

cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"

DO_TOOLS=1; DO_CKPT=1; DO_BENCH=1
for arg in "$@"; do
  case "$arg" in
    --no-tools) DO_TOOLS=0 ;;
    --no-ckpt)  DO_CKPT=0 ;;
    --no-bench) DO_BENCH=0 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

# Preflight: the interpreter must have the project deps (huggingface_hub is used by every download
# step). This is the usual "forgot to activate the env" trip-wire — fail early with a clear hint.
if ! "$PYTHON" -c "import huggingface_hub" >/dev/null 2>&1; then
  echo "ERROR: '$PYTHON' can't import huggingface_hub." >&2
  echo "       Activate the project env first (conda activate videoargus), or point PYTHON at it:" >&2
  echo "         PYTHON=/path/to/env/bin/python bash setup.sh" >&2
  exit 1
fi

echo "==> [1/3] Cloning CV-tool source (SAM3, UniPercept)"
if [ "$DO_TOOLS" = 1 ]; then "$PYTHON" setup_tools.py; else echo "    skipped (--no-tools)"; fi

echo "==> [2/3] Downloading CV-tool checkpoints"
if [ "$DO_CKPT" = 1 ]; then "$PYTHON" download_checkpoints.py; else echo "    skipped (--no-ckpt)"; fi

echo "==> [3/3] Downloading VideoArgusBench"
if [ "$DO_BENCH" = 1 ]; then "$PYTHON" download_bench.py; else echo "    skipped (--no-bench)"; fi

echo "==> setup complete. Next: serve a judge VLM, then run evaluate.py (see README)."
