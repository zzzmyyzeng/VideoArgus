"""Repository paths and root resolution for VideoArgus.

Everything is anchored to the repo root (the directory that contains this package), so the
release is fully portable — clone anywhere, no absolute paths baked in. A `.env` at the repo
root is loaded if present (optional; standard official API keys can also come from the shell
environment).
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# repo root = parent of the `videoargus/` package directory
REPO_ROOT = Path(__file__).resolve().parents[1]

# vendored CV-tool source (added to sys.path by te_tools when needed)
UNIPERCEPT_SRC = str(REPO_ROOT / "tool_models" / "UniPercept" / "src")

# optional .env at the repo root (never required — creds may live in the shell env)
load_dotenv(REPO_ROOT / ".env")


def root(env_key: str, *default_parts: str) -> str:
    """Resolve a directory with the same override semantics used throughout the pipeline:

    1. if the env var `env_key` is set, use it verbatim (absolute or relative to CWD);
    2. else join `default_parts` under the repo root.
    """
    v = os.environ.get(env_key)
    if v:
        return v
    return str(REPO_ROOT.joinpath(*default_parts))
