#!/usr/bin/env python
"""Clone the vendored CV-tool source packages VideoArgus's evaluation tools import at runtime.

VideoArgus does not bundle the SAM3 and UniPercept source; it clones them from upstream into
tool_models/ so you always get them from the original authors under their own licenses. The layout
after cloning matches what videoargus/te_tools.py and videoargus/paths.py expect:

    tool_models/sam3/sam3/...        -> `import sam3` (SAM3 image/video predictors)
    tool_models/UniPercept/src/...   -> paths.UNIPERCEPT_SRC (aesthetic / imaging-quality tool)

SAM3 ships a proper pyproject (package name `sam3`) with no compiled extensions, so after cloning
we `pip install -e --no-deps` it into the active environment — that registers the `sam3` package so
`import sam3` resolves (its runtime deps are already covered by requirements.txt). UniPercept is a
plain source tree with no installable package, so it is only added to sys.path at runtime (via
paths.UNIPERCEPT_SRC), not installed.

These are code only. The model *weights* are pulled separately by download_checkpoints.py.

Usage:
    python setup_tools.py                 # clone both + editable-install sam3 (skips existing clones)
    python setup_tools.py --tools sam3
    python setup_tools.py --force         # remove existing clones and re-clone
    python setup_tools.py --no-install    # clone only; skip the sam3 editable install
"""
from __future__ import annotations

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
TOOL_DIR = REPO_ROOT / "tool_models"

# name -> (git url, destination under tool_models/, optional pinned ref, pip-installable?)
# `pip_install` tools are `pip install -e --no-deps`'d after cloning so their package resolves on
# `import` (deps come from requirements.txt); the others are added to sys.path at runtime by paths.py.
TOOLS = {
    "sam3": ("https://github.com/facebookresearch/sam3", "sam3", None, True),
    "unipercept": ("https://github.com/thunderbolt215/UniPercept", "UniPercept", None, False),
}


def clone_one(name: str, force: bool) -> bool:
    url, sub, ref, _ = TOOLS[name]
    dest = TOOL_DIR / sub
    if dest.exists():
        if not force:
            print(f"  {name}: already present at {dest} (use --force to re-clone) — skipping", flush=True)
            return True
        print(f"  {name}: --force, removing {dest}", flush=True)
        shutil.rmtree(dest)

    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", "1"]
    if ref:
        cmd += ["--branch", ref]
    cmd += [url, str(dest)]
    print(f"  {name}: git clone {url} -> {dest}", flush=True)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"  {name}: FAILED (git exit {e.returncode})", flush=True)
        return False
    # drop the nested .git so the tool tree isn't a submodule/nested repo inside VideoArgus
    git_dir = dest / ".git"
    if git_dir.exists():
        shutil.rmtree(git_dir, ignore_errors=True)
    print(f"  {name}: OK", flush=True)
    return True


def pip_install_editable(name: str) -> bool:
    """Editable-install a clone (deps already provided by requirements.txt, so --no-deps)."""
    dest = TOOL_DIR / TOOLS[name][1]
    cmd = [sys.executable, "-m", "pip", "install", "-e", str(dest), "--no-deps"]
    print(f"  {name}: pip install -e {dest} --no-deps", flush=True)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"  {name}: pip install FAILED (exit {e.returncode})", flush=True)
        return False
    print(f"  {name}: installed", flush=True)
    return True


def patch_unipercept() -> None:
    """Make UniPercept's InternVLChatConfig load under modern (>=4.57) transformers.

    Upstream `configuration_internvl_chat.py` raises on an empty `llm_config.architectures`, but
    transformers >=4.57 constructs the config with no args (empty architecture) inside
    `to_diff_dict()` for diffing/logging — which would crash the aesthetic/quality tool at load.
    We insert a tolerant branch (treat empty/None architecture as the checkpoint's Qwen2 LLM) right
    before the `else: raise`. Idempotent; a no-op if the file already has the branch or moved.
    """
    cfg = (TOOL_DIR / "UniPercept" / "src" / "internvl" / "model" / "internvl_chat"
           / "configuration_internvl_chat.py")
    if not cfg.exists():
        print("  unipercept: config file not found — skipping compat patch", flush=True)
        return
    text = cfg.read_text()
    if "architectures'][0] in ('', None)" in text:
        print("  unipercept: transformers-compat patch already present", flush=True)
        return
    anchor = (
        "        elif llm_config['architectures'][0] == 'Qwen2ForCausalLM':\n"
        "            self.llm_config = Qwen2Config(**llm_config)\n"
        "        else:\n"
    )
    patched = (
        "        elif llm_config['architectures'][0] == 'Qwen2ForCausalLM':\n"
        "            self.llm_config = Qwen2Config(**llm_config)\n"
        "        elif llm_config['architectures'][0] in ('', None):\n"
        "            # transformers >=4.57 to_diff_dict() builds self.__class__() with an empty\n"
        "            # config for diffing/logging -> empty architecture. Tolerate it instead of\n"
        "            # raising, so UniPercept loads under the same modern transformers as the\n"
        "            # other CV tools (SAM3 / DINOv3 / Depth-Anything).\n"
        "            self.llm_config = Qwen2Config(**{k: v for k, v in llm_config.items() if k != 'architectures'})\n"
        "        else:\n"
    )
    if anchor not in text:
        print("  unipercept: expected anchor not found (upstream changed?) — skipping compat patch",
              flush=True)
        return
    cfg.write_text(text.replace(anchor, patched, 1))
    print("  unipercept: applied transformers-compat patch", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Clone SAM3 + UniPercept source into tool_models/.")
    ap.add_argument("--tools", nargs="*", choices=list(TOOLS), default=list(TOOLS),
                    help="subset to clone (default: all)")
    ap.add_argument("--force", action="store_true", help="remove and re-clone if already present")
    ap.add_argument("--no-install", action="store_true",
                    help="clone only; skip the sam3 editable pip install")
    args = ap.parse_args()

    if shutil.which("git") is None:
        raise SystemExit("git not found on PATH — install git first.")

    print(f"[setup_tools] tool_models={TOOL_DIR}  tools={args.tools}", flush=True)
    ok = all(clone_one(n, args.force) for n in args.tools)
    if ok and "unipercept" in args.tools:
        patch_unipercept()  # modern-transformers compat (idempotent)
    if ok and not args.no_install:
        for n in args.tools:
            if TOOLS[n][3]:  # pip-installable
                ok = pip_install_editable(n) and ok
    print("[setup_tools] done. Next: python download_checkpoints.py", flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
