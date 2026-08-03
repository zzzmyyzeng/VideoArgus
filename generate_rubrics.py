#!/usr/bin/env python
"""Generate sample-specific evaluation rubrics for a set of video-generation inputs.

Rubric induction is decoupled from any specific model provider: one unified prompt +
input-gated post-processing produces a rubric from the INPUT BUNDLE alone (text + optional
subject reference image(s) / first frame / source video). It NEVER sees a generated output
video — that is the fairness guarantee that lets the same rubric score every model.

This script owns provider dispatch. Three official SDKs are supported, each reading its
STANDARD environment variable for credentials (no gateway, no base_url override):

    --provider openai      OpenAI Python SDK            OPENAI_API_KEY
    --provider anthropic   Anthropic Python SDK         ANTHROPIC_API_KEY
    --provider gemini      Google Gen AI SDK            GEMINI_API_KEY
    --provider auto        infer from --model name (default):
                             gpt* / o[0-9]* -> openai, claude* -> anthropic, gemini* -> gemini

Usage:
    python generate_rubrics.py \
        --input  VideoArgusBench/TSV2V/input.jsonl \
        --output my_rubrics/TSV2V \
        --model  gpt-5.6 --provider auto

    # infer per-record task, or pin one with --task; media paths in the input jsonl are
    # resolved relative to --media-root (defaults to the input file's directory).

Input records (one JSON per line) use the clean input schema:
    {"id": "...", "input": "<text/instruction>",
     "subject_images": ["images/<id>.jpg", ...],   # optional
     "first_frame": "images/<id>.jpg",              # optional
     "source_video": "videos/<id>.mp4"}             # optional

Output: <output>/<id>.json  =  {task_understanding, dimensions_considered, criteria,
                                _meta:{id, task}}  (post-processed, version-agnostic).
"""
from __future__ import annotations

import os
import re
import sys
import json
import argparse

from videoargus.rubric_induction import InputBundle, build_messages, postprocess

TASKS = ["T2V", "TI2V", "TS2V", "TV2V", "TSV2V"]
DEFAULT_MAX_TOKENS = int(os.environ.get("VIDARGUS_MAX_TOKENS", "16000"))


# --------------------------------------------------------------------------- #
# provider inference + JSON extraction
# --------------------------------------------------------------------------- #
def infer_provider(model: str) -> str:
    m = model.lower()
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith("gemini"):
        return "gemini"
    if m.startswith("gpt") or re.match(r"^o\d", m):
        return "openai"
    raise SystemExit(f"cannot infer provider from model '{model}'; pass --provider explicitly")


def _strip_json(raw: str) -> dict:
    """Parse JSON from a model response, tolerating ```json fences / stray prose."""
    t = (raw or "").strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    if "```" in t:
        seg = t.split("```")
        t = seg[1][4:].strip() if len(seg) > 1 and seg[1].lower().startswith("json") else \
            (seg[1].strip() if len(seg) > 1 else t)
    if "{" in t and "}" in t:
        t = t[t.index("{"): t.rindex("}") + 1]
    return json.loads(t)


def infer_task(rec: dict) -> str:
    """Derive the task family from which media roles the record carries."""
    has_subj = bool(rec.get("subject_images"))
    has_ff = bool(rec.get("first_frame"))
    has_sv = bool(rec.get("source_video"))
    if has_subj and has_sv:
        return "TSV2V"
    if has_sv:
        return "TV2V"
    if has_subj:
        return "TS2V"
    if has_ff:
        return "TI2V"
    return "T2V"


# --------------------------------------------------------------------------- #
# provider calls — each takes (system_prompt, user_content) from build_messages
# and returns the raw JSON string. Official SDKs, standard env creds.
# --------------------------------------------------------------------------- #
def call_openai(system_prompt, content, model, max_tokens):
    from openai import OpenAI
    client = OpenAI()  # reads OPENAI_API_KEY
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": content}],
        response_format={"type": "json_object"},
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


def call_anthropic(system_prompt, content, model, max_tokens):
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    # translate OpenAI-style content blocks -> Anthropic content blocks
    blocks = []
    for c in content:
        if c["type"] == "text":
            blocks.append({"type": "text", "text": c["text"]})
        else:  # image_url with a data: URL
            url = c["image_url"]["url"]
            header, b64 = url.split(",", 1)
            media_type = header.split(";")[0][len("data:"):] or "image/png"
            blocks.append({"type": "image",
                           "source": {"type": "base64", "media_type": media_type, "data": b64}})
    # ask for raw JSON in the system prompt; Anthropic has no json_object mode
    sys_json = system_prompt + "\n\nReturn ONLY the JSON object, no prose, no code fences."
    resp = client.messages.create(
        model=model, system=sys_json, max_tokens=max_tokens,
        messages=[{"role": "user", "content": blocks}],
    )
    return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")


def call_gemini(system_prompt, content, model, max_tokens):
    from google import genai
    from google.genai import types
    client = genai.Client()  # reads GEMINI_API_KEY
    parts = []
    for c in content:
        if c["type"] == "text":
            parts.append(types.Part.from_text(text=c["text"]))
        else:
            url = c["image_url"]["url"]
            header, b64 = url.split(",", 1)
            import base64 as _b64
            media_type = header.split(";")[0][len("data:"):] or "image/png"
            parts.append(types.Part.from_bytes(data=_b64.b64decode(b64), mime_type=media_type))
    resp = client.models.generate_content(
        model=model, contents=[types.Content(role="user", parts=parts)],
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            max_output_tokens=max_tokens,
        ),
    )
    return resp.text


PROVIDER_FN = {"openai": call_openai, "anthropic": call_anthropic, "gemini": call_gemini}


# --------------------------------------------------------------------------- #
def load_records(input_path: str):
    """Read records from a .jsonl file, or from a <task>/input.jsonl inside a bench task dir."""
    if os.path.isdir(input_path):
        cand = os.path.join(input_path, "input.jsonl")
        if not os.path.exists(cand):
            cand = os.path.join(input_path, os.path.basename(input_path.rstrip("/")) + ".jsonl")
        input_path = cand
    recs = [json.loads(l) for l in open(input_path) if l.strip()]
    return recs, input_path


def resolve_media(rec: dict, media_root: str) -> dict:
    """Return absolute media paths for one record (relative to media_root)."""
    def _abs(p):
        return p if os.path.isabs(p) else os.path.join(media_root, p)
    return {
        "subject_images": [_abs(p) for p in rec.get("subject_images", [])],
        "first_frame": _abs(rec["first_frame"]) if rec.get("first_frame") else None,
        "source_video": _abs(rec["source_video"]) if rec.get("source_video") else None,
    }


def main():
    ap = argparse.ArgumentParser(description="Generate sample-specific video-eval rubrics.")
    ap.add_argument("--input", required=True, help="input .jsonl (or a bench task dir with input.jsonl)")
    ap.add_argument("--output", required=True, help="output directory for <id>.json rubrics")
    ap.add_argument("--model", required=True, help="model name/tag (e.g. gpt-5.6, claude-opus-4-8, gemini-3-pro)")
    ap.add_argument("--provider", choices=["auto", "openai", "anthropic", "gemini"], default="auto")
    ap.add_argument("--task", choices=TASKS, help="pin the task for all records (else inferred per record)")
    ap.add_argument("--media-root", default=None, help="root for relative media paths (default: input file's dir)")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--ids", nargs="*", help="only generate for these ids")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    provider = args.provider if args.provider != "auto" else infer_provider(args.model)
    call_fn = PROVIDER_FN[provider]

    recs, resolved_input = load_records(args.input)
    media_root = args.media_root or os.path.dirname(os.path.abspath(resolved_input))
    os.makedirs(args.output, exist_ok=True)
    if args.ids:
        keep = set(args.ids)
        recs = [r for r in recs if r["id"] in keep]

    print(f"[generate_rubrics] provider={provider} model={args.model} n={len(recs)} "
          f"media_root={media_root} -> {args.output}", flush=True)

    done = fail = skip = 0
    for rec in recs:
        vid = rec["id"]
        out_path = os.path.join(args.output, f"{vid}.json")
        if os.path.exists(out_path) and not args.overwrite:
            skip += 1
            continue
        task = args.task or infer_task(rec)
        media = resolve_media(rec, media_root)
        bundle = InputBundle(task_type=task, text=rec.get("input", ""),
                             subject_images=media["subject_images"],
                             first_frame=media["first_frame"],
                             source_video=media["source_video"])
        try:
            system_prompt, content = build_messages(bundle)
            raw = call_fn(system_prompt, content, args.model, args.max_tokens)
            rubric = _strip_json(raw)
            rubric = postprocess(rubric, bundle)
            # normalize output envelope: keep only the four canonical top-level fields
            out = {
                "task_understanding": rubric.get("task_understanding", ""),
                "dimensions_considered": rubric.get("dimensions_considered", []),
                "criteria": rubric.get("criteria", []),
                "_meta": {"id": vid, "task": task},
            }
            with open(out_path, "w") as f:
                json.dump(out, f, indent=1)
            done += 1
            print(f"  {vid} [{task}]: {len(out['criteria'])} criteria", flush=True)
        except Exception as e:
            fail += 1
            print(f"  {vid} [{task}]: ERROR {type(e).__name__}: {str(e)[:200]}", flush=True)

    print(f"[generate_rubrics] done={done} skipped={skip} failed={fail}", flush=True)
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
