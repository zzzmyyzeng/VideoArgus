"""VideoArgus tool executor — tool-sequence repair / normalization layer.

Rubrics are auto-generated, so tool_sequences carry subtle, systematic errors.
This module turns a raw evidence_plan into an EXECUTABLE, type-correct tool_sequence
(or a graceful vlm_qa fallback), logging every repair. It NEVER raises — a plan that
cannot be normalized degrades to a single vlm_qa on the criterion text.

Pipeline:  deterministic rules R0..R9  ->  typecheck  ->  (optional) one LLM repair
           ->  vlm_qa fallback.

The repair operates on a SYMBOLIC view of media availability (what identifiers will
exist at each step), so it can be run with NO GPU / NO API for --dry-run-repair.

Media identifiers:
  base:    output_video, source_video, first_frame, subject_image
  derived: reference (from reference_source or web_search), output_crop, source_crop
"""
from __future__ import annotations

import json
import copy
from typing import Optional

TOOLS = {"vlm_qa", "sam_track", "crop_compare", "dino_sim", "ocr", "web_search", "aesthetic",
         "depth_probe", "spatial_relation", "flicker_probe"}  # v2 continuous geometric/signal tools

# editing-only dimensions: criteria that ONLY make sense with a source video
SOURCE_REQUIRED_DIMS = {18, 19, 20}        # edit-following / locality / structure preservation


def _step(tool, **args):
    return {"tool": tool, "args": dict(args)}


def _resolvable_reference(reference_source: str, base_media: set[str]) -> bool:
    """Can 'reference' be produced for this criterion?"""
    if reference_source == "subject_image":
        return "subject_image" in base_media
    if reference_source == "first_frame":
        return "first_frame" in base_media
    if reference_source == "source_video":
        return "source_video" in base_media
    if reference_source == "web_search":
        return True  # web_search step (or its degrade) handles it
    return False


def normalize_plan(criterion: dict, base_media: set[str], web_enabled: bool) -> dict:
    """Return a normalized, executable plan dict:
        {"tool_sequence": [...], "reference_source": str, "repairs": [...],
         "degraded": bool, "fallback": bool}
    `base_media` = identifiers present before tools run (output_video always; plus
    source_video/first_frame/subject_image when the bundle has them).
    """
    plan = criterion.get("evidence_plan", {}) or {}
    seq = copy.deepcopy(plan.get("tool_sequence") or [])
    ref_src = plan.get("reference_source", "none") or "none"
    web_query = plan.get("web_search_query")
    dim = criterion.get("dimension")
    repairs: list[str] = []

    has_source = "source_video" in base_media

    # ---- sanitize step shapes: drop non-dict / unknown-tool steps -------------
    clean = []
    for s in seq:
        if not isinstance(s, dict) or s.get("tool") not in TOOLS:
            repairs.append(f"drop malformed/unknown step: {str(s)[:60]}")
            continue
        s.setdefault("args", {})
        if not isinstance(s["args"], dict):
            s["args"] = {}
        clean.append(s)
    seq = clean

    # ---- R1: reference_source needs absent media -> set none, drop ref consumers
    if ref_src not in ("none", None) and not _resolvable_reference(ref_src, base_media):
        repairs.append(f"reference_source={ref_src} unresolvable (base media {sorted(base_media)}) -> none")
        ref_src = "none"

    # ---- R5: steps on source_video when no source video ----------------------
    if not has_source:
        new = []
        for s in seq:
            on = s.get("args", {}).get("on")
            cf = s.get("args", {}).get("crop_from")
            if on == "source_video" or cf == "source_video":
                if dim in SOURCE_REQUIRED_DIMS:
                    repairs.append(f"drop {s['tool']} on source_video (no source, dim {dim})")
                    continue
                else:
                    if on == "source_video":
                        s["args"]["on"] = "output_video"
                    if cf == "source_video":
                        s["args"]["crop_from"] = "output_video"
                    repairs.append(f"rewrite {s['tool']} source_video->output_video")
            new.append(s)
        seq = new

    # ---- web_search handling (R1/R2) -----------------------------------------
    has_web = any(s["tool"] == "web_search" for s in seq)
    needs_ref = ref_src == "web_search" or any(
        s["tool"] == "dino_sim" and s["args"].get("b") == "reference" for s in seq)

    if has_web and not (web_query or any(s["args"].get("query") for s in seq if s["tool"] == "web_search")):
        # web_search with no query anywhere -> drop the step
        seq = [s for s in seq if s["tool"] != "web_search"]
        repairs.append("drop web_search: no query")
        has_web = False

    if (ref_src == "web_search" or has_web) and not web_enabled:
        # web unavailable -> remove any web_search step and the reference it would feed;
        # convert reference-consuming dino_sim into a vlm recognition question.
        # Triggers even when reference_source=web_search has NO explicit web_search step
        # (otherwise dino_sim(b=reference) would dangle: 'reference' never gets registered).
        repairs.append("web_search disabled (no key) -> degrade to vlm recognition")
        seq, ref_src = _degrade_web_to_vlm(seq, criterion, repairs)
        has_web = False
        needs_ref = False

    if needs_ref and ref_src == "web_search" and web_enabled and not has_web:
        # dino_sim wants a web reference but no web_search step -> insert one
        q = web_query or criterion.get("evidence_plan", {}).get("target_entity") or criterion.get("criterion", "")[:60]
        seq = [_step("web_search", query=q)] + seq
        repairs.append("insert missing web_search before reference consumer")
        has_web = True

    # ---- fill web_search query from web_search_query/target_entity -----------
    for s in seq:
        if s["tool"] == "web_search" and not s["args"].get("query"):
            s["args"]["query"] = web_query or plan.get("target_entity") or criterion.get("criterion", "")[:60]
            repairs.append("fill web_search.query")

    # ---- R6/R7: crop ids + dangling dino operands ----------------------------
    seq = _repair_crops_and_dino(seq, ref_src, has_source, repairs)

    # ---- R9: ensure web_search precedes the dino_sim that needs reference -----
    seq = _reorder_web_before_dino(seq, repairs)

    # ---- R8: a crop with nothing consuming it -> append vlm_qa ----------------
    produced_crops = {c for s in seq if s["tool"] == "crop_compare"
                      for c in [_crop_id_for(s)]}
    consumed = set()
    for s in seq:
        if s["tool"] == "dino_sim":
            consumed.add(s["args"].get("a")); consumed.add(s["args"].get("b"))
    if produced_crops and not (produced_crops & consumed):
        if not any(s["tool"] == "vlm_qa" for s in seq):
            seq.append(_step("vlm_qa", question=_criterion_question(criterion), on="output_video"))
            repairs.append("crop produced but unconsumed -> append vlm_qa")

    # ---- R0: empty sequence -> vlm_qa fallback -------------------------------
    if not seq:
        repairs.append("empty tool_sequence -> vlm_qa fallback")
        return {"tool_sequence": [_step("vlm_qa", question=_criterion_question(criterion),
                                        on="output_video")],
                "reference_source": "none", "repairs": repairs,
                "degraded": True, "fallback": True}

    return {"tool_sequence": seq, "reference_source": ref_src,
            "repairs": repairs, "degraded": bool(repairs), "fallback": False}


def _crop_id_for(step) -> str:
    cf = step.get("args", {}).get("crop_from", "output_video")
    return "source_crop" if cf == "source_video" else "output_crop"


def _criterion_question(criterion: dict) -> str:
    c = criterion.get("criterion", "")
    fm = criterion.get("failure_modes") or []
    q = (f"Evaluate this criterion against the video: \"{c}\". "
         f"Consider failure modes: {fm}. Describe the evidence and how well it is met.")
    return q


def _degrade_web_to_vlm(seq, criterion, repairs):
    """Remove web_search; turn dino_sim(b=reference) into a vlm recognition question;
    return (seq, ref_src='none')."""
    out = []
    target = (criterion.get("evidence_plan", {}) or {}).get("target_entity") or criterion.get("criterion", "")[:60]
    for s in seq:
        if s["tool"] == "web_search":
            continue
        if s["tool"] == "dino_sim" and s["args"].get("b") == "reference":
            out.append(_step("vlm_qa",
                             question=f"Is the depicted entity unmistakably '{target}'? Judge recognizability and accuracy.",
                             on="output_video"))
            repairs.append("dino_sim(reference) -> vlm recognition (web disabled)")
            continue
        out.append(s)
    return out, "none"


def _repair_crops_and_dino(seq, ref_src, has_source, repairs):
    """Enforce executor-owned crop ids and fix dangling dino_sim operands.

    Symbolic availability of crops as we scan: a crop_compare on output_video
    publishes 'output_crop'; on source_video publishes 'source_crop'.
    'reference' is available iff ref_src != none (subject/first_frame/source/web).
    """
    ref_available = ref_src not in ("none", None)
    have = {"output_crop": False, "source_crop": False}
    out = []

    def ensure_crop(crop_id):
        cf = "source_video" if crop_id == "source_crop" else "output_video"
        out.append(_step("crop_compare", crop_from=cf, region=""))
        have[crop_id] = True
        repairs.append(f"insert missing crop_compare -> {crop_id}")

    for s in seq:
        t = s["tool"]
        if t == "crop_compare":
            cid = _crop_id_for(s)
            # normalize: rewrite source crop to output crop if no source
            if cid == "source_crop" and not has_source:
                s["args"]["crop_from"] = "output_video"
                cid = "output_crop"
                repairs.append("crop_compare source->output (no source)")
            have[cid] = True
            out.append(s)
            continue
        if t == "dino_sim":
            a = s["args"].get("a"); b = s["args"].get("b")
            # operand a: must be a crop. lenient mapping.
            a2 = _coerce_crop_operand(a)
            if a2 != a:
                repairs.append(f"dino_sim.a {a}->{a2}")
                a = a2
            if a in have and not have[a]:
                ensure_crop(a)
            elif a not in ("output_crop", "source_crop"):
                # unknown a -> default to output_crop, ensure it
                repairs.append(f"dino_sim.a unknown ({a}) -> output_crop")
                a = "output_crop"
                if not have[a]:
                    ensure_crop(a)
            # operand b: crop, reference, or source_crop
            if b == "reference":
                if not ref_available:
                    repairs.append("dino_sim.b=reference but no reference -> drop dino_sim")
                    continue
            else:
                b2 = _coerce_crop_operand(b)
                if b2 != b:
                    repairs.append(f"dino_sim.b {b}->{b2}")
                    b = b2
                if b in have and not have[b]:
                    if b == "source_crop" and not has_source:
                        repairs.append("dino_sim.b=source_crop but no source -> drop dino_sim")
                        continue
                    ensure_crop(b)
            s["args"]["a"] = a; s["args"]["b"] = b
            out.append(s)
            continue
        out.append(s)
    return out


def _coerce_crop_operand(x: Optional[str]) -> Optional[str]:
    if x is None:
        return x
    xl = str(x).lower()
    if "source" in xl and "crop" in xl:
        return "source_crop"
    if "crop" in xl or "output" in xl:
        return "output_crop"
    if xl == "reference":
        return "reference"
    return x


def _reorder_web_before_dino(seq, repairs):
    web_idx = [i for i, s in enumerate(seq) if s["tool"] == "web_search"]
    dino_ref_idx = [i for i, s in enumerate(seq)
                    if s["tool"] == "dino_sim" and s["args"].get("b") == "reference"]
    if web_idx and dino_ref_idx and web_idx[0] > dino_ref_idx[0]:
        w = seq.pop(web_idx[0])
        seq.insert(dino_ref_idx[0], w)
        repairs.append("reorder web_search before dino_sim(reference)")
    return seq


# --------------------------------------------------------------------------- #
# typecheck — verify the normalized sequence is executable as a SYMBOLIC trace
# --------------------------------------------------------------------------- #
def typecheck(plan: dict, base_media: set[str]) -> list[str]:
    errs: list[str] = []
    avail = set(base_media)
    if plan["reference_source"] not in ("none", None):
        avail.add("reference")  # will be produced (web_search) or resolved (subject/etc.)
    for s in plan["tool_sequence"]:
        t = s["tool"]; a = s.get("args", {})
        if t == "crop_compare":
            avail.add(_crop_id_for(s))
        elif t == "web_search":
            avail.add("reference")
        elif t == "dino_sim":
            for k in ("a", "b"):
                if a.get(k) not in avail:
                    errs.append(f"dino_sim.{k}={a.get(k)} not available (have {sorted(avail)})")
        elif t in ("vlm_qa", "sam_track", "ocr"):
            on = a.get("on", "output_video")
            if on not in base_media:
                errs.append(f"{t}.on={on} not a base medium")
        elif t == "aesthetic":
            pass
    return errs


# --------------------------------------------------------------------------- #
# optional LLM repair (only when deterministic normalization still type-fails)
# --------------------------------------------------------------------------- #
TOOL_CONTRACT = """\
Tools (each step is {"tool":<name>,"args":{...}}):
- vlm_qa {question, on:"output_video"|"source_video"}
- sam_track {query, on}
- crop_compare {crop_from:"output_video"|"source_video", region}  # produces output_crop / source_crop
- dino_sim {a:"output_crop"|"source_crop", b:"output_crop"|"source_crop"|"reference"}
- ocr {on, target}
- web_search {query}        # produces reference
- aesthetic {on:"output_video"}
A dino_sim operand must already exist: a crop comes from a preceding crop_compare;
"reference" comes from web_search OR from reference_source (subject_image/first_frame/source_video).
"""


def llm_repair(criterion: dict, plan: dict, base_media: set[str], errors: list[str]) -> Optional[dict]:
    """Ask the judge model to emit a corrected tool_sequence. Returns a plan dict or None."""
    try:
        from . import te_tools
        prompt = (
            "You are fixing a buggy tool plan for evaluating ONE criterion of a generated video.\n"
            f"CRITERION: {criterion.get('criterion','')}\n"
            f"DIMENSION: {criterion.get('dimension')}\n"
            f"AVAILABLE MEDIA (identifiers that already exist): {sorted(base_media)}\n"
            f"REFERENCE SOURCE: {plan.get('reference_source')}\n"
            f"{TOOL_CONTRACT}\n"
            f"CURRENT (still invalid) tool_sequence:\n{json.dumps(plan['tool_sequence'], indent=1)}\n"
            f"TYPECHECK ERRORS:\n- " + "\n- ".join(errors) + "\n\n"
            "Return ONLY JSON: {\"tool_sequence\": [ ... ]} that is executable given the available "
            "media. Prefer the minimal fix. If unsure, fall back to a single vlm_qa step that judges "
            "the criterion on output_video."
        )
        resp = te_tools.chat_with_retry(
            [{"role": "user", "content": prompt}],
            model=None,  # the local judge backend serves its configured model regardless
            response_format={"type": "json_object"},
            max_tokens=1500,
        )
        data = json.loads(resp.choices[0].message.content)
        new_seq = data.get("tool_sequence")
        if not isinstance(new_seq, list) or not new_seq:
            return None
        return {"tool_sequence": new_seq, "reference_source": plan.get("reference_source", "none"),
                "repairs": plan["repairs"] + ["llm_repair applied"],
                "degraded": True, "fallback": False}
    except Exception:
        return None


def repair(criterion: dict, base_media: set[str], web_enabled: bool, use_llm: bool = True) -> dict:
    """Top-level: normalize -> typecheck -> (llm repair) -> vlm_qa fallback. Never raises."""
    try:
        plan = normalize_plan(criterion, base_media, web_enabled)
    except Exception as e:
        return {"tool_sequence": [_step("vlm_qa", question=_criterion_question(criterion),
                                        on="output_video")],
                "reference_source": "none", "repairs": [f"normalize crashed: {e}"],
                "degraded": True, "fallback": True}
    if plan["fallback"]:
        return plan
    errs = typecheck(plan, base_media)
    if not errs:
        return plan
    if use_llm:
        fixed = llm_repair(criterion, plan, base_media, errs)
        if fixed is not None and not typecheck(fixed, base_media):
            return fixed
    # final fallback
    plan["tool_sequence"] = [_step("vlm_qa", question=_criterion_question(criterion), on="output_video")]
    plan["reference_source"] = "none"
    plan["repairs"].append(f"typecheck failed ({errs}); vlm_qa fallback")
    plan["degraded"] = True; plan["fallback"] = True
    return plan
