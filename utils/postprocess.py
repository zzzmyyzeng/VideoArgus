"""Deterministic, input-gated post-processing of a generated rubric.

The generator obeys the importance/hard-cap rules in the system prompt only ~2/3 of the time,
so we enforce them programmatically after generation. The rules are keyed ENTIRELY on what the
input bundle contains (subject reference image? source video?), never on a task-type string, so
one function reproduces the correct behavior for all five task families:

  - text only              (no subject, no source)  -> no post-processing
  - first-frame only (I2V) (no subject, no source)  -> no post-processing
  - subject only (TS2V)    (subject, no source)     -> reference-conditioned down-weight
  - source only (TV2V)     (no subject, source)     -> reference-conditioned down-weight
  - subject + source (TSV2V)                        -> down-weight + edit-priority/degeneracy

Criteria are always KEPT (faithful to the input); only their importance weight and hard_cap
value are adjusted.
"""
from __future__ import annotations

# reference-conditioned weak-discriminator dimensions -> capped to "low"
_DOWNWEIGHT_DIMS = {5, 6, 8, 10}  # generic action / generic scene / temporal order / camera motion

# preservation-criterion id keywords (a preservation criterion the model may not have tagged dim 19)
_PRESERVE_KW = ("preserv", "consistency", "consisten", "unchanged", "background_preserved",
                "scene_preserved", "structure_motion", "motion_preserved", "temporal_consistency")

_ORDER = {"low": 0, "medium": 1, "high": 2}


def postprocess(rubric: dict, *, has_subject: bool, has_source_video: bool) -> dict:
    """Apply the input-gated importance/hard-cap rules in place and return the rubric.

    `has_subject`      = a subject reference image is present in the input bundle.
    `has_source_video` = a source video is present in the input bundle.
    """
    criteria = rubric.get("criteria", [])

    # (1) Reference-conditioned down-weight — applies whenever the task is anchored to a subject
    #     reference OR a source video (subject-driven generation or video editing). Camera / generic
    #     scene / generic action / temporal-order dimensions are weak discriminators there, so their
    #     importance is capped to "low". Pure text-to-video / plain image-to-video is left untouched.
    if has_subject or has_source_video:
        for c in criteria:
            if c.get("dimension") in _DOWNWEIGHT_DIMS and c.get("importance") != "low":
                c["importance"] = "low"

    # (2) Edit-priority / degeneracy enforcement — applies ONLY to subject-insertion editing, i.e.
    #     BOTH a source video AND a subject reference present. Plain text-editing (source only) falls
    #     back to rule (1) alone.
    if has_source_video and has_subject:
        for c in criteria:
            dim = c.get("dimension")
            cid = str(c.get("id", "")).lower()
            is_preserve = (dim == 19) or any(k in cid for k in _PRESERVE_KW)
            if dim in (18, 21):
                # edit-following (18) + inserted-subject identity (21) -> high; fundamental hard fail -> cap<=2
                if _ORDER.get(c.get("importance"), 0) < _ORDER["high"]:
                    c["importance"] = "high"
                if c.get("type") == "hard" and c.get("hard_cap") is not None and float(c["hard_cap"]) > 2:
                    c["hard_cap"] = 2
            elif dim == 17:
                # subject identity preservation hard break also catastrophic on editing -> cap<=2
                if c.get("type") == "hard" and c.get("hard_cap") is not None and float(c["hard_cap"]) > 2:
                    c["hard_cap"] = 2
            elif is_preserve:
                # preservation criteria capped at medium; hard fail severe-but-not-catastrophic -> cap<=4
                if _ORDER.get(c.get("importance"), 0) > _ORDER["medium"]:
                    c["importance"] = "medium"
                if c.get("type") == "hard" and c.get("hard_cap") is not None and float(c["hard_cap"]) > 4:
                    c["hard_cap"] = 4
    return rubric
