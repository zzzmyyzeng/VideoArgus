"""Rubric induction — build the provider-agnostic request for one input bundle.

Rubric induction is fully decoupled from the evaluation path and from any specific model provider.
This module does NOT call any API. It only:

  * `InputBundle`  — the inputs a user gave the video model (text + optional subject reference
    image(s) / first-frame image / source video), plus an advisory `task_type` label.
  * `build_messages(bundle)` — assemble the (system_prompt, user_content) pair, where user_content is
    an OpenAI-style multimodal content list (text blocks + base64 image_url blocks). The text comes
    from `utils.prompts` (a single unified prompt used for all five task families); images/frames are
    encoded by `utils.media`.
  * `postprocess(rubric, bundle)` — apply the deterministic, input-gated importance/hard-cap rules
    (delegates to `utils.postprocess`), keyed only on what the bundle contains.

The top-level `generate_rubrics.py` owns the actual provider dispatch (OpenAI / Anthropic / Gemini):
it takes the messages from `build_messages`, sends them through the chosen official SDK, parses the
returned JSON, then calls `postprocess`. Keeping the provider call out of this module is what makes
the same prompt+post-proc reproducible across every provider.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from utils.prompts import SYSTEM_PROMPT, SCHEMA_HINT, USER_PREFIX
from utils.media import img_path_to_data_url, video_to_frame_data_urls, VIDEO_FPS
from utils.postprocess import postprocess as _postprocess


@dataclass
class InputBundle:
    """The inputs a user provided to a video-generation model, for ONE sample.

    `task_type` is an advisory label only — the criteria are derived from the ACTUAL bundle
    contents (which visual inputs are attached), never from this string. Media are absolute or
    repo-relative file paths; only the roles that are present should be set.
    """
    task_type: str
    text: str
    subject_images: list = field(default_factory=list)  # subject reference image path(s) (TS2V/TSV2V)
    first_frame: Optional[str] = None                   # first-frame image path (TI2V)
    source_video: Optional[str] = None                  # source video path (TV2V/TSV2V)

    @property
    def has_subject(self) -> bool:
        return bool(self.subject_images)

    @property
    def has_source_video(self) -> bool:
        return self.source_video is not None


def build_messages(bundle: InputBundle, verbose: bool = False):
    """Return (system_prompt, user_content) for a rubric-generation request.

    user_content is an OpenAI-style list of {"type":"text",...} / {"type":"image_url",...} blocks:
    the task text + schema hint, then whichever visual inputs the bundle carries (subject reference
    image(s), first frame, source-video frames sampled in temporal order). Providers that don't take
    this shape directly (Anthropic, Gemini) adapt it in generate_rubrics.py.
    """
    content: list = [
        {"type": "text", "text": USER_PREFIX.format(task_type=bundle.task_type, text=bundle.text)},
        {"type": "text", "text": SCHEMA_HINT},
    ]
    if bundle.subject_images:
        content.append({"type": "text",
                        "text": f"[SUBJECT REFERENCE IMAGE(S)] ({len(bundle.subject_images)}):"})
        for p in bundle.subject_images:
            content.append({"type": "image_url", "image_url": {"url": img_path_to_data_url(p)}})
    if bundle.first_frame:
        content.append({"type": "text", "text": "[FIRST-FRAME IMAGE]:"})
        content.append({"type": "image_url", "image_url": {"url": img_path_to_data_url(bundle.first_frame)}})
    if bundle.source_video:
        urls = video_to_frame_data_urls(bundle.source_video)
        content.append({"type": "text",
                        "text": f"[SOURCE VIDEO] sampled at {VIDEO_FPS} fps, {len(urls)} frames in order:"})
        for u in urls:
            content.append({"type": "image_url", "image_url": {"url": u}})
    if verbose:
        n_imgs = sum(1 for c in content if c["type"] == "image_url")
        print(f"  [build_messages] task={bundle.task_type}  text+schema blocks + {n_imgs} image(s)")
    return SYSTEM_PROMPT, content


def postprocess(rubric: dict, bundle: InputBundle) -> dict:
    """Apply the deterministic input-gated importance/hard-cap rules for this bundle, in place.

    The generator obeys the prompt's importance/cap rules only ~2/3 of the time, so we enforce them
    programmatically. The rules are keyed entirely on bundle contents (subject reference? source
    video?), reproducing the correct per-task behavior for all five families with no task-type
    branch. See utils/postprocess.py.
    """
    return _postprocess(rubric, has_subject=bundle.has_subject,
                        has_source_video=bundle.has_source_video)
