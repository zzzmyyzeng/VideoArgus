"""VideoArgus tool executor — input-bundle resolution + ExecutionContext.

Resolves, for a given (task, model, id):
  - the output video under evaluation
  - the input bundle media (subject_images / first_frame / source_video), by FILE EXTENSION
    (NOT list position — TSV2V lists the image before the video).

Holds a per-video ExecutionContext: a media_registry mapping the rubric's media identifiers
(output_video / source_video / first_frame / subject_image / reference / <step>_crop) to
on-disk resources, plus a frame cache so each video is decoded once.
"""
from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from typing import Optional

from .paths import root as _root

# Benchmark root: contains <task>/manifest.jsonl + <task>/media/. Points at VideoArgusBench by
# default and is env-overridable so an alternate benchmark split runs through the same pipeline.
RUBRIC_ROOT = _root("VIDARGUS_RUBRIC_ROOT", "VideoArgusBench")
# Root under which generated (output) videos to evaluate live: <task>/<model>/<id>.mp4
VIDEO_ROOT = _root("VIDARGUS_VIDEO_ROOT", "videos_under_eval")

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
VID_EXT = (".mp4", ".mov", ".avi", ".webm", ".mkv")

# which input roles each task carries (used by the repair layer to know what media SHOULD exist;
# actual presence is always re-checked from the manifest).
TASK_ROLES = {
    "T2V":   set(),
    "TI2V":  {"first_frame"},
    "TS2V":  {"subject_image"},
    "TV2V":  {"source_video"},
    "TSV2V": {"subject_image", "source_video"},
}


@dataclass
class Bundle:
    task: str
    id: str
    text: str
    subject_images: list = field(default_factory=list)  # abs paths
    first_frame: Optional[str] = None                   # abs path
    source_video: Optional[str] = None                  # abs path

    def has(self, role: str) -> bool:
        if role == "subject_image":
            return len(self.subject_images) > 0
        if role == "first_frame":
            return self.first_frame is not None
        if role == "source_video":
            return self.source_video is not None
        return False


def _load_manifest(task: str) -> dict:
    path = os.path.join(RUBRIC_ROOT, task, "manifest.jsonl")
    rows: dict = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[row["id"]] = row
    return rows


_MANIFEST_CACHE: dict = {}


def get_manifest(task: str) -> dict:
    if task not in _MANIFEST_CACHE:
        _MANIFEST_CACHE[task] = _load_manifest(task)
    return _MANIFEST_CACHE[task]


def resolve_bundle(task: str, vid: str) -> Bundle:
    """Build the input bundle for one id, classifying media by extension."""
    row = get_manifest(task)[vid]
    media_dir = os.path.join(RUBRIC_ROOT, task)
    imgs: list = []
    vids: list = []
    for rel in row.get("media", []):
        p = os.path.join(media_dir, rel)
        ext = os.path.splitext(rel)[1].lower()
        if ext in IMG_EXT:
            imgs.append(p)
        elif ext in VID_EXT:
            vids.append(p)
    b = Bundle(task=task, id=vid, text=row.get("text", ""))
    roles = TASK_ROLES[task]
    if "subject_image" in roles:
        b.subject_images = imgs
    elif "first_frame" in roles:
        b.first_frame = imgs[0] if imgs else None
    if "source_video" in roles:
        b.source_video = vids[0] if vids else None
    return b


def output_video_path(task: str, model: str, vid: str) -> str:
    return os.path.join(VIDEO_ROOT, task, model, f"{vid}.mp4")


class ExecutionContext:
    """Per (task, model, id) state shared across a video's criteria.

    media_registry maps a rubric media identifier -> a resource descriptor:
        {"kind": "video"|"image"|"images", "path": str | "paths": list[str]}
    Crops and web/reference media are added dynamically during execution.
    Frames are decoded lazily and cached per video path.
    """

    def __init__(self, task: str, model: str, vid: str):
        self.task = task
        self.model = model
        self.id = vid
        self.bundle = resolve_bundle(task, vid)
        self.output_video = output_video_path(task, model, vid)
        self._frame_cache: dict = {}      # video path -> [PIL]
        self.media_registry: dict = {}
        self._init_registry()

    def _init_registry(self):
        reg = self.media_registry
        reg["output_video"] = {"kind": "video", "path": self.output_video}
        if self.bundle.source_video:
            reg["source_video"] = {"kind": "video", "path": self.bundle.source_video}
        if self.bundle.first_frame:
            reg["first_frame"] = {"kind": "image", "path": self.bundle.first_frame}
        if self.bundle.subject_images:
            reg["subject_image"] = {"kind": "images", "paths": list(self.bundle.subject_images)}

    # -- presence checks used by the repair layer ----------------------------
    def available_media(self) -> set:
        """Base media identifiers present BEFORE any tool runs."""
        return set(self.media_registry.keys())

    def has_media(self, ident: str) -> bool:
        return ident in self.media_registry

    def register(self, ident: str, descriptor: dict):
        self.media_registry[ident] = descriptor

    def output_exists(self) -> bool:
        return os.path.exists(self.output_video)

    # -- frame access (decode once, cache) -----------------------------------
    def frames(self, video_ident_or_path: str, max_frames: int = 8):
        """Return a list of PIL.Image for a video identifier or raw path."""
        path = self._resolve_video_path(video_ident_or_path)
        if path is None:
            return []
        if path in self._frame_cache:
            frames = self._frame_cache[path]
        else:
            frames = _decode_frames(path, max_frames=max_frames)
            self._frame_cache[path] = frames
        return frames

    def _resolve_video_path(self, ident: str) -> Optional[str]:
        if os.path.sep in ident or ident.endswith(VID_EXT):
            return ident
        desc = self.media_registry.get(ident)
        if desc and desc.get("kind") == "video":
            return desc.get("path")
        return None

    def representative_image(self, ident: str):
        """Return a single representative PIL image for any media identifier.

        - image  -> the image
        - images -> first image
        - video  -> middle sampled frame
        - crop   -> the stored crop image
        """
        desc = self.media_registry.get(ident)
        if desc is None:
            return None
        kind = desc.get("kind")
        if kind == "image":
            from PIL import Image
            return Image.open(desc["path"]).convert("RGB")
        if kind == "images":
            from PIL import Image
            return Image.open(desc["paths"][0]).convert("RGB")
        if kind == "pil":           # in-memory crop
            return desc["image"]
        if kind == "video":
            fr = self.frames(ident)
            return fr[len(fr) // 2] if fr else None
        return None


def _decode_frames(path: str, max_frames: int = 8, fps: float = 2.0):
    """Decode a video to a list of PIL.Image (decord -> cv2 fallback), capped to max_frames.

    Frame density is overridable via env so the judge VLM can see temporal degeneracy
    (flicker / warping) that sparse sampling hides. Defaults:
      VIDARGUS_FRAME_FPS      (default 2.0)  — sampling rate
      VIDARGUS_MAX_FRAMES     (default 8)    — hard cap on returned frames
    """
    import os as _os
    fps = float(_os.environ.get("VIDARGUS_FRAME_FPS", fps))
    max_frames = int(_os.environ.get("VIDARGUS_MAX_FRAMES", max_frames))
    from PIL import Image
    frames = []
    try:
        import decord  # type: ignore
        vr = decord.VideoReader(path)
        native = vr.get_avg_fps() or 30.0
        step = max(1, int(round(native / fps)))
        for i in range(0, len(vr), step):
            frames.append(Image.fromarray(vr[i].asnumpy()).convert("RGB"))
    except Exception:
        import cv2  # type: ignore
        cv2.setNumThreads(0)
        cap = cv2.VideoCapture(path)
        native = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(native / fps)))
        i = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i % step == 0:
                frames.append(Image.fromarray(fr[:, :, ::-1]).convert("RGB"))
            i += 1
        cap.release()
    if len(frames) > max_frames:
        idx = [round(k * (len(frames) - 1) / (max_frames - 1)) for k in range(max_frames)]
        frames = [frames[k] for k in idx]
    return frames
