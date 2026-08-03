"""Media -> data-URL helpers for rubric induction.

Images and (source) video frames are encoded as base64 PNG data URLs and attached to the
rubric-generation request. Video sampling: sample at `fps`, then uniformly downsample to at
most `max_frames` frames (a cost cap). decord is used if available, else OpenCV.
"""
from __future__ import annotations

import base64
from io import BytesIO

from PIL import Image

VIDEO_FPS = 2.0  # sample source video at 2 fps by default


def img_path_to_data_url(path: str, max_side: int = 768) -> str:
    img = Image.open(path).convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def video_to_frame_data_urls(path: str, fps: float = VIDEO_FPS, max_side: int = 768,
                             max_frames: int = 8) -> list:
    """Sample a video at `fps`, then UNIFORMLY downsample to at most max_frames (cost cap)."""
    frames = []
    try:
        import decord  # type: ignore
        vr = decord.VideoReader(path)
        native_fps = vr.get_avg_fps() or 30.0
        step = max(1, int(round(native_fps / fps)))
        idxs = list(range(0, len(vr), step))
        for i in idxs:
            arr = vr[i].asnumpy()
            frames.append(Image.fromarray(arr).convert("RGB"))
    except Exception:
        import cv2  # type: ignore
        cap = cv2.VideoCapture(path)
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(native_fps / fps)))
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                frame = frame[:, :, ::-1]  # BGR->RGB
                frames.append(Image.fromarray(frame).convert("RGB"))
            i += 1
        cap.release()

    # uniformly downsample to at most max_frames (cost cap)
    if len(frames) > max_frames:
        idx = [round(i * (len(frames) - 1) / (max_frames - 1)) for i in range(max_frames)]
        frames = [frames[i] for i in idx]
    urls = []
    for im in frames:
        if max(im.size) > max_side:
            im.thumbnail((max_side, max_side))
        buf = BytesIO()
        im.save(buf, format="PNG")
        urls.append("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())
    return urls
