"""VideoArgus tool executor — the 10 tool adapters + lazy heavy-model registry.

Tools (evidence_plan vocabulary):
  vlm_qa      {question, on}                 -> {answer}            (judge VLM)
  sam_track   {query, on}                    -> {found, coverage, frames}  (SAM3 video)
  crop_compare{crop_from, region}            -> {crop_id}          (SAM3 + PIL crop)
  dino_sim    {a, b}                         -> {cosine}           (DINOv3)
  ocr         {on, target}                   -> {texts}            (PaddleOCR)
  web_search  {query}                        -> {reference|unavailable} (Serper; needs SERPER_API_KEY)
  aesthetic   {on}                           -> {aesthetics, quality}    (UniPercept)
  depth_probe {on, query_a, query_b?}        -> {front_back, occlusion} (Depth-Anything-V2, per-frame)
  spatial_relation {on, entities, relation}  -> {verdict, count?}       (SAM3 + depth geometry)
  flicker_probe {on}                         -> {flicker_score, flagged} (dense RGB, no model)

The last three are CONTINUOUS (per-frame) checks: unlike vlm_qa which samples a few frames,
depth_probe / spatial_relation / flicker_probe densely read frames for geometric / signal
evidence (occlusion / left-right-front-back / count / temporal flicker) as a cross-check
alongside vlm_qa.

Heavy models (DINOv3, SAM3-image, SAM3-video, UniPercept, PaddleOCR) are lazy module-level
singletons loaded once and reused. Every adapter returns a JSON-serializable evidence dict
and is wrapped so an internal exception becomes {"ok": False, "error": ...} (non-fatal).

SAM3 inference MUST run under torch.autocast(cuda, bfloat16) (the add_prompt path already
wraps it internally; we still wrap image-model calls).

The judge VLM (vlm_qa + criterion judging) is served by an OpenAI-compatible endpoint POOL —
run your own local vLLM server(s) and point VIDARGUS_VLM_ENDPOINTS at them. No hosted-API
credentials are needed anywhere on the evaluation path.
"""
from __future__ import annotations

import os
import io
import base64
import tempfile
from typing import Optional

# Load an optional .env at the repo root + honor HF_TOKEN for gated checkpoints.
from .paths import UNIPERCEPT_SRC  # noqa: F401  (also triggers .env load via paths import)

_hf = os.environ.get("HF_TOKEN")
if _hf:
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", _hf)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import threading
# Serializes all GPU0 local-tool work (DINOv3/SAM3/UniPercept/Paddle singletons are NOT
# thread-safe, and crop->dino chains mutate ctx.media_registry). Held around a whole
# criterion's tool_sequence so its crops resolve atomically. VLM (HTTP) calls don't take it.
TOOL_LOCK = threading.Lock()

DINOV3_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DEPTH_ID = os.environ.get("VIDARGUS_DEPTH_ID", "depth-anything/Depth-Anything-V2-Small-hf")
# dense-frame sampling for continuous depth/rgb analysis (higher than vlm's ~8-frame sampling)
DENSE_FPS = float(os.environ.get("VIDARGUS_DENSE_FPS", "6"))
DENSE_MAX = int(os.environ.get("VIDARGUS_DENSE_MAX", "48"))

# Judge VLM served over an OpenAI-compatible local endpoint pool.
VLM_MODEL = os.environ.get("VIDARGUS_VLM_MODEL", "Qwen/Qwen3.6-27B")
VLM_API_KEY = os.environ.get("VIDARGUS_VLM_API_KEY", "EMPTY")
_VLM_ENDPOINTS = [u.strip() for u in os.environ.get("VIDARGUS_VLM_ENDPOINTS", "").split(",") if u.strip()]
# vlm_qa criterion-judge model label (kept as a name for logging; the served model is VLM_MODEL).
QA_MODEL = os.environ.get("VIDARGUS_QA_MODEL", VLM_MODEL)

_LOCAL_CLIENTS = {}     # url -> OpenAI(base_url=url)
_ENDPOINT_POOL = None   # queue.Queue of idle endpoint slots (a url may appear N times)
# How many concurrent in-flight requests per endpoint. =1 suits one TP=1 server per GPU;
# >1 lets a single large server (vLLM max_num_seqs batches) take several concurrent requests.
_SLOTS = int(os.environ.get("VIDARGUS_VLM_SLOTS_PER_ENDPOINT", "1"))
if _VLM_ENDPOINTS:
    import queue as _queue
    from openai import OpenAI as _OpenAI
    _ENDPOINT_POOL = _queue.Queue()
    for _u in _VLM_ENDPOINTS:
        _LOCAL_CLIENTS[_u] = _OpenAI(base_url=_u, api_key=VLM_API_KEY)
        for _ in range(_SLOTS):
            _ENDPOINT_POOL.put(_u)

# Qwen3.x thinking mode pollutes output; disable it (vLLM chat template kwarg).
_NOTHINK_EXTRA = {"chat_template_kwargs": {"enable_thinking": False}}


def _strip_think(text):
    """Defensively drop a leading <think>...</think> block if the server emitted one."""
    if not text:
        return text
    low = text.lower()
    if "<think>" in low and "</think>" in low:
        end = low.index("</think>") + len("</think>")
        return text[end:].strip()
    return text


def _is_retryable(msg):
    msg = msg.lower()
    return ("429" in msg or "rate" in msg or "budget" in msg or "overloaded" in msg
            or "timeout" in msg or "503" in msg or "502" in msg or "connection" in msg
            or "refused" in msg)


def chat_with_retry(messages, model=None, max_tokens=600, response_format=None,
                    tries=6, base_wait=8.0):
    """One chat.completions.create with exponential backoff on transient/429 errors.

    Picks an IDLE judge-VLM endpoint from the local pool, calls it, and returns it; on a
    dead/transient endpoint, retries on another. Thinking disabled; <think> stripped.
    Raises if all tries fail (or if no endpoint pool is configured).
    """
    import time
    if _ENDPOINT_POOL is None:
        raise RuntimeError(
            "No judge-VLM endpoint configured. Set VIDARGUS_VLM_ENDPOINTS to one or more "
            "OpenAI-compatible vLLM base URLs (comma-separated). evaluate.py sets this from "
            "--judge-endpoint.")
    last = None
    for i in range(tries):
        try:
            return _local_chat(messages, max_tokens, response_format)
        except Exception as e:
            last = e
            if not _is_retryable(str(e)) or i == tries - 1:
                raise
            time.sleep(min(base_wait * (2 ** i) + (i * 1.7), 180))
    raise last


def _local_chat(messages, max_tokens, response_format):
    """Check out an idle endpoint, call the judge VLM, check it back in. <think> stripped from
    the returned message content so downstream parsing/judging sees clean text."""
    url = _ENDPOINT_POOL.get()            # blocks until an endpoint is free
    try:
        client = _LOCAL_CLIENTS[url]
        kw = dict(model=VLM_MODEL, messages=messages, max_tokens=max_tokens)
        # The enable_thinking chat_template_kwarg is a vLLM-only extra_body arg; some
        # OpenAI-compatible servers reject unknown extra_body. Only send it to vLLM servers,
        # not to api.openai.com-style hosts.
        if "api.openai.com" not in url:
            kw["extra_body"] = dict(_NOTHINK_EXTRA)
        if response_format is not None:
            kw["response_format"] = response_format
        resp = client.chat.completions.create(**kw)
        try:
            msg = resp.choices[0].message
            if getattr(msg, "content", None):
                msg.content = _strip_think(msg.content)
        except Exception:
            pass
        return resp
    finally:
        _ENDPOINT_POOL.put(url)           # always return the endpoint to the pool

# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Lazy heavy-model registry
# --------------------------------------------------------------------------- #
class _Models:
    def __init__(self):
        self._dino = None
        self._dino_proc = None
        self._sam_img = None        # (processor)
        self._sam_vid = None
        self._uni = None            # (model, tokenizer, gen_cfg)
        self._paddle = None
        self._depth = None
        self._depth_proc = None

    # ---- Depth-Anything-V2 (per-frame metric-ish relative depth) ----
    def depth(self):
        if self._depth is None:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            self._depth_proc = AutoImageProcessor.from_pretrained(DEPTH_ID)
            self._depth = AutoModelForDepthEstimation.from_pretrained(DEPTH_ID).cuda().eval()
        return self._depth, self._depth_proc

    def depth_map(self, pil_img):
        """Return HxW float32 relative-depth array (larger = nearer, per Depth-Anything)."""
        import torch, numpy as np
        m, p = self.depth()
        x = p(images=pil_img.convert("RGB"), return_tensors="pt").to("cuda")
        with torch.inference_mode():
            d = m(**x).predicted_depth
        return d[0].float().cpu().numpy()

    # ---- DINOv3 ----
    def dino(self):
        if self._dino is None:
            import torch
            from transformers import AutoImageProcessor, AutoModel
            self._dino_proc = AutoImageProcessor.from_pretrained(DINOV3_ID)
            self._dino = AutoModel.from_pretrained(DINOV3_ID).cuda().eval()
        return self._dino, self._dino_proc

    def dino_embed(self, pil_img):
        import torch
        m, p = self.dino()
        x = p(images=pil_img.convert("RGB"), return_tensors="pt").to("cuda")
        with torch.inference_mode():
            return m(**x).pooler_output  # (1, D)

    # ---- SAM3 image (for single-frame bbox / crop) ----
    def sam_image(self):
        if self._sam_img is None:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            self._sam_img = Sam3Processor(build_sam3_image_model())
        return self._sam_img

    def sam_image_detect(self, pil_img, query: str):
        """Return list of (box_xyxy, score) for `query` on one PIL frame, best-first."""
        import torch
        pr = self.sam_image()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            st = pr.set_image(pil_img.convert("RGB"))
            out = pr.set_text_prompt(state=st, prompt=query)
        boxes = out.get("boxes")
        scores = out.get("scores")
        res = []
        if boxes is not None and len(boxes) > 0:
            import numpy as np
            # SAM3 under autocast returns bf16 tensors; .numpy() rejects bf16 -> .float() first.
            b = boxes.detach().float().cpu().numpy() if hasattr(boxes, "detach") else np.asarray(boxes, dtype=float)
            s = (scores.detach().float().cpu().numpy() if hasattr(scores, "detach")
                 else (np.asarray(scores, dtype=float) if scores is not None else None))
            for i in range(len(b)):
                res.append((b[i].tolist(), float(s[i]) if s is not None else 1.0))
            res.sort(key=lambda t: -t[1])
        return res

    # ---- SAM3 video predictor (tracking) ----
    def sam_video(self):
        if self._sam_vid is None:
            from sam3.model_builder import build_sam3_video_predictor
            self._sam_vid = build_sam3_video_predictor()
        return self._sam_vid

    # ---- UniPercept (aesthetic + quality) ----
    def unipercept(self):
        if self._uni is None:
            import sys
            if UNIPERCEPT_SRC not in sys.path:
                sys.path.append(UNIPERCEPT_SRC)
            import torch
            from transformers import AutoTokenizer
            from internvl.model.internvl_chat.modeling_unipercept import InternVLChatModel
            mp = "Thunderbolt215215/UniPercept"
            tok = AutoTokenizer.from_pretrained(mp, trust_remote_code=True, use_fast=False)
            m = InternVLChatModel.from_pretrained(
                mp, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                use_flash_attn=False).cuda().eval()
            gen_cfg = dict(max_new_tokens=512, do_sample=False)
            self._uni = (m, tok, gen_cfg)
        return self._uni

    def aesthetic_scores(self, pil_img):
        import torch
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        m, tok, gen_cfg = self.unipercept()
        MEAN = (0.485, 0.456, 0.406); STD = (0.229, 0.224, 0.225)
        tf = T.Compose([
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])
        px = tf(pil_img.convert("RGB")).unsqueeze(0).to(torch.bfloat16).cuda()
        dev = next(m.parameters()).device
        aes = m.score(dev, tok, px, gen_cfg, "aesthetics")
        qua = m.score(dev, tok, px, gen_cfg, "quality")
        return float(aes), float(qua)

    # ---- PaddleOCR ----
    def paddle(self):
        if self._paddle is None:
            # OCR runs on CPU by design: it is a light tool, and the paddlepaddle-gpu wheel would
            # downgrade the CUDA libraries torch 2.8.0 pins. oneDNN is disabled because paddle 3.3.x's
            # CPU oneDNN path crashes on the PP-OCRv6 detector (ConvertPirAttribute2RuntimeAttribute);
            # the default CPU backend runs fine without it.
            from paddleocr import PaddleOCR
            self._paddle = PaddleOCR(
                device="cpu", enable_mkldnn=False, use_doc_orientation_classify=False,
                use_doc_unwarping=False, use_textline_orientation=False)
        return self._paddle


MODELS = _Models()


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _pil_to_data_url(pil_img, max_side=768) -> str:
    img = pil_img.convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _save_tmp(pil_img) -> str:
    fd, path = tempfile.mkstemp(suffix=".png", prefix="te_")
    os.close(fd)
    pil_img.convert("RGB").save(path)
    return path


# --------------------------------------------------------------------------- #
# Tool adapters.  Each: (ctx, args, step_state) -> evidence dict
# step_state is a per-criterion scratch dict (lets crop_compare publish crops).
# --------------------------------------------------------------------------- #
def tool_vlm_qa(ctx, args, step_state) -> dict:
    question = args.get("question", "Describe what you see and judge the criterion.")
    on = args.get("on", "output_video")
    frames = ctx.frames(on) if ctx.has_media(on) else ctx.frames("output_video")
    content = [{"type": "text", "text": question}]
    for fr in frames:
        content.append({"type": "image_url", "image_url": {"url": _pil_to_data_url(fr)}})
    resp = chat_with_retry([{"role": "user", "content": content}], model=QA_MODEL, max_tokens=600)
    return {"tool": "vlm_qa", "on": on, "question": question,
            "answer": resp.choices[0].message.content}


def tool_sam_track(ctx, args, step_state) -> dict:
    """Track `query` across sampled frames using SAM3 image-model detection per frame.
    (Full video propagation is an available upgrade via predictor.handle_stream_request.)
    Records, per frame, whether the entity was detected + its best box; publishes the
    best (frame, box) into step_state for a following crop_compare.
    """
    query = args.get("query", "")
    on = args.get("on", "output_video")
    if not ctx.has_media(on):
        on = "output_video"
    frames = ctx.frames(on)
    per_frame = []
    best = None  # (score, frame_idx, box, pil)
    for i, fr in enumerate(frames):
        dets = MODELS.sam_image_detect(fr, query) if query else []
        found = len(dets) > 0
        sc = dets[0][1] if found else 0.0
        box = dets[0][0] if found else None
        per_frame.append({"frame": i, "found": found, "score": round(sc, 3)})
        if found and (best is None or sc > best[0]):
            best = (sc, i, box, fr)
    coverage = sum(1 for p in per_frame if p["found"]) / max(1, len(per_frame))
    if best is not None:
        step_state.setdefault("tracks", {})[on] = {
            "frame_idx": best[1], "box": best[2], "pil": best[3], "query": query}
    return {"tool": "sam_track", "query": query, "on": on,
            "found": best is not None, "coverage": round(coverage, 3),
            "n_frames": len(frames), "per_frame": per_frame}


def _crop_box(pil_img, box_xyxy, pad=0.12):
    w, h = pil_img.size
    x0, y0, x1, y1 = box_xyxy
    bw, bh = (x1 - x0), (y1 - y0)
    x0 = max(0, int(x0 - pad * bw)); y0 = max(0, int(y0 - pad * bh))
    x1 = min(w, int(x1 + pad * bw)); y1 = min(h, int(y1 + pad * bh))
    if x1 <= x0 or y1 <= y0:
        return pil_img
    return pil_img.crop((x0, y0, x1, y1))


def tool_crop_compare(ctx, args, step_state) -> dict:
    """Produce a crop of `region` from the named media and register it under an
    EXECUTOR-OWNED id: output_video -> 'output_crop', source_video -> 'source_crop'.
    Reuses a track from a preceding sam_track if available; else detects now.
    """
    crop_from = args.get("crop_from", "output_video")
    region = args.get("region", "")
    if not ctx.has_media(crop_from):
        crop_from = "output_video"
    crop_id = "source_crop" if crop_from == "source_video" else "output_crop"

    track = step_state.get("tracks", {}).get(crop_from)
    pil = None
    if track and (not region or region.strip() == track.get("query", "").strip() or True):
        if track.get("box") is not None and track.get("pil") is not None:
            pil = _crop_box(track["pil"], track["box"])
    if pil is None:
        # detect now on representative frame
        frames = ctx.frames(crop_from)
        rep = frames[len(frames) // 2] if frames else ctx.representative_image(crop_from)
        if rep is not None and region:
            dets = MODELS.sam_image_detect(rep, region)
            pil = _crop_box(rep, dets[0][0]) if dets else rep
        else:
            pil = rep
    if pil is None:
        return {"tool": "crop_compare", "ok": False, "error": "no media to crop",
                "crop_id": crop_id}
    ctx.register(crop_id, {"kind": "pil", "image": pil})
    return {"tool": "crop_compare", "crop_from": crop_from, "region": region,
            "crop_id": crop_id, "size": list(pil.size)}


def tool_dino_sim(ctx, args, step_state) -> dict:
    import torch.nn.functional as F
    a_id = args.get("a"); b_id = args.get("b")
    a_img = ctx.representative_image(a_id) if a_id else None
    b_img = ctx.representative_image(b_id) if b_id else None
    if a_img is None or b_img is None:
        return {"tool": "dino_sim", "ok": False,
                "error": f"missing operand image a={a_id}({a_img is not None}) b={b_id}({b_img is not None})"}
    ea = MODELS.dino_embed(a_img); eb = MODELS.dino_embed(b_img)
    cos = float(F.cosine_similarity(ea, eb).item())
    return {"tool": "dino_sim", "a": a_id, "b": b_id, "cosine": round(cos, 4)}


def tool_ocr(ctx, args, step_state) -> dict:
    on = args.get("on", "output_video")
    target = args.get("target", "")
    if not ctx.has_media(on):
        on = "output_video"
    frames = ctx.frames(on)
    rep = frames[len(frames) // 2] if frames else ctx.representative_image(on)
    if rep is None:
        return {"tool": "ocr", "ok": False, "error": "no media"}
    tmp = _save_tmp(rep)
    texts = []
    try:
        res = MODELS.paddle().predict(tmp)
        for r in res:
            # paddleocr 3.x returns dicts with 'rec_texts'
            d = r if isinstance(r, dict) else getattr(r, "json", {})
            rt = d.get("rec_texts") if isinstance(d, dict) else None
            if rt:
                texts.extend(list(rt))
    except Exception as e:
        return {"tool": "ocr", "ok": False, "error": str(e)[:200]}
    finally:
        try: os.remove(tmp)
        except OSError: pass
    return {"tool": "ocr", "on": on, "target": target, "texts": texts}


def tool_web_search(ctx, args, step_state) -> dict:
    """Serper image search -> download top image -> register as 'reference'.
    Until SERPER_API_KEY is set, returns unavailable (the repair layer degrades the
    consuming dino_sim; the criterion still gets judged via vlm_qa)."""
    query = args.get("query", "")
    key = os.environ.get("SERPER_API_KEY")
    if not key or not query:
        return {"tool": "web_search", "ok": False, "unavailable": True,
                "reason": "no SERPER_API_KEY" if not key else "no query", "query": query}
    try:
        import requests
        r = requests.post("https://google.serper.dev/images",
                          headers={"X-API-KEY": key, "Content-Type": "application/json"},
                          json={"q": query, "num": 5}, timeout=30)
        r.raise_for_status()
        items = r.json().get("images", [])
        img_url = next((it.get("imageUrl") for it in items if it.get("imageUrl")), None)
        if not img_url:
            return {"tool": "web_search", "ok": False, "unavailable": True,
                    "reason": "no image result", "query": query}
        from PIL import Image
        ib = requests.get(img_url, timeout=30).content
        pil = Image.open(io.BytesIO(ib)).convert("RGB")
        ctx.register("reference", {"kind": "pil", "image": pil})
        return {"tool": "web_search", "query": query, "image_url": img_url, "registered": "reference"}
    except Exception as e:
        return {"tool": "web_search", "ok": False, "unavailable": True,
                "reason": str(e)[:200], "query": query}


def tool_aesthetic(ctx, args, step_state) -> dict:
    on = args.get("on", "output_video")
    if not ctx.has_media(on):
        on = "output_video"
    frames = ctx.frames(on)
    rep = frames[len(frames) // 2] if frames else ctx.representative_image(on)
    if rep is None:
        return {"tool": "aesthetic", "ok": False, "error": "no media"}
    aes, qua = MODELS.aesthetic_scores(rep)
    return {"tool": "aesthetic", "on": on,
            "aesthetics_0_100": round(aes, 2), "quality_0_100": round(qua, 2)}


# --------------------------------------------------------------------------- #
# Continuous (per-frame) probes: depth / spatial-geometry / rgb-flicker.
# These densely decode frames (DENSE_FPS/DENSE_MAX) rather than the ~8-frame vlm sampling.
# --------------------------------------------------------------------------- #
def _dense_frames(ctx, on, cap=DENSE_MAX):
    """Decode up to `cap` frames at DENSE_FPS for continuous analysis (own decode, no 8-frame cache)."""
    from PIL import Image
    path = ctx._resolve_video_path(on) if hasattr(ctx, "_resolve_video_path") else None
    if path is None:
        # fall back to whatever ctx.frames gives (e.g. images)
        return ctx.frames(on, max_frames=cap) if ctx.has_media(on) else []
    import cv2
    cv2.setNumThreads(0)
    c = cv2.VideoCapture(path)
    native = c.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(native / DENSE_FPS)))
    frames = []; i = 0
    while True:
        ok, fr = c.read()
        if not ok:
            break
        if i % step == 0:
            frames.append(Image.fromarray(fr[:, :, ::-1]).convert("RGB"))
            if len(frames) >= cap:
                break
        i += 1
    c.release()
    return frames


def _sam_center_depth(pil_img, query, depth_map):
    """SAM-detect `query` on one frame; return (cx, cy, median_depth_in_box, box) or None."""
    dets = MODELS.sam_image_detect(pil_img, query) if query else []
    if not dets:
        return None
    box = dets[0][0]  # xyxy
    import numpy as np
    W, H = pil_img.size
    dh, dw = depth_map.shape
    sx, sy = dw / W, dh / H
    x0, y0, x1, y1 = [int(v) for v in box]
    dx0, dy0, dx1, dy1 = int(x0*sx), int(y0*sy), max(int(x1*sx), int(x0*sx)+1), max(int(y1*sy), int(y0*sy)+1)
    sub = depth_map[dy0:dy1, dx0:dx1]
    med = float(np.median(sub)) if sub.size else float("nan")
    return ((x0+x1)/2.0, (y0+y1)/2.0, med, box)


def tool_depth_probe(ctx, args, step_state) -> dict:
    """Per-frame depth via Depth-Anything-V2. Judges front/back of two entities and flags
    occlusion (mask overlap where depths cross). Larger depth value = nearer."""
    import numpy as np
    on = args.get("on", "output_video")
    if not ctx.has_media(on):
        on = "output_video"
    qa = args.get("query_a"); qb = args.get("query_b")
    frames = _dense_frames(ctx, on)
    if not frames:
        return {"tool": "depth_probe", "ok": False, "error": "no frames"}
    per = []; a_near = 0; b_near = 0; both = 0; occ = 0
    for i, fr in enumerate(frames):
        dm = MODELS.depth_map(fr)
        rec = {"frame": i}
        if qa and qb:
            ca = _sam_center_depth(fr, qa, dm); cb = _sam_center_depth(fr, qb, dm)
            if ca and cb:
                both += 1
                if ca[2] > cb[2]:
                    a_near += 1
                else:
                    b_near += 1
                # occlusion heuristic: boxes overlap AND depths are close/crossing
                ax0, ay0, ax1, ay1 = ca[3]; bx0, by0, bx1, by1 = cb[3]
                ox = max(0, min(ax1, bx1) - max(ax0, bx0)); oy = max(0, min(ay1, by1) - max(ay0, by0))
                if ox > 0 and oy > 0 and abs(ca[2] - cb[2]) < 0.08 * (abs(ca[2]) + abs(cb[2]) + 1e-6):
                    occ += 1
                rec.update({"a_depth": round(ca[2], 2), "b_depth": round(cb[2], 2)})
        per.append(rec)
    verdict = "ambiguous"
    if both:
        if a_near / both >= 0.6:
            verdict = f"{qa}_in_front"
        elif b_near / both >= 0.6:
            verdict = f"{qb}_in_front"
    return {"tool": "depth_probe", "on": on, "query_a": qa, "query_b": qb,
            "n_frames": len(frames), "both_detected_frames": both,
            "front_back": verdict, "occlusion_flag": (both > 0 and occ / max(1, both) >= 0.3),
            "occlusion_ratio": round(occ / max(1, both), 3) if both else 0.0,
            "per_frame": per[:16]}


def tool_spatial_relation(ctx, args, step_state) -> dict:
    """Geometric verification of a spatial claim via SAM (+depth for front/back).
    relation: 'left_right' | 'front_back' | 'count'."""
    import numpy as np
    on = args.get("on", "output_video")
    if not ctx.has_media(on):
        on = "output_video"
    rel = args.get("relation", "left_right")
    ents = args.get("entities") or []
    frames = _dense_frames(ctx, on)
    if not frames:
        return {"tool": "spatial_relation", "ok": False, "error": "no frames"}
    if rel == "count":
        target = ents[0] if ents else args.get("query", "")
        counts = []
        for fr in frames:
            dets = MODELS.sam_image_detect(fr, target) if target else []
            counts.append(len(dets))
        counts.sort()
        med = counts[len(counts)//2] if counts else 0
        return {"tool": "spatial_relation", "on": on, "relation": "count", "target": target,
                "count": int(med), "per_frame_counts": counts[:16], "n_frames": len(frames)}
    if rel == "left_right" and len(ents) >= 2:
        a, b = ents[0], ents[1]; a_left = 0; both = 0
        for fr in frames:
            da = MODELS.sam_image_detect(fr, a); db = MODELS.sam_image_detect(fr, b)
            if da and db:
                both += 1
                ax = (da[0][0][0] + da[0][0][2]) / 2; bx = (db[0][0][0] + db[0][0][2]) / 2
                if ax < bx:
                    a_left += 1
        verdict = "ambiguous"
        if both:
            verdict = f"{a}_left_of_{b}" if a_left/both >= 0.6 else (f"{b}_left_of_{a}" if a_left/both <= 0.4 else "ambiguous")
        return {"tool": "spatial_relation", "on": on, "relation": "left_right",
                "entities": [a, b], "verdict": verdict, "both_detected_frames": both, "n_frames": len(frames)}
    if rel == "front_back" and len(ents) >= 2:
        # delegate to depth_probe logic
        return tool_depth_probe(ctx, {"on": on, "query_a": ents[0], "query_b": ents[1]}, step_state)
    return {"tool": "spatial_relation", "ok": False, "error": f"bad relation/entities: {rel} {ents}"}


def tool_flicker_probe(ctx, args, step_state) -> dict:
    """Continuous RGB read to detect temporal flicker (high-frequency luma/color jumps).
    Pure numpy, no model — the finest-grained temporal-stability check."""
    import numpy as np
    on = args.get("on", "output_video")
    if not ctx.has_media(on):
        on = "output_video"
    frames = _dense_frames(ctx, on, cap=DENSE_MAX)
    if len(frames) < 3:
        return {"tool": "flicker_probe", "ok": False, "error": "too few frames"}
    lumas = []; means = []
    for fr in frames:
        a = np.asarray(fr, dtype=np.float32)  # HxWx3
        lumas.append(float((0.299*a[:,:,0] + 0.587*a[:,:,1] + 0.114*a[:,:,2]).mean()))
        means.append(a.reshape(-1, 3).mean(0))
    lumas = np.array(lumas); means = np.array(means)
    d_luma = np.abs(np.diff(lumas))                      # frame-to-frame luma change
    d_rgb = np.linalg.norm(np.diff(means, axis=0), axis=1)
    # flicker = high-frequency component: second difference magnitude (jitter, not smooth drift)
    dd = np.abs(np.diff(lumas, n=2)) if len(lumas) >= 3 else np.array([0.0])
    flicker_score = float(dd.mean())                      # avg jitter (0=smooth)
    flagged = [int(i) for i in np.where(d_luma > (d_luma.mean() + 3*d_luma.std() + 1e-6))[0]]
    return {"tool": "flicker_probe", "on": on, "n_frames": len(frames),
            "flicker_score": round(flicker_score, 3),
            "max_luma_delta": round(float(d_luma.max()), 3),
            "mean_luma_delta": round(float(d_luma.mean()), 3),
            "max_rgb_delta": round(float(d_rgb.max()), 3),
            "flagged_frames": flagged[:16],
            "mean_luma_series": [round(float(x), 1) for x in lumas[:32]]}


TOOL_FNS = {
    "vlm_qa": tool_vlm_qa,
    "sam_track": tool_sam_track,
    "crop_compare": tool_crop_compare,
    "dino_sim": tool_dino_sim,
    "ocr": tool_ocr,
    "web_search": tool_web_search,
    "aesthetic": tool_aesthetic,
    "depth_probe": tool_depth_probe,
    "spatial_relation": tool_spatial_relation,
    "flicker_probe": tool_flicker_probe,
}


def run_tool(name: str, ctx, args: dict, step_state: dict) -> dict:
    """Dispatch one tool step; never raises (errors captured into evidence)."""
    fn = TOOL_FNS.get(name)
    if fn is None:
        return {"tool": name, "ok": False, "error": "unknown tool"}
    try:
        ev = fn(ctx, args or {}, step_state)
        ev.setdefault("ok", True)
        return ev
    except Exception as e:  # non-fatal
        import traceback
        return {"tool": name, "ok": False, "error": str(e)[:300],
                "trace": traceback.format_exc()[-600:]}
