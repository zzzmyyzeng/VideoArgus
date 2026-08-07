# VideoArgus: Agentic Rubric-Grounded Unified Evaluation for Video Generation and Editing

📄 **[arXiv](https://arxiv.org/pdf/2608.05485)** &nbsp;·&nbsp; 🌐 **[Project page](https://zzzmyyzeng.github.io/VideoArgus/)** &nbsp;·&nbsp; 💻 **[GitHub](https://github.com/zzzmyyzeng/VideoArgus)** &nbsp;·&nbsp; 🤗 **[VideoArgusBench (HuggingFace)](https://huggingface.co/datasets/zengziyun/VideoArgusBench)**

VideoArgus scores a generated video against a rubric written *for that specific prompt* — not a fixed
global metric. For each input it generates a set of weighted, checkable criteria (with importance and
hard-cap semantics), then for each criterion an agent gathers evidence (CV tools + optional web image
search) and a judge VLM scores it 0–10. Per-criterion scores aggregate to a per-video final via an
importance-weighted mean with a soft hard-cap.

Two axes are deliberately decoupled:

* **Rubric generator** — rubrics can be produced by any strong LLM (GPT, Claude, or Gemini).
* **Scorer VLM** — scoring runs against *your own* local vLLM server (any OpenAI-compatible model).

The benchmark spans five conditioning settings:

| Task    | Conditioning                          |
|---------|---------------------------------------|
| `T2V`   | text                                  |
| `TI2V`  | text + first frame                    |
| `TS2V`  | text + subject image(s)               |
| `TV2V`  | text + source video                   |
| `TSV2V` | text + subject image + source video (edit)    |

Each rubric has 22 considered dimensions; which dimensions matter (and how hard they cap the score) is
gated by what the input actually provides.

---

## Install

**Conda (recommended):**

```bash
conda env create -f environment.yml
conda activate videoargus
```

Adjust `pytorch-cuda` in `environment.yml` to match your driver. Torch is pinned to **2.8.0**
(torchvision 0.23.0) — the version the CV tools are validated against; newer builds (2.10+) often
outpace driver/tool support.

**pip:** install torch **2.8.0** for your CUDA first (it ships on the cu126 / cu128 wheel indexes —
not cu124 — and its 12.6 runtime works on 12.x drivers), e.g.
`pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126`,
then `pip install -r requirements.txt`.

The CV-tool source (`sam3`, `UniPercept`) is cloned from upstream into `tool_models/` by `setup.sh`
(see [Setup](#setup)). SAM3 is then editable-installed into your environment (`pip install -e
tool_models/sam3`) so `import sam3` resolves; UniPercept is added to `sys.path` at runtime. Neither is
published on PyPI.

**Credentials:** copy `.env.example` to `.env` and fill in only what you use. `generate_rubrics.py`
reads the standard `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY`; the download scripts read
`HF_TOKEN`; the optional named-IP reference search reads `SERPER_API_KEY`. The evaluation path uses
**no** hosted-API credentials.

---

## Setup

The repo ships code only. Fetch the CV-tool source, model checkpoints, and the benchmark with one
command (run inside your activated environment):

```bash
bash setup.sh
```

This runs three steps, each also usable on its own:

```bash
python setup_tools.py          # clone SAM3 + UniPercept   -> tool_models/{sam3,UniPercept}
python download_checkpoints.py # DINOv3/Depth-Anything/SAM3/UniPercept weights -> tool_models/checkpoints/
python download_bench.py       # VideoArgusBench (inputs + rubrics) -> ./VideoArgusBench/
```

Skip a step with `bash setup.sh --no-bench` / `--no-ckpt` / `--no-tools`. `setup_tools.py` also
editable-installs SAM3 (`pip install -e tool_models/sam3`) so `import sam3` resolves. PaddleOCR
checkpoints auto-download on first use, and OCR runs on **CPU** (it is a light tool, and the
`paddlepaddle-gpu` wheel would downgrade the CUDA libraries torch 2.8.0 pins).

The judge VLM is served by you (next section) and is not fetched here.

---

## Quick start

### 1. (Optional) Regenerate rubrics with the LLM of your choice

The shipped rubrics are ready to use. To reproduce or swap the generator:

```bash
python generate_rubrics.py \
    --input VideoArgusBench/TSV2V/input.jsonl \
    --output my_rubrics/TSV2V \
    --model gpt-5.6 --provider auto \
    --media-root VideoArgusBench/TSV2V
```

`--provider auto` routes by model name (`gpt*`/`o*` → OpenAI, `claude*` → Anthropic,
`gemini*` → Gemini). The task is inferred per record from which conditioning inputs are present, or set
it explicitly with `--task`. Output is one `<id>.json` per input.

### 2. Serve a judge VLM (your own local vLLM)

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.6-27B --port 8000 --tensor-parallel-size 2
```

Any OpenAI-compatible, vision-capable model works. Run several servers and pass them comma-separated to
fan criteria out in parallel.

### 3. Evaluate your generated videos

Lay videos out as `<videos-dir>/<task>/<model>/<id>.mp4` (one `<id>` per benchmark input), then:

```bash
python evaluate.py \
    --bench VideoArgusBench --task TSV2V --model my_model \
    --videos-dir ./videos_under_eval \
    --judge-endpoint http://localhost:8000/v1 --judge-model Qwen/Qwen3.6-27B \
    --output ./eval_runs --run-name run1 \
    --report
```

Per-video results are written to `eval_runs/run1/<task>/<model>/<id>.json`. `--report` prints the
per-model mean at the soft-cap `--alpha` (default `0.5`) after scoring. To score against rubrics you
regenerated with a different LLM, keep them in a subfolder `<task>/rubrics/<name>/` and pass
`--rubric-subdir <name>`. Use `--shard/--num-shards` to split work across jobs.

---

## Scoring model

* **Importance weights:** `high=3`, `medium=2`, `low=1`; the base score is the importance-weighted mean
  of per-criterion 0–10 scores.
* **Hard caps:** a criterion marked `hard` that scores below the fail threshold caps the final score at
  its `hard_cap`. `--alpha` blends the capped and uncapped scores:
  `final = base·(1−α) + min(base, cap)·α`, so `α=1.0` is a strict hard cap and `α=0` ignores caps. All
  shipped reports use `α=0.5`.
* **Input-gated dimensions:** when the input supplies a subject or a source video, identity/appearance
  dimensions are down-weighted and (for edit tasks) preservation/consistency dimensions are promoted and
  capped, so a rubric only judges what the task actually asks for. This is applied uniformly in
  `utils/postprocess.py` for both freshly generated and shipped rubrics.

---

## Reproducing the benchmark

`VideoArgusBench/` ships the inputs and one rubric per input, but **not** the generated videos
(you supply your own model's outputs). To evaluate a model across all tasks, generate its videos into
`<videos-dir>/<task>/<model>/<id>.mp4` and run `evaluate.py` per task. To study rubric-generator
sensitivity, regenerate the rubrics with a different LLM via `generate_rubrics.py`, keep each set in its
own `<task>/rubrics/<name>/` subfolder, and score the same videos against each with `--rubric-subdir`.

The benchmark is also published on HuggingFace:
[**zengziyun/VideoArgusBench**](https://huggingface.co/datasets/zengziyun/VideoArgusBench). To fetch it
instead of using the copy in this repo:

```bash
huggingface-cli download zengziyun/VideoArgusBench --repo-type dataset --local-dir VideoArgusBench
```

---

## Citation

If you find VideoArgus useful, please cite our paper:

```bibtex
@article{zeng2026videoargus,
  title={VideoArgus: Agentic Rubric-Grounded Unified Evaluation for Video Generation and Editing},
  author={Zeng, Ziyun and Wang, Zixuan and Yu, Yongsheng and Hua, Hang and Luo, Jiebo},
  journal={arXiv preprint arXiv:2608.05485},
  year={2026}
}
```
