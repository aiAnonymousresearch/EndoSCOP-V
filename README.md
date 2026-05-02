# EndoSCOP-V

Multi-turn video Q&A benchmark for endoscopy and colonoscopy. This repository
ships an end-to-end evaluation pipeline that runs a configurable set of
multi-modal LLMs (local HuggingFace checkpoints + API models) against the
EndoSCOP-V cases, scores their answers, and produces a formatted Excel
report.

- 9 task families × 30 sub-tasks (disease ID, anatomical localization,
  morphology, grading, biopsy, action recognition, …).
- Both endoscopy (4 anatomical locations) and colonoscopy (9 locations).
- Multi-turn conversation per case: the model sees frames in turn 1, then
  answers a chain of follow-up questions referring back to those frames.
- Out of the box: Gemini 3 Flash Preview, OpenAI GPT-5.4-mini, Anthropic Claude,
  Qwen3-VL, Qwen3.5, MedGemma, Hulu-Med, ColonR1.

## Quick Start

```bash
# 0. Environment
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 1. Place dataverse_files.zip in data/
#    (download from Harvard Dataverse — see Step 1 below)

# 2. Unzip + organize + extract frames
bash prepare_data.sh

# 3. Edit config.yaml — set credentials, enable the models you want

# 4. Run all enabled models
python evaluate.py

# 5. Per-model JSON outputs land at:
#    evaluations/evaluated_cases/<Model_Name>/<case_id>.json

# 6. Consolidate to a single wide-format CSV
#    → evaluations/evaluated_cases/evaluated_cases_consolidated.csv
python consolidate_result_json.py evaluations/evaluated_cases/

# 7. Generate the formatted Excel summary
#    → evaluations/evaluated_cases/Evaluation_Summary.xlsx (next to the CSV)
python summarize.py evaluations/evaluated_cases/evaluated_cases_consolidated.csv
```

The final `Evaluation_Summary.xlsx` has six sheets: EndoColon, Task,
SubTask, Diseases, Task_Total, Disease_Total. Both scripts accept
`--output <path>` if you want the artifacts somewhere other than next
to the per-case JSONs.

## Directory Layout

```
distribution/
├── README.md                         (this file)
├── requirements.txt
├── config.yaml                       Single source of truth for paths, creds, models
├── prepare_data.sh                   Step 2 — unzip + organize + extract frames
├── evaluate.py                       Step 4 — run all enabled models
├── consolidate_result_json.py        Step 6 — predictions JSONs -> wide CSV
├── summarize.py                      Step 7 — CSV -> 6-sheet Excel
├── summarize_ex_DI-R.py              Optional — same workbook, DI-R rows excluded
├── eval/                             Evaluation engine
│   ├── run.py                        Single-model runner (used by evaluate.py)
│   ├── prompts.py                    System prompt + per-turn message builder
│   ├── scorer.py                     Answer extraction + scoring + aggregation
│   ├── extract_frames.py             Video -> 896×896 JPEGs
│   └── providers/
│       ├── base.py                   Abstract BaseProvider + ConversationSession
│       ├── transformers_provider.py  HuggingFace AutoModel (MedGemma, Qwen3-VL, …)
│       ├── hulumed_provider.py       Hulu-Med (custom Qwen-based architecture)
│       ├── colonr1_provider.py       ColonR1 (Qwen2.5-VL-3B GRPO, with weight self-heal)
│       ├── gemini.py                 Vertex AI / Google AI Studio
│       ├── openai_provider.py        Chat Completions API
│       └── anthropic_provider.py     Messages API with forced tool_use
├── data/
│   ├── dataverse_files.zip           Downloaded from Dataverse (gitignored)
│   ├── task_taxonomy.csv             Task/subtask display names
│   └── cases/                        Created by prepare_data.sh (gitignored)
└── evaluations/
    └── evaluated_cases/              (gitignored)
        ├── <Model_Name>/<case_id>.json    Per-model per-case predictions
        ├── evaluated_cases_consolidated.csv   Output of consolidate_result_json.py
        └── Evaluation_Summary.xlsx        Output of summarize.py
```

## Step 0 — Environment

- Python 3.10 or newer.
- For local GPU models: CUDA 12+ and a GPU with at least 24 GB VRAM
  (48 GB recommended for the 14 B+ models).
- Install PyTorch first from <https://pytorch.org> matching your CUDA
  toolkit, then `pip install -r requirements.txt`. The requirements file
  intentionally does not pin torch — installing it first lets you pick
  the wheel index URL for your CUDA version.

## Step 1 — Download the dataset

Download `dataverse_files.zip` from the EndoSCOP-V record on Harvard
Dataverse and place it at `data/dataverse_files.zip`. The path is
configurable via `paths.dataverse_zip` in `config.yaml`.

**Expected zip layout** — `prepare_data.sh` expects `qa.zip`,
`case_reports.zip`, and `videos/` to sit at the root of the archive:

```
dataverse_files.zip
├── qa.zip
├── case_reports.zip
└── videos/
    ├── <case_id>.mp4
    └── ...
```

If your download wraps everything in a top-level `dataverse_files/`
directory (some Dataverse downloads do), `prepare_data.sh` will report
`Total cases: 0` and exit silently. Re-zip the contents one level up so
the layout matches above before re-running.

## Step 2 — Prepare the data

```bash
bash prepare_data.sh
# bash prepare_data.sh --force          # if you need to re-do the unzip
# bash prepare_data.sh --zip /alt.zip   # if your zip lives elsewhere
```

This script:

1. Unzips the Dataverse archive into a temp directory.
2. Unpacks the nested `qa.zip`, `case_reports.zip`, and per-batch
   `videos/*.zip` archives.
3. Reorganizes everything into `data/cases/{ec\d+}/` per-case folders.
   Each folder ends up with `<case_id>.mp4`, `<case_id>_qa.json`, and
   `<case_id>_case_report.json`.
4. Calls `eval/extract_frames.py` to produce `<case_id>_siglip_896/`
   directories with up to 800 uniformly sampled 896×896 JPEGs.
   (Frames are pre-extracted once because every local-model run
   re-reads them — extracting on the fly per evaluation would be
   wasteful.)

The frame extractor handles the two source resolutions in the dataset
(1350×1080 and 900×720) with the appropriate crop/pad transform; other
resolutions get a center-crop fallback.

## Step 3 — Configure

Open `config.yaml`. The four sections you'll touch most:

**`credentials`** — env-var names (preferred) and optional
`*_file` fallbacks for the keys/tokens each provider needs.

For Gemini 3 Flash Preview, use **Vertex AI**: download a service-account
JSON from your GCP project and either set
`google_app_credentials_file:` to that path or
`export GOOGLE_APPLICATION_CREDENTIALS=...`. The provider auto-extracts
`project_id` from the JSON via `google.auth.default()` — no
`GOOGLE_CLOUD_PROJECT` or `vertex_project` setting needed. Older Gemini
models (2.x) still work with an AI Studio API key.

```yaml
credentials:
  hf_token_env: HF_TOKEN                    # Required for gated models (MedGemma, etc.)

  # Gemini — pick ONE:
  #   (A) AI Studio API key (free tier, simplest, Gemini 2.x):
  gemini_api_key_env: GOOGLE_API_KEY
  #   (B) Vertex AI service-account JSON (required for Gemini 3):
  # google_app_credentials_file: /abs/path/to/your_sa.json

  anthropic_api_key_env: ANTHROPIC_API_KEY
  openai_api_key_env: OPENAI_API_KEY
```

**`defaults`** — temperature, `max_new_tokens`, device. The default
device is `auto`; set `cuda:1` if your `cuda:0` is the display GPU.

**`frames`** — resolution, max frames extracted per video, parallel
workers for `extract_frames.py`. You'll rarely change these.

**`models`** — the entry that drives everything. Each entry:

```yaml
medgemma-4b:
  provider: transformers
  model: google/medgemma-4b-it
  results_name: MedGemma-4b-it       # Folder name under evaluated_cases/
  enabled: true                      # Toggle on/off
  frames: 200                        # Frames sampled from the 800-frame pre-extracted set
  procedure_filter: colonoscopy      # Optional — restrict to one procedure type
  notes: "Gated — accept the license at huggingface.co/google/medgemma-4b-it"
  provider_config:                   # Optional per-model overrides
    enable_thinking: false
```

Set `enabled: true` on the models you want to run; everything else stays
`false`. See the comments at the top of each `models:` block in
`config.yaml` for VRAM and frame-budget guidance.

## Step 4 — Evaluate

```bash
# Run every model with `enabled: true` in config.yaml
python evaluate.py

# Smoke test on one case across all enabled models
python evaluate.py --case-ids e1372

# Validate the pipeline without making API/model calls
python evaluate.py --dry-run --case-ids e1372

# Resume — skip cases that already have a JSON in evaluated_cases/<model>/
python evaluate.py --resume

# One-off ad-hoc run, ignoring the models: block
python evaluate.py --provider gemini --model gemini-3-flash-preview
```

`evaluate.py` iterates the enabled models, calls the per-model engine
(`eval/run.py`), and writes predictions flat to
`evaluations/evaluated_cases/<results_name>/<case_id>.json`.

For long overnight runs, prefer `nohup python evaluate.py > run.log 2>&1 &`
and monitor with `tail -F run.log`.

### Single-model smoke tests

For wiring up a new model, debugging a regression, or sanity-checking
one provider before kicking off the full sweep, run one model against
one or two cases:

```bash
# API model (no GPU)
python evaluate.py --provider gemini --model gemini-3-flash-preview --case-ids e1652

# Local model on a specific GPU
python evaluate.py --device cuda:1 --provider transformers --model google/medgemma-4b-it --case-ids e1652

# Override the frame budget for a context-limited model
python evaluate.py --device cuda:1 --provider hulumed --model ZJU-AI4H/Hulu-Med-4B --case-ids e1652 --frame-count 16
python evaluate.py --device cuda:1 --provider colonr1 --model ai4colonoscopy/ColonR1 --case-ids c1879 --frame-count 25
```

Things to know about single-model mode (`--provider X --model Y`):

- It bypasses the `models:` block in `config.yaml` and runs only the
  pair you name. If that exact `provider`/`model` pair appears in
  `models:`, the run inherits its `results_name`, `frames`, and
  `procedure_filter` from there; CLI flags override on top.
- **`--frame-count N` only takes effect in single-model mode.** Multi-model
  runs always read `frames:` from each model's entry. Use this when
  testing a model whose context budget is smaller than what's in config
  (Hulu-Med caps at 16, ColonR1 at 25 — exceed and you'll either OOM or
  silently truncate).
- **`--device` overrides `defaults.device`.** Use `cuda:1` if `cuda:0` is
  the display GPU; OOM on cuda:0 even when weights would fit is usually
  desktop overhead competing for the address space.
- **`procedure_filter` silently drops cases.** ColonR1 has
  `procedure_filter: colonoscopy`, so passing `--case-ids e1652`
  (an endoscopy case) yields "0 cases" with no error. Pick a `cNNNN`
  case for colonoscopy-only models.

Pass `-v` / `--verbose` to surface the third-party HTTP / auth logs that
are silenced by default.

## Step 5 — Per-case prediction format

```jsonc
{
  "case_id": "e1372",
  "procedure_type": "endoscopy",
  "diseases_found": ["Erosion"],
  "total_questions": 7,
  "results": [
    {
      "question_number": 1,
      "turn": 1,
      "phase": "disease_identification",
      "disease": "Erosion",
      "task": "DI",
      "subtask": "DI-M",
      "type": "MCQ",
      "multi_select": false,
      "correct_answer": "C",
      "model_answer": "C",
      "model_raw": "{\"answer\": \"C\"}",
      "is_correct": true
    }
  ],
  "case_accuracy": 0.857
}
```

## Step 6 — Consolidate

```bash
python consolidate_result_json.py evaluations/evaluated_cases/
# -> evaluations/evaluated_cases/evaluated_cases_consolidated.csv
#    (one row per question, one column per model, plus video / Q_Num /
#    scopy / disease / task / GT)

# Custom output path:
python consolidate_result_json.py evaluations/evaluated_cases/ \
    --output evaluations/consolidated_results.csv
```

If `model_answer` is null in a per-case JSON (the in-flight scorer's
regex fallback already failed), `consolidate_result_json.py` does a
final, more lenient pass over `model_raw` before giving up. Cells that
remain unrecoverable become empty.

By default, only models marked `enabled: true` in `config.yaml` are
included. Override with `--include-models A,B,C`.

## Step 7 — Summarize

```bash
python summarize.py evaluations/evaluated_cases/evaluated_cases_consolidated.csv
# -> evaluations/evaluated_cases/Evaluation_Summary.xlsx (sibling of input CSV)

# Custom output path:
python summarize.py evaluations/evaluated_cases/evaluated_cases_consolidated.csv \
    --output evaluations/Evaluation_Summary.xlsx
```

Six sheets:

1. **EndoColon** — models × {endoscopy, colonoscopy, both}, six accuracy
   metrics each (Binary, MCQ Single, MCQ Multi Strict, MCQ Multi Partial,
   Total Strict, Total Partial).
2. **Task** — models as columns, the 9 task families as row groups.
3. **SubTask** — same shape, 30 sub-tasks.
4. **Diseases** — same shape, per disease.
5. **Task_Total** — task and sub-task hierarchy collapsed to one Total
   Partial column per model.
6. **Disease_Total** — diseases (regular and special) collapsed to one
   column per model.

Multi-select MCQs are scored both strictly (exact set match) and
partially (+1/n per correct, −1/(2n) per spurious selection, floor at 0).

### Optional — summarize excluding DI-R

DI-R (Disease Identification — Recall) is 100% binary and accounts for
~15% of all rows / ~48% of the DI task. Including it inflates raw
accuracy and crowds out signal from harder question types. Run a
parallel summary that drops DI-R rows before computing metrics:

```bash
python summarize_ex_DI-R.py evaluations/evaluated_cases/evaluated_cases_consolidated.csv
# -> evaluations/evaluated_cases/Evaluation_Summary_ex_DI-R.xlsx
```

The output workbook has the same six sheets as `Evaluation_Summary.xlsx`,
just with DI-R filtered out. Same `--output` and `--include-models`
flags as `summarize.py`. Useful when ranking models on the harder
sub-tasks (DI-M, DI-C, AL-*, MO-*, BX-*, AR-*, GR-*, etc.) without
the DI-R floor inflating everything.

## Models

| Provider | Models | Input | Notes |
|---|---|---|---|
| `gemini` | gemini-3-flash-preview | raw mp4 (native video) | Vertex AI service-account JSON or Google AI Studio key. |
| `openai` | gpt-5.4-mini (and other Chat Completions models) | base64 frames | Set `OPENAI_API_KEY`. Uses `max_completion_tokens` (gpt-5+ standard). |
| `anthropic` | claude-sonnet-4-6 (and others) | base64 frames | Set `ANTHROPIC_API_KEY`. Forced tool_use → structured JSON. |
| `transformers` | MedGemma-4b/1.5-4b/27b, Qwen3-VL-8b/32b, Qwen3.5-4b/9b/27b | pre-extracted JPEGs | Standard HF chat template path. |
| `hulumed` | Hulu-Med-4B/7B/14B/32B | pre-extracted JPEGs | 16-frame cap (16K context). `trust_remote_code: true`. |
| `colonr1` | ai4colonoscopy/ColonR1 | pre-extracted JPEGs | Colonoscopy-only; 32 K context. Weight-filename self-heal built in. |

## Hardware Notes

Frame budgets are per-case. Higher budgets give the model more visual
detail but consume more VRAM and run slower (the vision encoder
re-runs every turn). 

If `cuda:0` is your display GPU, override per-run with `--device cuda:1`
or set `defaults.device: cuda:1` in `config.yaml`. Heavy models with
contiguous activation allocations OOM on cuda:0 even when their
steady-state weights would fit, because the desktop overhead competes
for the same address space.

## Troubleshooting

- **`GatedRepoError` on model load** — visit the model's HuggingFace
  page while signed in and click *Agree and access*. Then verify
  `hf auth whoami` shows your account.
- **OOM during inference (not during load)** — context saturated. Drop
  `frames` for that model in `config.yaml`. Don't retry in a loop;
  fragmentation makes the second attempt OOM harder.
- **`trust_remote_code` ValueError** — set `provider_config.trust_remote_code: true`
  in the relevant model's entry. Required for Hulu-Med; not needed for
  MedGemma, Qwen3-VL, Qwen3.5, ColonR1.
- **Hulu-Med predictions look truncated** — Hulu-Med has a hard 16K
  context. At >16 frames at 896², it silently truncates. Always run
  Hulu-Med at `frames: 16`.
- **Qwen3.5 emits `<think>...</think>`** — the scorer strips it before
  parsing. If you want clean output for analysis, set
  `provider_config.enable_thinking: false` in the model's config entry.
- **ColonR1 first-time load fails with `does not appear to have files
  named ...`** — the published HF snapshot has weight files with extra
  `-003` / `-001` version suffixes. The provider self-heals this on
  first use by creating symlinks to the canonical names; if you see
  this error, it usually means the symlink call failed. Run
  `python -c "from huggingface_hub import snapshot_download; print(snapshot_download('ai4colonoscopy/ColonR1'))"`
  and check write permissions in the printed cache dir.
- **`OPENAI_API_KEY not found`** — either export the env var or set
  `credentials.openai_api_key_file: /path/to/key.txt` in `config.yaml`.
- **Gemini auth fails / no project_id** — confirm the service-account
  JSON is readable and contains a `project_id` field. The provider calls
  `google.auth.default()` which extracts it; if your JSON is atypical,
  set `GOOGLE_CLOUD_PROJECT` explicitly or add `vertex_project: <id>`
  under `provider_config:` of the `gemini-flash` model entry in
  `config.yaml`.

## Citation

```bibtex
@misc{endoscopv_2026,
  title  = {EndoSCOP-V: Multi-turn Video Q&A Benchmark for Endoscopy and Colonoscopy},
  author = {Anonymous},
  year   = {2026},
  note   = {Harvard Dataverse},
}
```

## License

See `LICENSE`. Distributed for research use.
