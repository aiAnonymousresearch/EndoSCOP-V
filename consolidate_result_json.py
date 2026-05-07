"""Consolidate per-case prediction JSONs across all models into a flat CSV.

Walks <dataset_dir>/<model>/<case>.json and writes one row per (case, question),
with one column per model containing that model's extracted answer.

When `model_answer` is null in the JSON (the existing scorer's regex fallback
already failed), a more lenient regex pass over `model_raw` is attempted.
Anything still unrecoverable becomes an empty cell.

Usage
-----
    python consolidate_result_json.py evaluations/evaluated_cases/

    python consolidate_result_json.py /path/to/evaluated_cases \\
        --output evaluations/consolidated_results.csv \\
        --include-models ColonR1,MedGemma-4b-it \\
        --include-correctness
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = _SCRIPT_DIR / "config.yaml"


def _load_enabled_results_names() -> set[str] | None:
    """Collect results_name values for models with enabled: true in config.yaml.

    Returns None if config is missing (no filtering), or a set of names.
    """
    if not _CONFIG_PATH.exists():
        return None
    try:
        with open(_CONFIG_PATH) as f:
            cfg = yaml.safe_load(f) or {}
        enabled = set()
        for key, entry in (cfg.get("models") or {}).items():
            if entry.get("enabled", True):
                name = entry.get("results_name", key)
                enabled.add(name)
        return enabled
    except Exception:
        return None


ENABLED_RESULTS_NAMES = _load_enabled_results_names()


def normalize_type(qtype: str, multi_select: bool) -> str:
    if (qtype or "").lower() == "binary":
        return "binary"
    if (qtype or "").upper() == "MCQ":
        return "mcq_multi" if multi_select else "mcq_single"
    return (qtype or "").lower()


_LETTER = re.compile(r"\b([A-E])\b")
_BOXED = re.compile(r"\\boxed\{\s*([A-E](?:\s*,\s*[A-E])*)\s*\}")
_BOLD = re.compile(r"\*\*\s*([A-E])\s*\*\*")
_JSON_ANSWER_STR = re.compile(r'"answer"\s*:\s*"([^"]+)"')
_JSON_ANSWER_LIST = re.compile(r'"answer"\s*:\s*\[([^\]]+)\]')
_FINAL_ANSWER = re.compile(r"(?:final answer|answer is|the answer)\s*[:\-]?\s*([A-E])", re.I)
_YES_NO = re.compile(r"\b(Yes|No)\b", re.I)


def fallback_extract(raw: str | None, qtype: str, multi_select: bool) -> str:
    """Last-ditch regex extraction from prose/raw output. Returns "Not_Extracted" on failure."""
    if not raw:
        return "Not_Extracted"
    text = str(raw)

    m = _JSON_ANSWER_STR.search(text)
    if m:
        return m.group(1).strip()
    m = _JSON_ANSWER_LIST.search(text)
    if m:
        items = re.findall(r'"([^"]+)"', m.group(1))
        if items:
            return "|".join(items)

    if (qtype or "").lower() == "binary":
        m = _YES_NO.search(text)
        if m:
            return m.group(1).capitalize()
        return "Not_Extracted"

    m = _BOXED.search(text)
    if m:
        letters = [x.strip() for x in m.group(1).split(",") if x.strip()]
        return "|".join(letters) if multi_select else letters[0]
    m = _BOLD.search(text)
    if m:
        return m.group(1)
    m = _FINAL_ANSWER.search(text)
    if m:
        return m.group(1).upper()

    last_letter = None
    for m in _LETTER.finditer(text):
        last_letter = m.group(1)
    return last_letter or "Not_Extracted"


def cell_value(model_answer, model_raw, qtype: str, multi_select: bool) -> str:
    if model_answer is None:
        return fallback_extract(model_raw, qtype, multi_select)
    if isinstance(model_answer, list):
        return "|".join(str(x) for x in model_answer)
    return str(model_answer)


def truth_value(correct) -> str:
    if isinstance(correct, list):
        return "|".join(str(x) for x in correct)
    return "" if correct is None else str(correct)


def discover_models(dataset_dir: Path, include: set[str] | None) -> list[str]:
    """Find model dirs. If include is not None, keep only those in the set."""
    models = []
    for p in sorted(dataset_dir.iterdir()):
        if not p.is_dir():
            continue
        if not any(p.glob("*.json")):
            continue
        if include is not None and p.name not in include:
            continue
        models.append(p.name)
    return models


def consolidate(
    dataset_dir: Path,
    out_csv: Path,
    include: set[str] | None,
    include_correctness: bool,
) -> tuple[int, int, list[str]]:
    """Returns (rows_written, fallback_count, model_list)."""
    models = discover_models(dataset_dir, include)
    if not models:
        return 0, 0, []

    rows: dict[tuple[str, int], dict] = {}
    fallback_count = 0

    for model in models:
        for jpath in sorted((dataset_dir / model).glob("*.json")):
            try:
                doc = json.loads(jpath.read_text())
            except Exception as e:
                print(f"WARN: cannot parse {jpath}: {e}", file=sys.stderr)
                continue

            case_id = doc.get("case_id") or jpath.stem
            scopy = doc.get("procedure_type", "") or ""

            for r in doc.get("results", []):
                q_num = r.get("question_number")
                if q_num is None:
                    continue
                key = (case_id, q_num)
                row = rows.setdefault(key, {
                    "video": case_id,
                    "Q_Num": q_num,
                    "scopy": scopy,
                    "disease": r.get("disease") or "",
                    "Type": normalize_type(r.get("type", ""), bool(r.get("multi_select"))),
                    "Task": r.get("task") or "",
                    "SubTask": r.get("subtask") or "",
                    "Ground_Truth": truth_value(r.get("correct_answer")),
                })
                ma = r.get("model_answer")
                cell = cell_value(ma, r.get("model_raw"), r.get("type", ""), bool(r.get("multi_select")))
                if ma is None and cell and cell != "Not_Extracted":
                    fallback_count += 1
                row[model] = cell
                if include_correctness:
                    row[f"{model}__correct"] = "1" if r.get("is_correct") else "0"

    base_cols = ["video", "Q_Num", "scopy", "disease", "Type", "Task", "SubTask", "Ground_Truth"]
    model_cols = []
    for m in models:
        model_cols.append(m)
        if include_correctness:
            model_cols.append(f"{m}__correct")
    cols = base_cols + model_cols

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sorted_keys = sorted(rows.keys(), key=lambda k: (k[0], k[1]))
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for k in sorted_keys:
            w.writerow({c: rows[k].get(c, "") for c in cols})

    return len(rows), fallback_count, models


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir", type=Path,
                    help="dataset folder containing model-name subfolders with per-case JSONs")
    ap.add_argument("--output", type=Path, default=None,
                    help="output CSV path (default: <dataset_dir>/<dataset_name>_consolidated.csv)")
    ap.add_argument("--include-models", default=None,
                    help="comma-separated model results_names to include (default: enabled models from config.yaml)")
    ap.add_argument("--include-correctness", action="store_true",
                    help="add a <model>__correct (0/1) column next to each model column")
    args = ap.parse_args()

    if not args.dataset_dir.is_dir():
        sys.exit(f"dataset_dir not found: {args.dataset_dir}")

    out_csv = args.output or (args.dataset_dir / f"{args.dataset_dir.name}_consolidated.csv")

    if args.include_models:
        include = {s.strip() for s in args.include_models.split(",") if s.strip()}
    else:
        include = ENABLED_RESULTS_NAMES  # from config.yaml (None = no filtering)

    print(f"input:  {args.dataset_dir}")
    print(f"output: {out_csv}")
    if include is not None:
        print(f"including models: {sorted(include)}")
    else:
        print("including: all models (no config.yaml found)")
    print()

    n, fb, models = consolidate(args.dataset_dir, out_csv, include, args.include_correctness)
    if n == 0:
        sys.exit("no eligible model dirs found")

    print(f"models ({len(models)}): {models}")
    print(f"{n} rows × {len(models)} model columns, {fb} regex-fallback cells → {out_csv}")


if __name__ == "__main__":
    main()
