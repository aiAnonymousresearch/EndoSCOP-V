"""Summarize consolidated evaluation results into a 6-sheet Excel workbook.

Reads the flat CSV produced by consolidate_result_json.py and computes
accuracies across procedure type, task, subtask, and disease dimensions.

Sheets
------
1. EndoColon     — models as rows, Endoscopy/Colonoscopy/Both as column groups
2. Task          — models as columns, tasks as row groups (6 metrics each)
3. SubTask       — models as columns, subtasks as row groups
4. Diseases      — models as columns, diseases as row groups
5. Task_Total    — models as columns, tasks+subtasks as rows (Total Partial only)
6. Disease_Total — models as columns, diseases as rows (Total Partial only)

With --ci, four extra CI sheets follow (95% half-widths in percentage points):
7.  EndoColon_CI     — same shape as EndoColon (6 metrics × 3 procedure types)
8.  Task_Total_CI    — same shape as Task_Total (Total Partial only)
9.  SubTask_CI       — same shape as SubTask (6 metrics per sub-task)
10. Disease_Total_CI — same shape as Disease_Total (Total Partial only)

Partial marking for mcq_multi:
    n correct answers  →  +1/n per correct selection, −1/(2n) per incorrect.
    Floor at 0.

Usage
-----
    python summarize.py evaluations/consolidated_results.csv
    python summarize.py path/to/consolidated.csv --output /tmp/out.xlsx
    python summarize.py path/to/consolidated.csv --include-models ColonR1,MedGemma-4b-it
    python summarize.py path/to/consolidated.csv --ci
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path

import xlsxwriter
import yaml

# ── Constants ────────────────────────────────────────────────────────────

SPECIAL_DISEASES = ["Normal", "Ampulla of Vater / Papilla", "Hills hiatus", "Mucosa"]

DISEASE_NORMALIZE = {
    "Suspected Barrett's esophagus": "Barrett's esophagus",
    "Suspected Esophageal Motility Disorder/Achalasia": "Esophageal Motility Disorder/Achalasia",
    "Nodule (Mucosal)": "Mucosal Nodule",
    "Nodule": "Mucosal Nodule",
    "Polyp": "Ileo-Colonic Polyp",
    "Ulcer": "Colonic Ulcer",
}

TASK_ORDER = ["DI", "AL", "MA", "PF", "QN", "GS", "CF", "CR", "AR"]
SUBTASK_MAP = {
    "DI": ["DI-M", "DI-C", "DI-R", "DI-N"],
    "AL": ["AL-O", "AL-P", "AL-W"],
    "MA": ["MA-T", "MA-S", "MA-C", "MA-M", "MA-SZ"],
    "PF": ["PF-D", "PF-M", "PF-U", "PF-S", "PF-B"],
    "QN": ["QN-L", "QN-S"],
    "GS": ["GS-MC", "GS-SS", "GS-EC", "GS-AG"],
    "CF": ["CF-B", "CF-S", "CF-A"],
    "CR": ["CR-E", "CR-I"],
    "AR": ["AR-B", "AR-V"],
}

SCOPY_LABELS = {"endoscopy": "Endoscopy", "colonoscopy": "Colonoscopy", "both": "Both"}
SUB_HEADERS = ["Binary", "Single", "Multi", "Partial", "Strict", "Partial"]
METRIC_LABELS = ["Binary", "MCQ Single", "MCQ Multi", "MCQ Partial",
                 "Total Strict", "Total Partial"]
COLS_PER_GROUP = 6

_SCRIPT_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = _SCRIPT_DIR / "config.yaml"


# ── Format cache ─────────────────────────────────────────────────────────
# Base format properties.  Fmt.get(name, **extra) creates variants on the fly
# (e.g. adding borders) and caches them.

_FMT_DEFS: dict[str, dict] = {
    # Shared
    "pct":          {"num_format": "0.0%"},
    "pct_bold":     {"num_format": "0.0%", "bold": True},
    # Matrix (EndoColon) headers
    "header":       {"bold": True, "align": "center", "bottom": 1},
    "group_hdr":    {"bold": True, "align": "center", "bg_color": "#D9E1F2"},
    "sub_hdr":      {"bold": True, "align": "center", "bg_color": "#E8ECF4"},
    "count":        {"italic": True, "font_color": "#888888",
                     "align": "center", "num_format": "0"},
    "count_label":  {"italic": True, "font_color": "#888888", "align": "right"},
    # Group-name rows (grouped-rows sheets + Task_Total task rows)
    "grp_name":     {"bold": True, "bg_color": "#FFFFF0",
                     "top": 1, "bottom": 1, "align": "left"},
    "grp_bg":       {"bg_color": "#FFFFF0", "top": 1, "bottom": 1},
    "grp_pct":      {"bg_color": "#FFFFF0", "top": 1, "bottom": 1,
                     "num_format": "0.0%"},
    "grp_pct_bold": {"bg_color": "#FFFFF0", "top": 1, "bottom": 1,
                     "num_format": "0.0%", "bold": True},
    # Metric-label in grouped-rows sheets
    "met_label":    {"indent": 1},
    # Subtask rows in Task_Total
    "st_name":      {"italic": True, "align": "right"},
    "st_pct":       {"italic": True, "num_format": "0.0%"},
    "st_pct_bold":  {"italic": True, "num_format": "0.0%", "bold": True},
    # Disease_Total special disease name
    "d_special":    {"align": "right"},
    # Plain (no styling, used as base for border-only cells)
    "plain":        {},
}


class Fmt:
    """Lazily creates and caches xlsxwriter Format objects."""

    def __init__(self, wb: xlsxwriter.Workbook):
        self._wb = wb
        self._cache: dict[tuple, xlsxwriter.workbook.Format] = {}

    def __getitem__(self, name: str):
        return self.get(name)

    def get(self, name: str, **extra):
        key = (name, tuple(sorted(extra.items())))
        if key not in self._cache:
            props = dict(_FMT_DEFS.get(name, {}))
            props.update(extra)
            self._cache[key] = self._wb.add_format(props)
        return self._cache[key]


# ── Scoring ──────────────────────────────────────────────────────────────

def _split_set(s: str) -> set[str]:
    return set(x.strip() for x in s.split("|") if x.strip())


def score_strict(gt: str, answer: str, qtype: str) -> int:
    if not answer:
        return 0
    if qtype in ("binary", "mcq_single"):
        return 1 if answer.strip() == gt.strip() else 0
    if qtype == "mcq_multi":
        return 1 if _split_set(gt) == _split_set(answer) else 0
    return 0


def score_partial(gt: str, answer: str, qtype: str) -> float:
    if not answer:
        return 0.0
    if qtype in ("binary", "mcq_single"):
        return 1.0 if answer.strip() == gt.strip() else 0.0
    if qtype == "mcq_multi":
        gt_s, ma_s = _split_set(gt), _split_set(answer)
        n = len(gt_s)
        if n == 0:
            return 0.0
        return max(0.0, len(gt_s & ma_s) / n - len(ma_s - gt_s) / (2 * n))
    return 0.0


# ── Confidence intervals ─────────────────────────────────────────────────

# z = 1.959963984540054 for alpha = 0.05 (two-sided 95%).
_Z_95 = 1.959963984540054


def wilson_half_width(numer: float, denom: int) -> float | None:
    """Return half-width of the 95% Wilson score interval for a Bernoulli proportion.

    Use for binary, mcq_single, mcq_multi (strict), and total (strict) cells —
    every observation is 0 or 1.

    Returns None if denom == 0 so the caller can render an empty cell.
    """
    if denom <= 0:
        return None
    p = max(0.0, min(1.0, numer / denom))
    z2 = _Z_95 * _Z_95
    denom_factor = 1 + z2 / denom
    margin = (_Z_95 * math.sqrt(p * (1 - p) / denom + z2 / (4 * denom * denom))) / denom_factor
    return margin


def normal_half_width(score_sum: float, score_sq_sum: float, denom: int) -> float | None:
    """Half-width of a 95% normal-approximation CI for the mean of fractional scores.

    Use for mcq_multi (partial) and total (partial) cells where each observation
    is a real number in [0, 1] rather than 0/1. We compute Var = E[X²] - E[X]²
    with a Bessel-corrected estimator and SE = sqrt(Var / n).

    Returns None if denom < 2 (Bessel correction needs n ≥ 2).
    """
    if denom < 2:
        return None
    mean = score_sum / denom
    var_biased = max(0.0, score_sq_sum / denom - mean * mean)
    var = var_biased * denom / (denom - 1)
    se = math.sqrt(var / denom)
    return _Z_95 * se


# ── Data loading ─────────────────────────────────────────────────────────

def load_csv(path: Path) -> tuple[list[dict], list[str]]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit("No data rows in CSV")
    # Normalize disease names
    for row in rows:
        d = row.get("disease", "")
        if d in DISEASE_NORMALIZE:
            row["disease"] = DISEASE_NORMALIZE[d]
    all_cols = list(rows[0].keys())
    gt_idx = all_cols.index("Ground_Truth")
    models = all_cols[gt_idx + 1:]
    return rows, models


def load_taxonomy(cfg_path: Path) -> dict[str, str]:
    """Return {abbr: display_name} from task_taxonomy.csv referenced in config."""
    tax_path = None
    if cfg_path.exists():
        try:
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f) or {}
            rel = (cfg.get("summary") or {}).get("task_taxonomy")
            if rel:
                p = Path(rel)
                if not p.is_absolute():
                    p = cfg_path.parent / p
                tax_path = p
        except Exception:
            pass
    if not tax_path or not tax_path.exists():
        return {}
    mapping: dict[str, str] = {}
    with open(tax_path, newline="") as f:
        for row in csv.DictReader(f):
            abbr = row.get("Abbr", "").strip()
            task = row.get("Task", "").strip()
            sub = row.get("Sub-task", "").strip()
            if not abbr:
                continue
            mapping[abbr] = f"{sub} ({abbr})"
            m = re.search(r"\(([^)]+)\)$", task)
            if m:
                mapping.setdefault(m.group(1), task)
    return mapping


# ── Metrics computation ──────────────────────────────────────────────────

def _empty_entry():
    # Each value is [sum, n]; the extra "_sq" entries hold sum-of-squares,
    # used only for the normal-approximation CI on partial scores.
    return {
        "binary": [0, 0],
        "mcq_single": [0, 0],
        "mcq_multi_strict": [0, 0],
        "mcq_multi_partial": [0.0, 0],
        "mcq_multi_partial_sq": 0.0,
    }


def _empty_counts():
    return {"binary": 0, "mcq_single": 0, "mcq_multi": 0, "total": 0}


def compute_metrics(rows, models, group_fn, skip_empty_group=False):
    """Compute per-model per-group scoring metrics.

    group_fn: column name (str) or callable(row) -> group value.
    Returns (metrics, counts):
        metrics = {model: {group: entry}}
        counts  = {group: {binary: N, mcq_single: N, mcq_multi: N, total: N}}
    """
    get_group = (lambda r, c=group_fn: r.get(c, "")) if isinstance(group_fn, str) else group_fn
    metrics = {m: {} for m in models}
    counts: dict[str, dict] = {}

    for row in rows:
        gval = get_group(row)
        if skip_empty_group and not gval:
            continue
        qtype = row["Type"]
        gt = row["Ground_Truth"]

        if gval not in counts:
            counts[gval] = _empty_counts()
        counts[gval]["total"] += 1
        if qtype in ("binary", "mcq_single", "mcq_multi"):
            counts[gval][qtype] += 1

        for model in models:
            if gval not in metrics[model]:
                metrics[model][gval] = _empty_entry()
            e = metrics[model][gval]
            ans = row.get(model, "")
            if not ans:
                # Empty cell = model was not run on this case (e.g., procedure
                # filter mismatch). Skip — do not penalize the model for a
                # question it never attempted. "Not_Extracted" (extraction
                # failure on an attempt) is non-empty and DOES score below.
                continue
            if qtype == "binary":
                e["binary"][1] += 1
                e["binary"][0] += score_strict(gt, ans, qtype)
            elif qtype == "mcq_single":
                e["mcq_single"][1] += 1
                e["mcq_single"][0] += score_strict(gt, ans, qtype)
            elif qtype == "mcq_multi":
                e["mcq_multi_strict"][1] += 1
                e["mcq_multi_strict"][0] += score_strict(gt, ans, qtype)
                ps = score_partial(gt, ans, qtype)
                e["mcq_multi_partial"][1] += 1
                e["mcq_multi_partial"][0] += ps
                e["mcq_multi_partial_sq"] += ps * ps

    return metrics, counts


def derive_6(entry) -> list[float | None]:
    """Six accuracy values: [binary, single, multi, partial, total_strict, total_partial]."""
    bc, bt = entry["binary"]
    sc, st = entry["mcq_single"]
    mc, mt = entry["mcq_multi_strict"]
    ps, pt = entry["mcq_multi_partial"]
    total = bt + st + mt
    return [
        bc / bt if bt else None,
        sc / st if st else None,
        mc / mt if mt else None,
        ps / pt if pt else None,
        (bc + sc + mc) / total if total else None,
        (bc + sc + ps) / total if total else None,
    ]


def derive_6_ci(entry) -> list[float | None]:
    """95% half-widths matching `derive_6`'s six positions.

    Bernoulli cells (binary, single, multi-strict, total-strict) use Wilson.
    Cells that mix in fractional partial scores (multi-partial, total-partial)
    use a normal-approximation CI computed from the per-question sum-of-squares
    we tracked alongside the running sum. Returns None where n is too small
    (Bernoulli: n < 1; normal-approx: n < 2).
    """
    bc, bt = entry["binary"]
    sc, st = entry["mcq_single"]
    mc, mt = entry["mcq_multi_strict"]
    ps, pt = entry["mcq_multi_partial"]
    pss = entry["mcq_multi_partial_sq"]
    total = bt + st + mt
    # Total-partial mixes 0/1 scores (binary + single) with fractional (multi
    # partial). When mt == 0 every observation is Bernoulli, so Wilson is
    # appropriate and tighter; otherwise we fall back to normal-approx with
    # the tracked sum-of-squares.
    total_partial = bc + sc + ps
    if mt == 0:
        total_partial_hw = wilson_half_width(total_partial, total)
    else:
        total_partial_sq = bc + sc + pss   # 0²=0 and 1²=1 for Bernoulli
        total_partial_hw = normal_half_width(total_partial, total_partial_sq, total)
    return [
        wilson_half_width(bc, bt),
        wilson_half_width(sc, st),
        wilson_half_width(mc, mt),
        normal_half_width(ps, pss, pt),
        wilson_half_width(bc + sc + mc, total),
        total_partial_hw,
    ]


# ── Ordering helpers ─────────────────────────────────────────────────────

def order_diseases(disease_set):
    regular = sorted(d for d in disease_set if d not in SPECIAL_DISEASES)
    special = [d for d in SPECIAL_DISEASES if d in disease_set]
    return regular + special


def order_tasks(task_set):
    ordered = [t for t in TASK_ORDER if t in task_set]
    return ordered + sorted(t for t in task_set if t not in ordered)


def order_subtasks(subtask_set):
    ordered = []
    for task in TASK_ORDER:
        for st in SUBTASK_MAP.get(task, []):
            if st in subtask_set:
                ordered.append(st)
    return ordered + sorted(st for st in subtask_set if st not in ordered)


def subtask_parent(st: str) -> str:
    """DI-M -> DI, CF-B -> CF."""
    return st.split("-")[0] if "-" in st else st


# ── Helpers ──────────────────────────────────────────────────────────────

def _cell_borders(col, row, grp_ranges, mcq_ranges, last_row):
    """Return border kwargs for a cell in the EndoColon matrix sheet."""
    b = {}
    for gs, ge in grp_ranges:
        if col == gs:
            b["left"] = 1
        if col == ge:
            b["right"] = 1
    for ms, me in mcq_ranges:
        if col == ms:
            b["left"] = 1
        if col == me:
            b["right"] = 1
    if row == 0:
        b["top"] = 1
    if row == last_row:
        b["bottom"] = 1
    return b


def _bold_best_in_row(ws, row, col_start, vals, fmt_n, fmt_b, fmt_blank=None):
    """Write vals starting at col_start; bold the maximum value(s)."""
    best = max((v for v in vals if v is not None), default=None)
    for ci, v in enumerate(vals):
        if v is None:
            if fmt_blank:
                ws.write_blank(row, col_start + ci, None, fmt_blank)
            continue
        f = fmt_b if (best is not None and v == best) else fmt_n
        ws.write_number(row, col_start + ci, v, f)


# ── Sheet 1: EndoColon (matrix with group borders) ──────────────────────

def write_matrix_sheet(wb, sheet_name, models, groups, group_labels,
                       metrics, counts, F):
    ws = wb.add_worksheet(sheet_name)
    ws.hide_gridlines(2)
    ng = len(groups)
    nm = len(models)
    last_row = 3 + nm  # rows 0-2 headers, 3 count, 4..3+nm data

    # Group column ranges and MCQ sub-ranges
    grp_ranges = []
    mcq_ranges = []
    col = 1
    for _ in groups:
        grp_ranges.append((col, col + COLS_PER_GROUP - 1))
        mcq_ranges.append((col + 1, col + 3))
        col += COLS_PER_GROUP

    def bdr(c, r):
        return _cell_borders(c, r, grp_ranges, mcq_ranges, last_row)

    # Row 0 — group headers (merged, full border box)
    for gi, g in enumerate(groups):
        gs, ge = grp_ranges[gi]
        ws.merge_range(0, gs, 0, ge, group_labels.get(g, g),
                       F.get("group_hdr", top=1, bottom=1, left=1, right=1))

    # Row 1 — MCQ / Total sub-headers
    for gs, ge in grp_ranges:
        # Binary column: same bg as neighbours + left group border
        ws.write_blank(1, gs, None, F.get("sub_hdr", left=1))
        # MCQ merge with left+right sub-group borders
        ws.merge_range(1, gs + 1, 1, gs + 3, "MCQ",
                       F.get("sub_hdr", left=1, right=1))
        # Total merge with right group border
        ws.merge_range(1, gs + 4, 1, ge, "Total", F.get("sub_hdr", right=1))

    # Row 2 — column sub-headers
    ws.write(2, 0, "Model", F["header"])
    for gs, ge in grp_ranges:
        for i, sh in enumerate(SUB_HEADERS):
            c = gs + i
            ws.write(2, c, sh, F.get("header", **bdr(c, 2)))

    # Row 3 — count row
    ws.write(3, 0, "(n=)", F["count_label"])
    for gi, g in enumerate(groups):
        gs, ge = grp_ranges[gi]
        gc = counts.get(g, _empty_counts())
        cnt_vals = [gc["binary"], gc["mcq_single"], gc["mcq_multi"],
                    gc["mcq_multi"], gc["total"], gc["total"]]
        for i, v in enumerate(cnt_vals):
            c = gs + i
            ws.write(3, c, v, F.get("count", **bdr(c, 3)))

    # Rows 4+ — model data
    grid: list[list[float | None]] = []
    for mi, model in enumerate(models):
        r = 4 + mi
        if r == last_row:
            ws.write(r, 0, model, F.get("plain", bottom=1))
        else:
            ws.write(r, 0, model)
        row_vals: list[float | None] = []
        for gi, g in enumerate(groups):
            gs, ge = grp_ranges[gi]
            entry = metrics.get(model, {}).get(g)
            accs = derive_6(entry) if entry else [None] * 6
            for i, val in enumerate(accs):
                c = gs + i
                b = bdr(c, r)
                if val is not None:
                    ws.write_number(r, c, val, F.get("pct", **b))
                elif b:
                    ws.write_blank(r, c, None, F.get("plain", **b))
                row_vals.append(val)
        grid.append(row_vals)

    # Bold best per column
    total_dcols = ng * COLS_PER_GROUP
    for ci in range(total_dcols):
        best = max((grid[mi][ci] for mi in range(nm)
                    if grid[mi][ci] is not None), default=None)
        if best is None:
            continue
        for mi in range(nm):
            if grid[mi][ci] == best:
                r = 4 + mi
                abs_col = 1 + ci
                b = bdr(abs_col, r)
                ws.write_number(r, abs_col, best, F.get("pct_bold", **b))

    ws.set_column(0, 0, 22)
    ws.set_column(1, total_dcols, 9)


# ── Sheets 2-4: Grouped-rows (Task / SubTask / Diseases) ────────────────

def write_grouped_rows_sheet(wb, sheet_name, models, groups, labels,
                             metrics, counts, F):
    """Models as columns, each group = header row + 6 metric rows."""
    ws = wb.add_worksheet(sheet_name)
    ws.hide_gridlines(2)
    ws.freeze_panes(1, 1)
    nm = len(models)
    mc = 2  # model columns start at index 2 (col C); col B = counts

    # Row 0 — headers
    ws.write(0, 1, "(n=)", F["count_label"])
    for ci, model in enumerate(models):
        ws.write(0, mc + ci, model, F["header"])

    last_data_row = len(groups) * 7  # 1 header + 6 metrics per group

    row = 1
    for g in groups:
        gc = counts.get(g, _empty_counts())
        label = labels.get(g, g)

        # Group header row (yellow band)
        ws.write(row, 0, label, F["grp_name"])
        ws.write_blank(row, 1, None, F["grp_bg"])
        for ci in range(nm):
            ws.write_blank(row, mc + ci, None, F["grp_bg"])
        row += 1

        # 6 metric rows
        metric_counts = [gc["binary"], gc["mcq_single"], gc["mcq_multi"],
                         gc["mcq_multi"], gc["total"], gc["total"]]
        for mi_idx, (mname, cnt) in enumerate(zip(METRIC_LABELS, metric_counts)):
            bot = row == last_data_row
            ws.write(row, 0, mname,
                     F.get("met_label", bottom=1) if bot else F["met_label"])
            ws.write(row, 1, cnt,
                     F.get("count", bottom=1) if bot else F["count"])
            vals = []
            for ci, model in enumerate(models):
                entry = metrics.get(model, {}).get(g)
                val = derive_6(entry)[mi_idx] if entry else None
                vals.append(val)
            if bot:
                _bold_best_in_row(ws, row, mc, vals,
                                  F.get("pct", bottom=1),
                                  F.get("pct_bold", bottom=1),
                                  F.get("plain", bottom=1))
            else:
                _bold_best_in_row(ws, row, mc, vals, F["pct"], F["pct_bold"])
            row += 1

    ws.set_column(0, 0, 35)
    ws.set_column(1, 1, 6)
    ws.set_column(mc, mc + nm - 1, 18)


# ── Sheet 5: Task_Total ─────────────────────────────────────────────────

def write_task_total(wb, models, task_metrics, subtask_metrics,
                     task_counts, subtask_counts, taxonomy, F):
    ws = wb.add_worksheet("Task_Total")
    ws.hide_gridlines(2)
    ws.freeze_panes(1, 1)
    nm = len(models)
    mc = 2  # model columns start at index 2; col 1 = counts

    ws.write(0, 1, "(n=)", F["count_label"])
    for ci, model in enumerate(models):
        ws.write(0, mc + ci, model, F["header"])

    tasks_in_data = set()
    for m in models:
        tasks_in_data.update(task_metrics.get(m, {}).keys())
    subtasks_in_data = set()
    for m in models:
        subtasks_in_data.update(subtask_metrics.get(m, {}).keys())

    # Pre-compute total row count for bottom border
    total_rows = 0
    for task in order_tasks(tasks_in_data):
        total_rows += 1
        for st in SUBTASK_MAP.get(task, []):
            if st in subtasks_in_data:
                total_rows += 1
    last_data_row = total_rows  # 1-based from row 1

    row = 1
    for task in order_tasks(tasks_in_data):
        label = taxonomy.get(task, task)
        tc = task_counts.get(task, _empty_counts())
        bot = row == last_data_row
        ws.write(row, 0, label,
                 F.get("grp_name", bottom=2) if bot else F["grp_name"])
        ws.write(row, 1, tc["total"],
                 F.get("grp_bg", **{"num_format": "0", "bold": True}) if not bot
                 else F.get("grp_bg", **{"num_format": "0", "bold": True, "bottom": 2}))
        vals = []
        for ci, model in enumerate(models):
            entry = task_metrics.get(model, {}).get(task)
            val = derive_6(entry)[5] if entry else None
            vals.append(val)
        if bot:
            _bold_best_in_row(ws, row, mc, vals,
                              F.get("grp_pct", bottom=2),
                              F.get("grp_pct_bold", bottom=2),
                              F.get("grp_bg", bottom=2))
        else:
            _bold_best_in_row(ws, row, mc, vals, F["grp_pct"], F["grp_pct_bold"])
            for ci in range(nm):
                if vals[ci] is None:
                    ws.write_blank(row, mc + ci, None, F["grp_pct"])
        row += 1

        for st in SUBTASK_MAP.get(task, []):
            if st not in subtasks_in_data:
                continue
            st_label = taxonomy.get(st, st)
            sc = subtask_counts.get(st, _empty_counts())
            bot = row == last_data_row
            ws.write(row, 0, st_label,
                     F.get("st_name", bottom=1) if bot else F["st_name"])
            ws.write(row, 1, sc["total"],
                     F.get("count", bottom=1) if bot else F["count"])
            vals = []
            for ci, model in enumerate(models):
                entry = subtask_metrics.get(model, {}).get(st)
                val = derive_6(entry)[5] if entry else None
                vals.append(val)
            if bot:
                _bold_best_in_row(ws, row, mc, vals,
                                  F.get("st_pct", bottom=1),
                                  F.get("st_pct_bold", bottom=1),
                                  F.get("plain", bottom=1))
            else:
                _bold_best_in_row(ws, row, mc, vals, F["st_pct"], F["st_pct_bold"])
            row += 1

    ws.set_column(0, 0, 38)
    ws.set_column(1, 1, 6)
    ws.set_column(mc, mc + nm - 1, 18)


# ── Sheet 6: Disease_Total ──────────────────────────────────────────────

def write_disease_total(wb, models, disease_metrics, disease_counts, F):
    ws = wb.add_worksheet("Disease_Total")
    ws.hide_gridlines(2)
    ws.freeze_panes(1, 1)
    nm = len(models)
    mc = 2  # model columns start at index 2; col 1 = counts

    ws.write(0, 1, "(n=)", F["count_label"])
    for ci, model in enumerate(models):
        ws.write(0, mc + ci, model, F["header"])

    diseases_in_data = set()
    for m in models:
        diseases_in_data.update(disease_metrics.get(m, {}).keys())
    diseases_in_data.discard("")
    ordered = order_diseases(diseases_in_data)
    n_diseases = len(ordered)

    row = 1
    for di, disease in enumerate(ordered):
        is_special = disease in SPECIAL_DISEASES
        is_last = (di == n_diseases - 1)
        need_bottom = ((di + 1) % 9 == 0) or is_last
        dc = disease_counts.get(disease, _empty_counts())

        # Name cell
        base_name = "d_special" if is_special else "plain"
        if need_bottom:
            ws.write(row, 0, disease, F.get(base_name, bottom=1))
        else:
            ws.write(row, 0, disease, F[base_name] if is_special else None)

        # Count cell
        if need_bottom:
            ws.write(row, 1, dc["total"], F.get("count", bottom=1))
        else:
            ws.write(row, 1, dc["total"], F["count"])

        # Value cells
        vals = []
        for ci, model in enumerate(models):
            entry = disease_metrics.get(model, {}).get(disease)
            val = derive_6(entry)[5] if entry else None
            vals.append(val)

        if need_bottom:
            _bold_best_in_row(ws, row, mc, vals,
                              F.get("pct", bottom=1),
                              F.get("pct_bold", bottom=1),
                              F.get("plain", bottom=1))
        else:
            _bold_best_in_row(ws, row, mc, vals, F["pct"], F["pct_bold"])
        row += 1

    ws.set_column(0, 0, 32)
    ws.set_column(1, 1, 6)
    ws.set_column(mc, mc + nm - 1, 18)


# ── CI sheets (rendered only with --ci) ─────────────────────────────────

def _ci_format(F: "Fmt"):
    """Lazily make a percentage-point format used only by the CI sheets."""
    return F.get("ci_pp", num_format="0.0", italic=True)


def write_subtask_ci(wb, models, sub_metrics, sub_counts, taxonomy, F):
    """Half-widths of 95% CIs for each (subtask, metric, model) cell.

    Mirrors the SubTask sheet: one yellow header row per subtask, then six
    metric rows. Each cell shows the half-width as percentage points
    (so 8.2 means ±8.2 pp around the corresponding accuracy in SubTask).
    """
    ws = wb.add_worksheet("SubTask_CI")
    ws.hide_gridlines(2)
    ws.freeze_panes(1, 1)
    nm = len(models)
    mc = 2

    ws.write(0, 0, "± half-width, percentage points (95% CI)",
             F.get("count_label", italic=True))
    ws.write(0, 1, "(n=)", F["count_label"])
    for ci, model in enumerate(models):
        ws.write(0, mc + ci, model, F["header"])

    sub_groups = order_subtasks(set(sub_counts.keys()))
    last_data_row = len(sub_groups) * 7
    fmt_pp = _ci_format(F)

    row = 1
    for st in sub_groups:
        sc = sub_counts.get(st, _empty_counts())
        label = taxonomy.get(st, st)

        ws.write(row, 0, label, F["grp_name"])
        ws.write_blank(row, 1, None, F["grp_bg"])
        for ci in range(nm):
            ws.write_blank(row, mc + ci, None, F["grp_bg"])
        row += 1

        metric_counts = [sc["binary"], sc["mcq_single"], sc["mcq_multi"],
                         sc["mcq_multi"], sc["total"], sc["total"]]
        for mi_idx, (mname, cnt) in enumerate(zip(METRIC_LABELS, metric_counts)):
            bot = row == last_data_row
            ws.write(row, 0, mname,
                     F.get("met_label", bottom=1) if bot else F["met_label"])
            ws.write(row, 1, cnt,
                     F.get("count", bottom=1) if bot else F["count"])
            for ci, model in enumerate(models):
                entry = sub_metrics.get(model, {}).get(st)
                hw = derive_6_ci(entry)[mi_idx] if entry else None
                cell_fmt = F.get("ci_pp", num_format="0.0", italic=True,
                                 bottom=1) if bot else fmt_pp
                if hw is None:
                    ws.write_blank(row, mc + ci, None,
                                   F.get("plain", bottom=1) if bot else None)
                else:
                    ws.write_number(row, mc + ci, hw * 100, cell_fmt)
            row += 1

    ws.set_column(0, 0, 35)
    ws.set_column(1, 1, 6)
    ws.set_column(mc, mc + nm - 1, 18)


def write_task_total_ci(wb, models, task_metrics, subtask_metrics,
                        task_counts, subtask_counts, taxonomy, F):
    """Half-widths matching the Task_Total sheet's single 'Total Partial' column.

    Rows = tasks (yellow header row, like Task_Total) followed by their sub-
    tasks (italic). Each cell is the 95% half-width on Total Partial in
    percentage points. Use this side-by-side with Task_Total when reporting
    paper numbers.
    """
    ws = wb.add_worksheet("Task_Total_CI")
    ws.hide_gridlines(2)
    ws.freeze_panes(1, 1)
    nm = len(models)
    mc = 2

    ws.write(0, 0, "± half-width on Total Partial, pp (95% CI)",
             F.get("count_label", italic=True))
    ws.write(0, 1, "(n=)", F["count_label"])
    for ci, model in enumerate(models):
        ws.write(0, mc + ci, model, F["header"])

    tasks_in_data = set()
    for m in models:
        tasks_in_data.update(task_metrics.get(m, {}).keys())
    subtasks_in_data = set()
    for m in models:
        subtasks_in_data.update(subtask_metrics.get(m, {}).keys())

    total_rows = 0
    for task in order_tasks(tasks_in_data):
        total_rows += 1
        for st in SUBTASK_MAP.get(task, []):
            if st in subtasks_in_data:
                total_rows += 1
    last_data_row = total_rows

    fmt_grp_pp = F.get("grp_pct", num_format="0.0", bold=True)
    fmt_st_pp = F.get("ci_pp", num_format="0.0", italic=True)

    row = 1
    for task in order_tasks(tasks_in_data):
        label = taxonomy.get(task, task)
        tc = task_counts.get(task, _empty_counts())
        bot = row == last_data_row
        ws.write(row, 0, label,
                 F.get("grp_name", bottom=2) if bot else F["grp_name"])
        ws.write(row, 1, tc["total"],
                 F.get("grp_bg", **{"num_format": "0", "bold": True,
                                    "bottom": 2}) if bot
                 else F.get("grp_bg", **{"num_format": "0", "bold": True}))
        for ci, model in enumerate(models):
            entry = task_metrics.get(model, {}).get(task)
            hw = derive_6_ci(entry)[5] if entry else None  # index 5 = total_partial
            cell_fmt = F.get("grp_pct", num_format="0.0", bold=True,
                             bottom=2) if bot else fmt_grp_pp
            if hw is None:
                ws.write_blank(row, mc + ci, None,
                               F.get("grp_bg", bottom=2) if bot else F["grp_bg"])
            else:
                ws.write_number(row, mc + ci, hw * 100, cell_fmt)
        row += 1

        for st in SUBTASK_MAP.get(task, []):
            if st not in subtasks_in_data:
                continue
            st_label = taxonomy.get(st, st)
            sc = subtask_counts.get(st, _empty_counts())
            bot = row == last_data_row
            ws.write(row, 0, st_label,
                     F.get("st_name", bottom=1) if bot else F["st_name"])
            ws.write(row, 1, sc["total"],
                     F.get("count", bottom=1) if bot else F["count"])
            for ci, model in enumerate(models):
                entry = subtask_metrics.get(model, {}).get(st)
                hw = derive_6_ci(entry)[5] if entry else None
                cell_fmt = F.get("ci_pp", num_format="0.0", italic=True,
                                 bottom=1) if bot else fmt_st_pp
                if hw is None:
                    ws.write_blank(row, mc + ci, None,
                                   F.get("plain", bottom=1) if bot else None)
                else:
                    ws.write_number(row, mc + ci, hw * 100, cell_fmt)
            row += 1

    ws.set_column(0, 0, 38)
    ws.set_column(1, 1, 6)
    ws.set_column(mc, mc + nm - 1, 18)


def write_endocolon_ci(wb, models, groups, group_labels, metrics, counts, F):
    """Half-widths of 95% CIs for each (procedure_type, metric, model) cell.

    Mirrors the EndoColon matrix sheet: 6 metrics × 3 procedure types per
    row, with the same group / sub-group borders. Each cell shows the
    half-width as percentage points (so a cell of 2.4 next to an EndoColon
    accuracy of 63.0% means 63.0% ± 2.4 pp). Wilson is used on Bernoulli
    cells (Binary, Single, Multi-Strict, Total Strict, and Total Partial
    when no multi-select is mixed in); normal-approx on the partial-score
    cells.
    """
    ws = wb.add_worksheet("EndoColon_CI")
    ws.hide_gridlines(2)
    ng = len(groups)
    nm = len(models)
    last_row = 3 + nm  # rows 0-2 headers, 3 count, 4..3+nm data

    grp_ranges = []
    mcq_ranges = []
    col = 1
    for _ in groups:
        grp_ranges.append((col, col + COLS_PER_GROUP - 1))
        mcq_ranges.append((col + 1, col + 3))
        col += COLS_PER_GROUP

    def bdr(c, r):
        return _cell_borders(c, r, grp_ranges, mcq_ranges, last_row)

    # Row 0 — group headers (merged, full border box)
    for gi, g in enumerate(groups):
        gs, ge = grp_ranges[gi]
        ws.merge_range(0, gs, 0, ge, group_labels.get(g, g),
                       F.get("group_hdr", top=1, bottom=1, left=1, right=1))

    # Row 1 — MCQ / Total sub-headers
    for gs, ge in grp_ranges:
        ws.write_blank(1, gs, None, F.get("sub_hdr", left=1))
        ws.merge_range(1, gs + 1, 1, gs + 3, "MCQ",
                       F.get("sub_hdr", left=1, right=1))
        ws.merge_range(1, gs + 4, 1, ge, "Total", F.get("sub_hdr", right=1))

    # Row 2 — column sub-headers
    ws.write(2, 0, "Model (± pp, 95% CI)", F["header"])
    for gs, ge in grp_ranges:
        for i, sh in enumerate(SUB_HEADERS):
            c = gs + i
            ws.write(2, c, sh, F.get("header", **bdr(c, 2)))

    # Row 3 — count row
    ws.write(3, 0, "(n=)", F["count_label"])
    for gi, g in enumerate(groups):
        gs, ge = grp_ranges[gi]
        gc = counts.get(g, _empty_counts())
        cnt_vals = [gc["binary"], gc["mcq_single"], gc["mcq_multi"],
                    gc["mcq_multi"], gc["total"], gc["total"]]
        for i, v in enumerate(cnt_vals):
            c = gs + i
            ws.write(3, c, v, F.get("count", **bdr(c, 3)))

    # Rows 4+ — model data (half-widths in pp, no bold-best)
    for mi, model in enumerate(models):
        r = 4 + mi
        if r == last_row:
            ws.write(r, 0, model, F.get("plain", bottom=1))
        else:
            ws.write(r, 0, model)
        for gi, g in enumerate(groups):
            gs, _ge = grp_ranges[gi]
            entry = metrics.get(model, {}).get(g)
            hws = derive_6_ci(entry) if entry else [None] * 6
            for i, hw in enumerate(hws):
                c = gs + i
                b = bdr(c, r)
                if hw is not None:
                    ws.write_number(r, c, hw * 100,
                                    F.get("ci_pp", num_format="0.0",
                                          italic=True, **b))
                elif b:
                    ws.write_blank(r, c, None, F.get("plain", **b))

    ws.set_column(0, 0, 22)
    ws.set_column(1, ng * COLS_PER_GROUP, 9)


def write_disease_total_ci(wb, models, disease_metrics, disease_counts, F):
    """Half-widths matching Disease_Total's single 'Total Partial' column.

    Models as columns, diseases as rows. Each cell is the 95% half-width on
    Total Partial in percentage points. Per-disease eval subsets are typically
    <100 questions per model, so half-widths frequently run ±8–15 pp — read
    side-by-side with Disease_Total before quoting per-disease numbers.
    """
    ws = wb.add_worksheet("Disease_Total_CI")
    ws.hide_gridlines(2)
    ws.freeze_panes(1, 1)
    nm = len(models)
    mc = 2

    ws.write(0, 0, "± half-width on Total Partial, pp (95% CI)",
             F.get("count_label", italic=True))
    ws.write(0, 1, "(n=)", F["count_label"])
    for ci, model in enumerate(models):
        ws.write(0, mc + ci, model, F["header"])

    diseases_in_data = set()
    for m in models:
        diseases_in_data.update(disease_metrics.get(m, {}).keys())
    diseases_in_data.discard("")
    ordered = order_diseases(diseases_in_data)
    n_diseases = len(ordered)

    fmt_pp = F.get("ci_pp", num_format="0.0", italic=True)

    row = 1
    for di, disease in enumerate(ordered):
        is_special = disease in SPECIAL_DISEASES
        is_last = (di == n_diseases - 1)
        need_bottom = ((di + 1) % 9 == 0) or is_last
        dc = disease_counts.get(disease, _empty_counts())

        base_name = "d_special" if is_special else "plain"
        if need_bottom:
            ws.write(row, 0, disease, F.get(base_name, bottom=1))
        else:
            ws.write(row, 0, disease, F[base_name] if is_special else None)

        if need_bottom:
            ws.write(row, 1, dc["total"], F.get("count", bottom=1))
        else:
            ws.write(row, 1, dc["total"], F["count"])

        for ci, model in enumerate(models):
            entry = disease_metrics.get(model, {}).get(disease)
            hw = derive_6_ci(entry)[5] if entry else None
            cell_fmt = F.get("ci_pp", num_format="0.0", italic=True,
                             bottom=1) if need_bottom else fmt_pp
            if hw is None:
                ws.write_blank(row, mc + ci, None,
                               F.get("plain", bottom=1) if need_bottom else None)
            else:
                ws.write_number(row, mc + ci, hw * 100, cell_fmt)
        row += 1

    ws.set_column(0, 0, 32)
    ws.set_column(1, 1, 6)
    ws.set_column(mc, mc + nm - 1, 18)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_path", type=Path, help="consolidated results CSV")
    ap.add_argument("--output", type=Path, default=None,
                    help="output xlsx path (default: <csv_dir>/Evaluation_Summary.xlsx)")
    ap.add_argument("--include-models", default=None,
                    help="comma-separated model names to include (default: all)")
    ap.add_argument("--ci", action="store_true",
                    help="emit four extra CI sheets (EndoColon_CI, "
                         "Task_Total_CI, SubTask_CI, Disease_Total_CI) "
                         "with 95% half-widths in percentage points")
    args = ap.parse_args()

    if not args.csv_path.is_file():
        sys.exit(f"CSV not found: {args.csv_path}")
    out_path = args.output or (args.csv_path.parent / "Evaluation_Summary.xlsx")

    # Load data
    rows, all_models = load_csv(args.csv_path)
    if args.include_models:
        keep = {s.strip() for s in args.include_models.split(",") if s.strip()}
        models = [m for m in all_models if m in keep]
    else:
        models = all_models
    if not models:
        sys.exit("No models found")

    taxonomy = load_taxonomy(_CONFIG_PATH)

    print(f"input:    {args.csv_path}")
    print(f"output:   {out_path}")
    print(f"models:   {models}")
    print(f"rows:     {len(rows)}")
    print(f"taxonomy: {len(taxonomy)} entries loaded")
    print()

    # Compute metrics for each grouping dimension
    scopy_m, scopy_c = compute_metrics(rows, models, "scopy")
    both_m, both_c = compute_metrics(rows, models, lambda _r: "both")
    task_m, task_c = compute_metrics(rows, models, "Task")
    sub_m, sub_c = compute_metrics(rows, models, "SubTask")
    dis_m, dis_c = compute_metrics(rows, models, "disease", skip_empty_group=True)

    # Merge "both" into scopy metrics/counts
    for model in models:
        scopy_m[model]["both"] = both_m[model]["both"]
    scopy_c["both"] = both_c["both"]

    # Determine group orderings
    scopy_groups = ["endoscopy", "colonoscopy", "both"]
    task_groups = order_tasks(set(task_c.keys()))
    sub_groups = order_subtasks(set(sub_c.keys()))
    dis_groups = order_diseases(set(dis_c.keys()))

    # Build label dicts
    task_labels = {t: taxonomy.get(t, t) for t in task_groups}
    sub_labels = {s: taxonomy.get(s, s) for s in sub_groups}
    dis_labels = {d: d for d in dis_groups}

    # Write Excel
    wb = xlsxwriter.Workbook(str(out_path))
    F = Fmt(wb)

    print("Writing EndoColon...", end=" ", flush=True)
    write_matrix_sheet(wb, "EndoColon", models, scopy_groups, SCOPY_LABELS,
                       scopy_m, scopy_c, F)
    print("done")

    print("Writing Task...", end=" ", flush=True)
    write_grouped_rows_sheet(wb, "Task", models, task_groups, task_labels,
                             task_m, task_c, F)
    print("done")

    print("Writing SubTask...", end=" ", flush=True)
    write_grouped_rows_sheet(wb, "SubTask", models, sub_groups, sub_labels,
                             sub_m, sub_c, F)
    print("done")

    print("Writing Diseases...", end=" ", flush=True)
    write_grouped_rows_sheet(wb, "Diseases", models, dis_groups, dis_labels,
                             dis_m, dis_c, F)
    print("done")

    print("Writing Task_Total...", end=" ", flush=True)
    write_task_total(wb, models, task_m, sub_m, task_c, sub_c, taxonomy, F)
    print("done")

    print("Writing Disease_Total...", end=" ", flush=True)
    write_disease_total(wb, models, dis_m, dis_c, F)
    print("done")

    if args.ci:
        print("Writing EndoColon_CI...", end=" ", flush=True)
        write_endocolon_ci(wb, models, scopy_groups, SCOPY_LABELS,
                           scopy_m, scopy_c, F)
        print("done")

        print("Writing Task_Total_CI...", end=" ", flush=True)
        write_task_total_ci(wb, models, task_m, sub_m, task_c, sub_c, taxonomy, F)
        print("done")

        print("Writing SubTask_CI...", end=" ", flush=True)
        write_subtask_ci(wb, models, sub_m, sub_c, taxonomy, F)
        print("done")

        print("Writing Disease_Total_CI...", end=" ", flush=True)
        write_disease_total_ci(wb, models, dis_m, dis_c, F)
        print("done")

    wb.close()
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
