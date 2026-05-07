"""Summarize evaluation results EXCLUDING the DI-R subtask.

DI-R (Distractor Rejection) is 100% binary and accounts for
~15% of all rows / ~48% of the DI task.  Because the correct answer is
"No" by construction, models that default to "No" inflate raw accuracy
here. Excluding it gives a cleaner picture of performance on the
remaining, harder question types.

Delegates all computation and formatting to summarize.py — the only
change is a row filter applied after CSV loading.

Usage
-----
    python benchmark/code/summarize_ex_DI-R.py /tmp/bench_merged_228/consolidated.csv
    python benchmark/code/summarize_ex_DI-R.py path/to/consolidated.csv --output /tmp/out.xlsx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import xlsxwriter

from summarize import (
    SCOPY_LABELS,
    Fmt,
    _CONFIG_PATH,
    compute_metrics,
    load_csv,
    load_taxonomy,
    order_diseases,
    order_subtasks,
    order_tasks,
    write_disease_total,
    write_disease_total_ci,
    write_endocolon_ci,
    write_grouped_rows_sheet,
    write_matrix_sheet,
    write_subtask_ci,
    write_task_total,
    write_task_total_ci,
)

EXCLUDE_SUBTASKS = {"DI-R"}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_path", type=Path, help="consolidated results CSV")
    ap.add_argument("--output", type=Path, default=None,
                    help="output xlsx path (default: <csv_dir>/Evaluation_Summary_ex_DI-R.xlsx)")
    ap.add_argument("--include-models", default=None,
                    help="comma-separated model names to include (default: all)")
    ap.add_argument("--ci", action="store_true",
                    help="emit four extra CI sheets (EndoColon_CI, "
                         "Task_Total_CI, SubTask_CI, Disease_Total_CI) "
                         "with 95% half-widths in percentage points, "
                         "computed on the DI-R-excluded subset")
    args = ap.parse_args()

    if not args.csv_path.is_file():
        sys.exit(f"CSV not found: {args.csv_path}")
    out_path = args.output or (args.csv_path.parent / "Evaluation_Summary_ex_DI-R.xlsx")

    # Load and filter
    rows, all_models = load_csv(args.csv_path)
    total_before = len(rows)
    rows = [r for r in rows if r.get("SubTask", "") not in EXCLUDE_SUBTASKS]
    total_after = len(rows)

    if args.include_models:
        keep = {s.strip() for s in args.include_models.split(",") if s.strip()}
        models = [m for m in all_models if m in keep]
    else:
        models = all_models
    if not models:
        sys.exit("No models found")

    taxonomy = load_taxonomy(_CONFIG_PATH)

    excluded = sorted(EXCLUDE_SUBTASKS)
    print(f"input:    {args.csv_path}")
    print(f"output:   {out_path}")
    print(f"excluded: {excluded}")
    print(f"rows:     {total_before} -> {total_after} ({total_before - total_after} removed)")
    print(f"models:   {models}")
    print(f"taxonomy: {len(taxonomy)} entries loaded")
    print()

    # Compute metrics (identical to summarize.py)
    scopy_m, scopy_c = compute_metrics(rows, models, "scopy")
    both_m, both_c = compute_metrics(rows, models, lambda _r: "both")
    task_m, task_c = compute_metrics(rows, models, "Task")
    sub_m, sub_c = compute_metrics(rows, models, "SubTask")
    dis_m, dis_c = compute_metrics(rows, models, "disease", skip_empty_group=True)

    for model in models:
        scopy_m[model]["both"] = both_m[model]["both"]
    scopy_c["both"] = both_c["both"]

    scopy_groups = ["endoscopy", "colonoscopy", "both"]
    task_groups = order_tasks(set(task_c.keys()))
    sub_groups = order_subtasks(set(sub_c.keys()))
    dis_groups = order_diseases(set(dis_c.keys()))

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
