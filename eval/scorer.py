"""Answer scoring and metrics aggregation."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Per-question scoring
# ---------------------------------------------------------------------------

def extract_answer(
    response: dict | str | None,
    question: dict,
) -> str | list[str] | None:
    """
    Extract and normalize the answer from a provider response.

    Handles:
    - String responses (parse as JSON)
    - Extra keys in response dict
    - Multi-select answer given as string → wrap in list
    - Validate against enum values
    """
    if response is None:
        return None

    if isinstance(response, str):
        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            # Try regex extraction
            m = re.search(r'"answer"\s*:\s*"([^"]+)"', response)
            if m:
                response = {"answer": m.group(1)}
            else:
                return None

    if not isinstance(response, dict):
        return None

    answer = response.get("answer")
    if answer is None:
        return None

    valid_values = _get_valid_values(question["answer_schema"])

    if question.get("multi_select"):
        if isinstance(answer, str):
            answer = [answer]
        if not isinstance(answer, list):
            return None
        answer = [a for a in answer if a in valid_values]
        return answer if answer else None
    else:
        return answer if answer in valid_values else None


def _get_valid_values(schema: dict) -> set[str]:
    """Extract valid enum values from an answer_schema."""
    props = schema.get("properties", {})
    answer_prop = props.get("answer", {})
    if answer_prop.get("type") == "array":
        return set(answer_prop.get("items", {}).get("enum", []))
    return set(answer_prop.get("enum", []))


def score_question(
    model_answer: str | list[str] | None,
    correct_answer: str | list[str],
    multi_select: bool,
) -> bool:
    """Score a single question. Returns True for exact match."""
    if model_answer is None:
        return False
    if multi_select:
        return set(model_answer) == set(correct_answer)
    return model_answer == correct_answer


def compute_multi_select_metrics(
    model_answer: list[str] | None,
    correct_answer: list[str],
) -> dict:
    """Compute precision, recall, F1 for multi-select questions."""
    if model_answer is None:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact_match": False}
    pred = set(model_answer)
    gold = set(correct_answer)
    tp = len(pred & gold)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "exact_match": pred == gold,
    }


# ---------------------------------------------------------------------------
# Run-level aggregation
# ---------------------------------------------------------------------------

def score_run(predictions_dir: str | Path) -> dict:
    """
    Aggregate scores across all prediction files.

    Args:
        predictions_dir: Directory containing per-case prediction JSONs.

    Returns:
        Summary dict with accuracy broken down by multiple dimensions.
    """
    predictions_dir = Path(predictions_dir)
    all_results: list[dict] = []
    case_accuracies: list[dict] = []

    for path in sorted(predictions_dir.glob("*.json")):
        data = json.loads(path.read_text())
        case_id = data["case_id"]
        proc = data.get("procedure_type", "unknown")
        diseases = data.get("diseases_found", [])

        for r in data.get("results", []):
            r["_case_id"] = case_id
            r["_procedure_type"] = proc
            r["_diseases"] = diseases
            all_results.append(r)

        case_accuracies.append({
            "case_id": case_id,
            "procedure_type": proc,
            "accuracy": data.get("case_accuracy", 0.0),
        })

    if not all_results:
        return {"total_questions": 0, "overall_accuracy": 0.0}

    total = len(all_results)
    correct = sum(1 for r in all_results if r.get("is_correct"))

    summary: dict[str, Any] = {
        "total_cases": len(case_accuracies),
        "total_questions": total,
        "total_correct": correct,
        "overall_accuracy": round(correct / total, 4),
    }

    # Aggregate by multiple dimensions
    dimensions = [
        ("by_procedure_type", "_procedure_type"),
        ("by_task", "task"),
        ("by_subtask", "subtask"),
        ("by_phase", "phase"),
        ("by_type", "type"),
    ]

    for dim_name, key in dimensions:
        summary[dim_name] = _aggregate_by(all_results, key)

    # By multi_select
    summary["by_multi_select"] = _aggregate_by(
        all_results, "multi_select", key_transform=str
    )

    # By disease (flatten diseases_found per result)
    disease_groups: dict[str, list[dict]] = defaultdict(list)
    for r in all_results:
        disease = r.get("disease") or r.get("_diseases", ["unknown"])[0]
        disease_groups[disease].append(r)
    summary["by_disease"] = {
        d: _compute_group_metrics(results)
        for d, results in sorted(disease_groups.items())
    }

    # Task × subtask matrix
    summary["task_subtask_matrix"] = _build_task_subtask_matrix(all_results)

    return summary


def _aggregate_by(
    results: list[dict],
    key: str,
    key_transform=None,
) -> dict[str, dict]:
    """Group results by a key and compute metrics for each group."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        val = r.get(key, "unknown")
        if key_transform:
            val = key_transform(val)
        groups[val].append(r)

    return {
        k: _compute_group_metrics(v)
        for k, v in sorted(groups.items())
    }


def _compute_group_metrics(results: list[dict]) -> dict:
    """Compute accuracy and optional multi-select F1 for a group."""
    total = len(results)
    correct = sum(1 for r in results if r.get("is_correct"))
    metrics: dict[str, Any] = {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
    }

    # If any multi-select questions, compute mean F1
    ms_results = [r for r in results if r.get("multi_select")]
    if ms_results:
        f1_scores = []
        for r in ms_results:
            m = compute_multi_select_metrics(
                r.get("model_answer"), r.get("correct_answer", [])
            )
            f1_scores.append(m["f1"])
        metrics["multi_select_count"] = len(ms_results)
        metrics["multi_select_mean_f1"] = round(
            sum(f1_scores) / len(f1_scores), 4
        ) if f1_scores else 0.0

    return metrics


def _build_task_subtask_matrix(results: list[dict]) -> dict:
    """Build a task → {subtask → metrics} matrix."""
    task_groups: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in results:
        task = r.get("task", "unknown")
        subtask = r.get("subtask", "unknown")
        task_groups[task][subtask].append(r)

    matrix = {}
    for task in sorted(task_groups):
        task_all = []
        subtasks = {}
        for subtask in sorted(task_groups[task]):
            sub_results = task_groups[task][subtask]
            task_all.extend(sub_results)
            subtasks[subtask] = _compute_group_metrics(sub_results)
        matrix[task] = {
            "_total": _compute_group_metrics(task_all),
            **subtasks,
        }

    return matrix
