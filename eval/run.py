#!/usr/bin/env python3
"""
EndoSCOP-V single-model evaluation runner.

For multi-model runs driven by config.yaml's `models:` block, use evaluate.py
instead — it iterates enabled models and calls run_evaluation per model with
--results-name set.

Usage:
    python -m eval                                       # config.yaml defaults
    python -m eval --config path/to/config.yaml
    python -m eval --provider openai --model gpt-4o
    python -m eval --case-ids e1013 c1003
    python -m eval --procedure-type endoscopy
    python -m eval --resume
    python -m eval --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
from tqdm import tqdm

from .prompts import build_system_prompt, build_turn_message
from .providers import create_provider
from .providers.base import ConversationSession, ProviderConfig, ProviderError
from .scorer import extract_answer, score_question, score_run

_CASE_ID_RE = re.compile(r"^[ec]\d{4}$")

log = logging.getLogger("eval")

_EVAL_DIR = Path(__file__).resolve().parent


def _find_project_root() -> Path:
    """Walk up from eval/ looking for config.yaml.

    Distribution layout:    distribution/eval/run.py  →  distribution/
    Legacy benchmark/eval/: benchmark/eval/run.py     →  endo_ehr/
    """
    p = _EVAL_DIR
    for _ in range(4):
        if (p / "config.yaml").exists():
            return p
        p = p.parent
    return _EVAL_DIR.parent


_PROJECT_ROOT = _find_project_root()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_config(config_path: str | None) -> dict:
    """Load YAML config, falling back to bundled default."""
    if config_path:
        path = Path(config_path)
    else:
        # Prefer project root (distribution layout), then eval/ subdir (legacy).
        path = _PROJECT_ROOT / "config.yaml"
        if not path.exists():
            path = _EVAL_DIR / "config.yaml"

    if path.exists():
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
        log.info("Loaded config from %s", path)
        return cfg

    log.warning("Config file not found: %s — using defaults", path)
    return {}


def _resolve_path(path_str: str, base: Path | None = None) -> Path:
    """Resolve a path relative to project root (or `base`) if not absolute."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (base or _PROJECT_ROOT) / p


# ---------------------------------------------------------------------------
# Case loading
# ---------------------------------------------------------------------------

def discover_case_dirs(cases_dir: Path) -> dict[str, Path]:
    """
    Walk cases_dir recursively; return {case_id: case_dir} for every folder
    whose basename matches [ec]\\d{4}. If a case_id appears more than once,
    the first-seen path wins and a warning is logged.
    """
    found: dict[str, Path] = {}
    for p in cases_dir.rglob("*"):
        if not p.is_dir():
            continue
        if not _CASE_ID_RE.match(p.name):
            continue
        if p.name in found:
            log.warning(
                "Duplicate case id %s at %s (keeping %s)",
                p.name, p, found[p.name],
            )
            continue
        found[p.name] = p
    return found


def load_cases(
    cases_dir: Path,
    case_ids: list[str] | None = None,
    procedure_type: str | None = None,
    frames_subdir_suffix: str | None = None,
) -> list[tuple[dict, Path]]:
    """
    Discover case folders under cases_dir and load their QA JSON.

    Returns a list of (qa_data, case_dir) tuples, sorted by case_id.

    Cases that have neither pre-extracted frames nor an mp4 are silently
    dropped — there is nothing to feed the model. Cases the caller named
    explicitly via case_ids are kept regardless, so an explicit "run case
    cXXXX" still surfaces a loud error if the media is missing.
    """
    index = discover_case_dirs(cases_dir)

    if case_ids:
        missing = [cid for cid in case_ids if cid not in index]
        if missing:
            log.warning("Case ids not found under %s: %s", cases_dir, missing)
        selected = [(cid, index[cid]) for cid in case_ids if cid in index]
    else:
        selected = sorted(index.items())

    suffix = frames_subdir_suffix or "_siglip_896"
    explicit = set(case_ids or [])
    out: list[tuple[dict, Path]] = []
    skipped_no_media = 0
    for cid, cdir in selected:
        qa_path = cdir / f"{cid}_qa.json"
        if not qa_path.exists():
            log.warning("Missing QA JSON for %s: %s", cid, qa_path)
            continue
        with open(qa_path) as f:
            data = json.load(f)
        if procedure_type and data.get("procedure_type") != procedure_type:
            continue
        if cid not in explicit:
            video_path = cdir / data.get("video", f"{cid}.mp4")
            frames_dir = cdir / f"{cid}{suffix}"
            if not video_path.exists() and not frames_dir.is_dir():
                skipped_no_media += 1
                continue
        out.append((data, cdir))
    if skipped_no_media:
        log.info(
            "Skipped %d case(s) with no video/frames (QA-only). "
            "Pass --case-ids to force-include a specific case.",
            skipped_no_media,
        )
    return out


# ---------------------------------------------------------------------------
# Per-case evaluation
# ---------------------------------------------------------------------------

def evaluate_case(
    provider,
    qa_data: dict,
    case_dir: Path,
    frames_cfg: dict,
    dry_run: bool = False,
) -> dict:
    """Run multi-turn conversation for a single case."""
    case_id = qa_data["case_id"]

    # Preferred input: pre-extracted SigLIP frames in {case_id}{suffix}/
    subdir_suffix = frames_cfg.get("subdir_suffix", "_siglip_896")
    frames_dir = case_dir / f"{case_id}{subdir_suffix}"
    video_path = case_dir / qa_data.get("video", f"{case_id}.mp4")

    if not frames_dir.is_dir() and not video_path.exists():
        raise FileNotFoundError(
            f"Neither frames dir nor video found for {case_id}: "
            f"{frames_dir} / {video_path}"
        )

    system_prompt = build_system_prompt(qa_data["procedure_type"])
    session = provider.create_session(system_prompt)

    # Providers that prefer frames should read session.frames_dir;
    # video-native providers (gemini) fall back to the mp4 path.
    session.frames_dir = str(frames_dir) if frames_dir.is_dir() else None
    session.case_dir = str(case_dir)
    provider.load_video(str(video_path), session)

    results = []
    for question in qa_data["questions"]:
        user_text = build_turn_message(question)
        is_first = question["turn"] == 1

        if dry_run:
            results.append(_make_result(question, None, None, False))
            continue

        if provider.config.api_delay > 0 and not is_first:
            time.sleep(provider.config.api_delay)

        try:
            raw_response = provider.send_turn(
                session=session,
                user_text=user_text,
                answer_schema=question["answer_schema"],
                is_first_turn=is_first,
            )
            model_answer = extract_answer(raw_response, question)
            is_correct = score_question(
                model_answer, question["correct"], question.get("multi_select", False)
            )
            results.append(_make_result(
                question, model_answer, json.dumps(raw_response), is_correct
            ))
            log.debug(
                "  Q%d (%s/%s) %s | model=%r | gold=%r | %s",
                question["question_number"],
                question.get("task", ""),
                question.get("subtask", ""),
                question.get("type", ""),
                model_answer,
                question["correct"],
                "PASS" if is_correct else "FAIL",
            )

        except Exception as e:
            log.warning("Turn %d failed for %s: %s", question["turn"], case_id, e)
            results.append(_make_result(
                question, None, str(e), False, error=str(e)
            ))
            # If turn 1 (the image-bearing turn) fails, the model never saw
            # the frames — subsequent text-only turns are meaningless. Abort
            # the case so it shows up in errors.json and isn't silently scored.
            if is_first:
                raise ProviderError(
                    f"Turn 1 failed for {case_id}; aborting case: {e}"
                ) from e

    provider.cleanup(session)

    answered = [r for r in results if r["model_answer"] is not None]
    case_accuracy = (
        sum(r["is_correct"] for r in answered) / len(answered)
        if answered else 0.0
    )

    return {
        "case_id": case_id,
        "procedure_type": qa_data["procedure_type"],
        "diseases_found": qa_data.get("diseases_found", []),
        "total_questions": len(results),
        "results": results,
        "case_accuracy": round(case_accuracy, 4),
    }


def _make_result(
    question: dict,
    model_answer,
    model_raw: str | None,
    is_correct: bool,
    error: str | None = None,
) -> dict:
    """Build a result dict for one question."""
    r = {
        "question_number": question["question_number"],
        "turn": question["turn"],
        "phase": question.get("phase", ""),
        "disease": question.get("disease", ""),
        "task": question.get("task", ""),
        "subtask": question.get("subtask", ""),
        "type": question.get("type", ""),
        "multi_select": question.get("multi_select", False),
        "correct_answer": question["correct"],
        "model_answer": model_answer,
        "model_raw": model_raw,
        "is_correct": is_correct,
    }
    if error:
        r["error"] = error
    return r


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run_evaluation(args: argparse.Namespace) -> None:
    """Main evaluation loop."""
    cfg = _load_config(args.config)

    # Merge config with CLI overrides. Supports both the new distribution
    # schema (paths.*, defaults.*) and the legacy benchmark/eval schema.
    paths_cfg = cfg.get("paths", {})
    defaults_cfg = cfg.get("defaults", {})

    provider_name = args.provider or cfg.get("provider", "gemini")
    model_name = args.model or cfg.get("model", "gemini-2.5-flash")
    # cases_dir resolution priority: CLI > ENDO_CASES_DIR env > config > default
    cases_dir_raw = (
        args.cases_dir
        or os.environ.get("ENDO_CASES_DIR")
        or paths_cfg.get("cases_dir")
        or cfg.get("cases_dir", "data/cases/")
    )
    cases_dir = _resolve_path(cases_dir_raw)
    output_dir_raw = (
        args.output_dir
        or paths_cfg.get("results_dir")
        or cfg.get("output_dir", "evaluations/evaluated_cases/")
    )
    output_dir = _resolve_path(output_dir_raw)

    frames_cfg = cfg.get("frames", {})
    frame_count = (
        args.frame_count if args.frame_count is not None
        else frames_cfg.get("sample_count", 32)
    )
    frame_strategy = frames_cfg.get("sample_strategy", "uniform")

    model_params = cfg.get("model_params", {})
    temperature = (
        args.temperature
        if args.temperature is not None
        else defaults_cfg.get("temperature", model_params.get("temperature", 0.0))
    )
    max_tokens = defaults_cfg.get(
        "max_new_tokens",
        model_params.get(
            "max_new_tokens", model_params.get("max_output_tokens", 256)
        ),
    )
    do_sample = model_params.get("do_sample", False)

    api_cfg = cfg.get("api", {})
    api_delay = args.api_delay if args.api_delay is not None else api_cfg.get("delay", 1.0)
    max_retries = api_cfg.get("max_retries", 3)

    # Provider-specific extra config. Providers in the transformers family
    # (transformers, hulumed, colonr1) share the `transformers:` config
    # section; provider-name-specific overrides merge on top.
    _TRANSFORMERS_FAMILY = {"transformers", "hulumed", "colonr1"}
    if provider_name in _TRANSFORMERS_FAMILY:
        extra = {**cfg.get("transformers", {}), **cfg.get(provider_name, {})}
    else:
        extra = cfg.get(provider_name, {})

    # CLI --device override > defaults.device > provider section's device
    if args.device is not None:
        extra["device"] = args.device
    elif "device" in defaults_cfg and "device" not in extra:
        extra["device"] = defaults_cfg["device"]
    # CLI --resize-to override (only meaningful for transformers-family providers)
    if args.resize_to is not None:
        extra["resize_to"] = args.resize_to

    # Output layout. Two modes:
    #   1. --results-name MODEL  (distribution): flat folder, no timestamp,
    #      predictions land at output_dir/MODEL/<case_id>.json. Used by the
    #      multi-model evaluate.py wrapper so consolidate_result_json.py can
    #      read predictions directly.
    #   2. timestamped run-dir (legacy): output_dir/{provider}_{model}_{ts}/
    #      predictions/<case_id>.json + config.json + summary.json + errors.json.
    flat_layout = bool(getattr(args, "results_name", None))
    if flat_layout:
        run_dir = output_dir / args.results_name
        run_dir.mkdir(parents=True, exist_ok=True)
        pred_dir = run_dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_slug = model_name.replace("/", "--").replace(" ", "_")
        run_name = f"{provider_name}_{model_slug}_{timestamp}"
        run_dir = output_dir / run_name

        # Resume: find latest matching run
        if args.resume:
            prefix = f"{provider_name}_{model_slug}_"
            existing = sorted(
                d for d in output_dir.iterdir()
                if d.is_dir() and d.name.startswith(prefix)
            )
            if existing:
                run_dir = existing[-1]
                log.info("Resuming run: %s", run_dir)
            else:
                log.info("No previous run to resume, starting fresh.")

        run_dir.mkdir(parents=True, exist_ok=True)
        pred_dir = run_dir / "predictions"
        pred_dir.mkdir(exist_ok=True)

    # Save run config
    run_config = {
        "provider": provider_name,
        "model": model_name,
        "cases_dir": str(cases_dir),
        "frames_cfg": frames_cfg,
        "frame_count": frame_count,
        "frame_strategy": frame_strategy,
        "temperature": temperature,
        "max_new_tokens": max_tokens,
        "do_sample": do_sample,
        "api_delay": api_delay,
        "max_retries": max_retries,
        "case_ids": args.case_ids,
        "procedure_type": args.procedure_type,
        "dry_run": args.dry_run,
        "timestamp": timestamp,
    }
    if not flat_layout:
        (run_dir / "config.json").write_text(json.dumps(run_config, indent=2))

    # Load cases
    if not cases_dir.is_dir():
        log.error(
            "cases_dir does not exist: %s\n"
            "  Set it via --cases-dir, the ENDO_CASES_DIR env var, or config.yaml.",
            cases_dir,
        )
        sys.exit(1)

    cases = load_cases(
        cases_dir,
        args.case_ids,
        args.procedure_type,
        frames_subdir_suffix=frames_cfg.get("subdir_suffix"),
    )
    if not cases:
        log.error(
            "No [ec]NNNN case folders found under %s\n"
            "  - Expected folders like e1372/ c1249/ (possibly nested in batch_*/)\n"
            "  - Set ENDO_CASES_DIR=<path> or pass --cases-dir <path> to override.",
            cases_dir,
        )
        sys.exit(1)

    # Skip completed cases if resuming
    if args.resume:
        completed = {p.stem for p in pred_dir.glob("*.json")}
        before = len(cases)
        cases = [(qa, cd) for qa, cd in cases if qa["case_id"] not in completed]
        if before > len(cases):
            log.info("Skipping %d already-completed cases.", before - len(cases))

    if frame_count == 0:
        log.info(
            "Evaluating %d cases with %s/%s (native video input)",
            len(cases), provider_name, model_name,
        )
    else:
        log.info(
            "Evaluating %d cases with %s/%s (frames_subdir=%s, sample_count=%s)",
            len(cases), provider_name, model_name,
            frames_cfg.get("subdir_suffix", "_siglip_896"), frame_count,
        )

    if args.dry_run:
        log.info("DRY RUN — no API calls will be made.")

    # Initialize provider
    provider_config = ProviderConfig(
        model_name=model_name,
        temperature=temperature,
        max_output_tokens=max_tokens,
        api_delay=api_delay,
        max_retries=max_retries,
        frame_count=frame_count,
        frame_strategy=frame_strategy,
        extra={**extra, "do_sample": do_sample},
    )

    if not args.dry_run:
        provider = create_provider(provider_name, provider_config)
    else:
        # Dry run: use a mock provider
        provider = _DryRunProvider(provider_config)

    # Evaluate
    errors = []
    for qa_data, case_dir in tqdm(cases, desc="Cases"):
        case_id = qa_data["case_id"]
        try:
            result = evaluate_case(
                provider, qa_data, case_dir, frames_cfg, dry_run=args.dry_run,
            )
            save_path = pred_dir / f"{case_id}.json"
            save_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

        except Exception as e:
            log.exception("Case %s failed", case_id)
            errors.append({"case_id": case_id, "error": str(e)})

    # Save errors (legacy timestamp-dir layout only — flat layout owns the
    # whole folder for case JSONs, so consolidate_result_json.py wouldn't
    # know to skip a metadata file).
    if not flat_layout:
        (run_dir / "errors.json").write_text(json.dumps(errors, indent=2))

    # Score
    if not args.dry_run:
        summary = score_run(pred_dir)
        if not flat_layout:
            (run_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False)
            )
        log.info(
            "Done. Accuracy: %.1f%% (%d/%d). Results: %s",
            summary.get("overall_accuracy", 0.0) * 100,
            summary.get("total_correct", 0),
            summary.get("total_questions", 0),
            run_dir,
        )
    else:
        log.info("Dry run complete. %d cases validated. Dir: %s", len(cases), run_dir)

    if errors and not flat_layout:
        log.warning("%d cases had errors. See %s", len(errors), run_dir / "errors.json")
    elif errors:
        log.warning("%d cases had errors.", len(errors))


# ---------------------------------------------------------------------------
# Dry-run mock provider
# ---------------------------------------------------------------------------

class _DryRunProvider:
    """Placeholder that validates the pipeline without calling APIs."""

    def __init__(self, config):
        self.config = config

    def initialize(self):
        pass

    def create_session(self, system_prompt):
        return ConversationSession()

    def load_video(self, video_path, session):
        # Validation is handled by evaluate_case (frames dir OR video file).
        session.video_loaded = True

    def send_turn(self, session, user_text, answer_schema, is_first_turn):
        return None

    def cleanup(self, session):
        pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run benchmark evaluation on endoscopy/colonoscopy video Q&A.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config YAML (default: ./config.yaml in project root)",
    )
    parser.add_argument(
        "--provider", type=str,
        help="Override provider (gemini|openai|anthropic|transformers|hulumed|colonr1)",
    )
    parser.add_argument("--model", type=str, help="Override model name")
    parser.add_argument("--cases-dir", type=str, help="Path to cases directory")
    parser.add_argument("--output-dir", type=str, help="Results output directory")
    parser.add_argument(
        "--results-name", type=str, default=None,
        help=(
            "If set, predictions are written flat to <output-dir>/<results-name>/"
            "<case_id>.json (no timestamped run-dir). Used by evaluate.py."
        ),
    )

    parser.add_argument("--case-ids", nargs="+", help="Evaluate specific case IDs only")
    parser.add_argument("--procedure-type", choices=["endoscopy", "colonoscopy"], help="Filter by procedure type")

    parser.add_argument("--temperature", type=float, default=None, help="Override temperature")
    parser.add_argument("--frame-count", type=int, default=None, help="Frames sampled from _siglip_896 (null = all)")
    parser.add_argument("--api-delay", type=float, default=None, help="Seconds between turns")
    parser.add_argument("--device", type=str, default=None, help="Override transformers device (cuda:0|cuda:1|auto|cpu)")
    parser.add_argument("--resize-to", type=int, default=None, help="Resize frames to (N,N) before tokenization (transformers family only)")

    parser.add_argument("--resume", action="store_true", help="Resume from latest matching run")
    parser.add_argument("--dry-run", action="store_true", help="Validate pipeline without API calls")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    run_evaluation(args)


if __name__ == "__main__":
    main()
