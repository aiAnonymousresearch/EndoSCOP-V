#!/usr/bin/env python3
"""EndoSCOP-V evaluation entry point.

Reads `config.yaml` and either:
  - runs every model under `models:` with `enabled: true` (default), or
  - runs a single model when --provider and --model are passed (one-off mode).

Per-model predictions are written flat to:
  <results_dir>/<results_name>/<case_id>.json

That layout is what consolidate_result_json.py and summarize.py expect.

Usage
-----
    # Run all enabled models from config.yaml
    python evaluate.py

    # Smoke test on one case across all enabled models
    python evaluate.py --case-ids e1372

    # Single model, ad-hoc (bypasses the models: block)
    python evaluate.py --provider gemini --model gemini-3-flash-preview

    # Validate the pipeline without making any API/model calls
    python evaluate.py --dry-run --case-ids e1372
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

# Make `eval/` importable when running this script directly.
_DIST_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIST_ROOT))

from eval import run as eval_run  # noqa: E402

log = logging.getLogger("evaluate")


def _set_credential_envs(creds: dict) -> None:
    """Promote `*_file` credentials into env vars so providers see a uniform API.

    config.yaml lets users supply credentials either via env var name
    (default) or by pointing to a key file. This reads any *_file paths
    and exports the contents to the matching *_env var so provider code
    only has to check os.environ.
    """
    pairs = [
        ("hf_token_file", "hf_token_env", "HF_TOKEN"),
        ("gemini_api_key_file", "gemini_api_key_env", "GOOGLE_API_KEY"),
        ("google_app_credentials_file", "google_app_credentials_env",
         "GOOGLE_APPLICATION_CREDENTIALS"),
        ("anthropic_api_key_file", "anthropic_api_key_env", "ANTHROPIC_API_KEY"),
        ("openai_api_key_file", "openai_api_key_env", "OPENAI_API_KEY"),
    ]
    for file_key, env_key, default_env in pairs:
        env_name = creds.get(env_key, default_env)
        # If the env is already set, leave it.
        if os.environ.get(env_name):
            continue
        file_path = creds.get(file_key)
        if file_path and Path(file_path).is_file():
            value = Path(file_path).read_text().strip()
            # GOOGLE_APPLICATION_CREDENTIALS expects a path, not file contents
            if env_name == "GOOGLE_APPLICATION_CREDENTIALS":
                os.environ[env_name] = file_path
            else:
                os.environ[env_name] = value
            log.info("Loaded %s from %s", env_name, file_path)


def _build_args_namespace(
    base: argparse.Namespace,
    provider: str,
    model: str,
    results_name: str,
    frame_count: int | None,
    procedure_filter: str | None,
) -> argparse.Namespace:
    """Build an argparse.Namespace that eval.run.run_evaluation expects."""
    return argparse.Namespace(
        config=base.config,
        provider=provider,
        model=model,
        cases_dir=base.cases_dir,
        output_dir=base.output_dir,
        results_name=results_name,
        case_ids=base.case_ids,
        procedure_type=procedure_filter or base.procedure_type,
        temperature=base.temperature,
        frame_count=frame_count,
        api_delay=base.api_delay,
        device=base.device,
        resize_to=base.resize_to,
        resume=base.resume,
        dry_run=base.dry_run,
        verbose=base.verbose,
    )


def _apply_provider_config(cfg: dict, provider: str, provider_cfg: dict) -> None:
    """Merge a per-model `provider_config:` block into the top-level cfg.

    eval/run.py reads provider extras from cfg[provider_name] (or the merged
    transformers section for the transformers family). For multi-model runs
    we mutate cfg in-place between models so each gets its own block.
    """
    section = dict(cfg.get(provider, {}))
    section.update(provider_cfg or {})
    cfg[provider] = section


def _write_temp_config_for_run(cfg: dict, tmp_path: Path) -> None:
    """Write the (possibly mutated) cfg to a temp YAML so run.py picks it up."""
    tmp_path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run EndoSCOP-V multi-model evaluation from config.yaml.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.yaml (default: ./config.yaml)")
    parser.add_argument("--provider", type=str,
                        help="Run a single provider (skips models: block)")
    parser.add_argument("--model", type=str,
                        help="Model name (used with --provider)")
    parser.add_argument("--results-name", type=str, default=None,
                        help="Override results folder name (single-model mode only)")
    parser.add_argument("--cases-dir", type=str, help="Path to cases directory")
    parser.add_argument("--output-dir", type=str, help="Per-model results root")
    parser.add_argument("--case-ids", nargs="+", help="Run only these case IDs")
    parser.add_argument("--procedure-type",
                        choices=["endoscopy", "colonoscopy"],
                        help="Filter cases by procedure type")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--frame-count", type=int, default=None,
                        help="Override frames per case (single-model mode)")
    parser.add_argument("--api-delay", type=float, default=None)
    parser.add_argument("--device", type=str, default=None,
                        help="cuda:0 | cuda:1 | auto | cpu")
    parser.add_argument("--resize-to", type=int, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Skip cases already in <results-name>/")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet third-party libs whose INFO output is just one line per HTTP call —
    # useful for debugging auth, deafening during a normal multi-turn run.
    if not args.verbose:
        for noisy in ("httpx", "google_genai.models", "google.auth",
                      "urllib3", "anthropic"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    config_path = Path(args.config) if args.config else _DIST_ROOT / "config.yaml"
    if not config_path.is_file():
        log.error("config.yaml not found at %s", config_path)
        sys.exit(1)
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}

    _set_credential_envs(cfg.get("credentials", {}) or {})

    # Single-model mode: --provider AND --model given
    if args.provider and args.model:
        # Look up matching models: entry to inherit results_name, frames, and
        # procedure_filter from config so single-model and multi-model runs
        # behave identically. CLI flags still win when explicitly passed.
        matched_entry: dict | None = None
        for _mid, _mcfg in (cfg.get("models") or {}).items():
            if (_mcfg.get("provider") == args.provider
                    and _mcfg.get("model") == args.model):
                matched_entry = _mcfg
                if not args.results_name:
                    args.results_name = _mcfg.get("results_name", _mid)
                break
        if not args.results_name:
            slug = args.model.replace("/", "--").replace(" ", "_")
            args.results_name = f"{args.provider}__{slug}"
        frame_count = (
            args.frame_count if args.frame_count is not None
            else (matched_entry.get("frames") if matched_entry else None)
        )
        procedure_filter = (
            args.procedure_type
            or (matched_entry.get("procedure_filter") if matched_entry else None)
        )
        log.info("Single-model run: %s/%s -> %s",
                 args.provider, args.model, args.results_name)
        run_args = _build_args_namespace(
            args, args.provider, args.model, args.results_name,
            frame_count, procedure_filter,
        )
        eval_run.run_evaluation(run_args)
        return

    # Multi-model mode: iterate enabled models
    models = cfg.get("models", {}) or {}
    enabled = [(mid, mcfg) for mid, mcfg in models.items()
               if mcfg.get("enabled", False)]
    if not enabled:
        log.error(
            "No models with `enabled: true` in %s. "
            "Edit config.yaml or pass --provider/--model.",
            config_path,
        )
        sys.exit(1)

    log.info("Running %d enabled model(s): %s",
             len(enabled), [m[0] for m in enabled])

    # Per-model run loop. We mutate cfg[provider] in place via the temp YAML
    # so each model can supply its own provider_config block.
    failures: list[str] = []
    tmp_cfg_path = _DIST_ROOT / "evaluations" / ".evaluate_runtime_config.yaml"
    tmp_cfg_path.parent.mkdir(parents=True, exist_ok=True)

    for mid, mcfg in enabled:
        provider = mcfg.get("provider")
        model = mcfg.get("model")
        if not provider or not model:
            log.warning("Skipping model %s — missing provider or model.", mid)
            continue
        results_name = mcfg.get("results_name", mid)
        frame_count = mcfg.get("frames")
        procedure_filter = mcfg.get("procedure_filter")

        # Build a fresh cfg copy with this model's provider_config merged in
        cfg_for_run = dict(cfg)
        if "provider_config" in mcfg:
            _apply_provider_config(cfg_for_run, provider, mcfg["provider_config"])
        _write_temp_config_for_run(cfg_for_run, tmp_cfg_path)

        log.info("=" * 70)
        log.info("Model: %s  (%s/%s)  -> %s  [frames=%s, procedure_filter=%s]",
                 mid, provider, model, results_name, frame_count, procedure_filter)
        log.info("=" * 70)

        run_args = _build_args_namespace(
            args, provider, model, results_name, frame_count, procedure_filter,
        )
        # Point run.py at the per-model temp config
        run_args.config = str(tmp_cfg_path)
        try:
            eval_run.run_evaluation(run_args)
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001
            log.exception("Model %s failed: %s", mid, e)
            failures.append(mid)

    # Tidy up temp config
    try:
        tmp_cfg_path.unlink()
    except OSError:
        pass

    if failures:
        log.warning("Models with errors: %s", failures)
        sys.exit(2)
    log.info("All %d enabled models complete.", len(enabled))


if __name__ == "__main__":
    main()
