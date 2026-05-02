"""ColonR1 provider.

ColonR1 (ai4colonoscopy/ColonR1) is a Qwen2.5-VL-3B-Instruct checkpoint
fine-tuned with GRPO ("R1-style" reasoning) for colonoscopy VQA. It is NOT a
Mixture-of-Experts model — the "R1" refers to the DeepSeek-R1 training
methodology.

Output contract (per the repo's quickstart.py):

    <think>step-by-step reasoning...</think><answer>X</answer>

So this provider differs from the generic TransformersProvider in three ways:

1. **Task suffix** appended to every user question, instructing the model to
   emit its reasoning in <think>...</think> then its answer in
   <answer>...</answer>.

2. **Answer extraction** via a three-pass matcher: exact schema-enum hit,
   stripped literal, then word-boundary fuzzy match.

3. **History compaction** — we strip <think>...</think> blocks from prior
   assistant turns before re-passing the conversation. Qwen2.5-VL-3B has a
   32K context and thinking traces can easily be ~300-1000 tokens each. At
   100 vision-frames + 10 turns of raw thinking, context saturates fast.
   Keeping only the final <answer> in history bounds the cost.

The model was trained on single-turn colonoscopy VQA, so multi-turn
performance is expected to be mildly degraded vs. its trained distribution.
We use multi-turn anyway for apples-to-apples comparison with the other
models in the benchmark.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import torch

from .base import ConversationSession, ProviderError
from . import register
from .transformers_provider import TransformersProvider

log = logging.getLogger("eval.colonr1")


_COLONR1_REPO = "ai4colonoscopy/ColonR1"
# The published HF snapshot has weight files with extra version suffixes
# (`-003`, `-001`) that don't match the canonical names referenced in
# `model.safetensors.index.json`. Without symlinks, transformers raises
# `OSError: ... does not appear to have files named ...`.
_COLONR1_WEIGHT_SYMLINKS = [
    ("model-00001-of-00002.safetensors", "model-00001-of-00002-003.safetensors"),
    ("model-00002-of-00002.safetensors", "model-00002-of-00002-001.safetensors"),
]


def _heal_colonr1_weights(snapshot_dir: Path) -> None:
    """Create canonical-name symlinks pointing to the version-suffixed weights."""
    for canonical, suffixed in _COLONR1_WEIGHT_SYMLINKS:
        canonical_path = snapshot_dir / canonical
        suffixed_path = snapshot_dir / suffixed
        if canonical_path.exists() or canonical_path.is_symlink():
            continue
        if not suffixed_path.exists():
            log.warning(
                "ColonR1 weight self-heal: neither %s nor %s found in %s",
                canonical, suffixed, snapshot_dir,
            )
            continue
        canonical_path.symlink_to(suffixed)
        log.info("ColonR1 weight self-heal: linked %s -> %s", canonical, suffixed)


TASK_SUFFIX = (
    "\n\nYour task: 1. First, Think through the question step by step, "
    "enclose your reasoning process in <think>...</think> tags. "
    "2. Then provide the correct answer inside <answer>...</answer> tags. "
    "3. No extra information or text outside of these tags."
)

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_ANSWER_BLOCK_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)


@register("colonr1")
class ColonR1Provider(TransformersProvider):
    """ColonR1 — Qwen2.5-VL-3B with GRPO-thinking output format."""

    def initialize(self) -> None:
        # Heal the published snapshot's broken filenames before super() loads
        # weights via from_pretrained. snapshot_download is a no-op if the
        # files are already cached.
        if self.config.model_name == _COLONR1_REPO:
            try:
                from huggingface_hub import snapshot_download
                snapshot_dir = Path(snapshot_download(_COLONR1_REPO))
                _heal_colonr1_weights(snapshot_dir)
            except Exception as e:  # noqa: BLE001
                log.warning("ColonR1 weight self-heal skipped: %s", e)
        super().initialize()

    def send_turn(
        self,
        session: ConversationSession,
        user_text: str,
        answer_schema: dict,
        is_first_turn: bool,
    ) -> dict | str:
        state = session.provider_state
        messages = state["messages"]
        images = state["images"] if is_first_turn else []

        # Append the ColonR1 task suffix so the model emits <think>/<answer>.
        user_text_with_suffix = user_text + TASK_SUFFIX

        user_content: list[dict] = []
        for img in images:
            user_content.append({"type": "image", "image": img})
        user_content.append({"type": "text", "text": user_text_with_suffix})
        messages.append({"role": "user", "content": user_content})

        last_error = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                raw = self._generate(messages)
                # For history: keep only the extracted <answer>...</answer>
                # (or a compact fallback). The raw full reasoning is not
                # useful to carry forward and blows up context.
                compact_reply = _compact_for_history(raw)
                parsed = _extract_structured_answer(raw, answer_schema)

                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": compact_reply}],
                })
                session.messages.append({"role": "user", "text": user_text})
                # Store the raw (with <think>) in session.messages for audit,
                # but NOT in the state['messages'] that gets re-passed.
                session.messages.append({"role": "assistant", "text": raw})

                # Return a dict in the standard shape so scorer.extract_answer
                # treats it the same as every other provider.
                if parsed is not None:
                    return parsed
                # Fall back to the raw text; scorer will regex-hunt.
                return raw

            except torch.cuda.OutOfMemoryError as e:
                last_error = e
                log.error("CUDA OOM on attempt %d: %s", attempt, e)
                torch.cuda.empty_cache()
            except Exception as e:  # noqa: BLE001
                last_error = e
                log.warning("Generate attempt %d failed: %s", attempt, e)

            if attempt < self.config.max_retries:
                time.sleep(1.0)

        messages.pop()
        raise ProviderError(
            f"ColonR1 generate failed after {self.config.max_retries} attempts: {last_error}"
        )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _compact_for_history(raw: str) -> str:
    """Strip <think>...</think> and keep only <answer>...</answer> (or a
    one-line fallback) so prior turns don't blow up the context window."""
    # Remove reasoning.
    without_think = _THINK_BLOCK_RE.sub("", raw).strip()
    # Prefer the answer block if present.
    m = _ANSWER_BLOCK_RE.search(without_think)
    if m:
        return f"<answer>{m.group(1).strip()}</answer>"
    # Model didn't follow format — keep a short snippet.
    short = without_think.split("\n", 1)[0][:200].strip()
    return short or raw[:200]


def _extract_structured_answer(raw: str, answer_schema: dict):
    """Extract the answer from a ColonR1 reply and map it to the question's
    schema. Returns a dict like {"answer": "C"} or {"answer": ["B","C"]}, or
    None if nothing matched.

    Three-pass matcher:
      1. exact match vs. enum
      2. whitespace/punctuation-stripped literal
      3. word-boundary regex hit for any enum value
    """
    props = answer_schema.get("properties", {}) or {}
    answer_prop = props.get("answer", {}) or {}
    is_array = answer_prop.get("type") == "array"
    if is_array:
        enum = list(answer_prop.get("items", {}).get("enum", []))
    else:
        enum = list(answer_prop.get("enum", []))
    if not enum:
        return None

    # Strip <think>...</think>.
    text = _THINK_BLOCK_RE.sub("", raw).strip()
    # Pull content from <answer>...</answer>; fall back to post-think text.
    m = _ANSWER_BLOCK_RE.search(text)
    answer_text = (m.group(1) if m else text).strip()

    # Pass 1: exact.
    if answer_text in enum:
        return {"answer": [answer_text]} if is_array else {"answer": answer_text}

    # Pass 2: stripped of trailing punctuation and whitespace.
    stripped = re.sub(r"[\s\.\,\;\:\!\?\"\']+$", "", answer_text).strip()
    stripped = re.sub(r"^[\s\.\,\;\:\!\?\"\']+", "", stripped).strip()
    if stripped in enum:
        return {"answer": [stripped]} if is_array else {"answer": stripped}

    # Pass 3: word-boundary regex hit for each enum value.
    # For multi-select schemas, collect every value mentioned.
    hits = []
    for v in enum:
        # Use word boundary; escape for safety.
        pat = r"\b" + re.escape(v) + r"\b"
        if re.search(pat, answer_text, re.IGNORECASE):
            hits.append(v)

    if not hits:
        return None

    if is_array:
        # Preserve enum order.
        ordered = [v for v in enum if v in hits]
        return {"answer": ordered}

    # Single-select with multiple candidate mentions — pick the first.
    return {"answer": hits[0]}
