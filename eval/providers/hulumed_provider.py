"""Hulu-Med provider.

Hulu-Med (ZJU-AI4H) ships a custom `HulumedProcessor` whose chat template is
string-only — it does not understand HuggingFace's structured content items
(`{"type":"image","image":pil}`). We must insert `<image>` tokens into the
text ourselves and pass PIL images through the separate `images=` kwarg.

Flow per turn:
  1. Build a messages list where each user/assistant `content` is a plain
     string. For the turn that carries images, prepend N copies of `<image>`
     where N = number of images attached.
  2. text = processor.apply_chat_template(messages, add_generation_prompt=True,
           tokenize=False, image_token="<image>")
  3. inputs = processor(text=text, images=<all PILs>, return_tensors="pt")
  4. output = model.generate(**inputs, ...)
  5. decoded = processor.decode(new_tokens, use_think=False)   # strips <think>

Compatibility shim: ZJU's `processing_hulumed.py` was written against
transformers 4.x and breaks against 5.x in three known ways (signature of
`_get_arguments_from_pretrained`, and two `common_kwargs` KeyErrors in
`_merge_kwargs`). On first load we catch the expected error, patch the
just-downloaded cached file, drop the stale module from `sys.modules`, and
retry. All other machines then "just work" on first use.

The rest (frame loading, subsampling, session lifecycle, JSON parsing) is
identical to the generic TransformersProvider, so we subclass it.
"""

from __future__ import annotations

import glob
import logging
import os
import sys
import time

import torch

from .base import ConversationSession, ProviderError
from . import register
from .transformers_provider import TransformersProvider, _try_parse_json

log = logging.getLogger("benchmark.eval.hulumed")

_IMAGE_TOKEN = "<image>"

# ---------------------------------------------------------------------------
# Runtime compat patches for ZJU's HulumedProcessor on transformers >= 5.x
# ---------------------------------------------------------------------------

_HF_HOME = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
_HULUMED_CACHE_GLOB = os.path.join(
    _HF_HOME,
    "modules/transformers_modules/"
    "ZJU_hyphen_AI4H/Hulu_hyphen_Med_hyphen_*/*/processing_hulumed.py",
)

# (old, new) string replacements. Idempotent: a second application is a no-op.
_PATCH_PAIRS = [
    (
        "def _get_arguments_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):",
        "def _get_arguments_from_pretrained(cls, pretrained_model_name_or_path, processor_dict=None, **kwargs):",
    ),
    (
        "        for modality in default_kwargs:\n"
        "            default_kwargs[modality] = ModelProcessorKwargs._defaults.get(modality, {}).copy()\n"
        "            for modality_key in ModelProcessorKwargs.__annotations__[modality].__annotations__.keys():\n"
        "                if modality_key in tokenizer_init_kwargs:",
        "        for modality in default_kwargs:\n"
        "            default_kwargs[modality] = ModelProcessorKwargs._defaults.get(modality, {}).copy()\n"
        "            _modality_cls = ModelProcessorKwargs.__annotations__.get(modality)\n"
        "            if _modality_cls is None or not hasattr(_modality_cls, \"__annotations__\"):\n"
        "                continue\n"
        "            for modality_key in _modality_cls.__annotations__.keys():\n"
        "                if modality_key in tokenizer_init_kwargs:",
    ),
    (
        "        non_modality_kwargs = set(kwargs) - set(output_kwargs)\n"
        "        for modality in output_kwargs:\n"
        "            for modality_key in ModelProcessorKwargs.__annotations__[modality].__annotations__.keys():",
        "        non_modality_kwargs = set(kwargs) - set(output_kwargs)\n"
        "        for modality in output_kwargs:\n"
        "            _modality_cls = ModelProcessorKwargs.__annotations__.get(modality)\n"
        "            if _modality_cls is None or not hasattr(_modality_cls, \"__annotations__\"):\n"
        "                continue\n"
        "            for modality_key in _modality_cls.__annotations__.keys():",
    ),
]


def _apply_hulumed_file_patches() -> int:
    """Apply string patches to every cached processing_hulumed.py.

    Returns number of files actually modified. Files that are already patched
    (subsequent runs, or a different model size sharing the same source) are
    left untouched and not counted.
    """
    modified = 0
    files = glob.glob(_HULUMED_CACHE_GLOB)
    if not files:
        log.warning("No cached processing_hulumed.py files found to patch.")
    for p in files:
        with open(p) as f:
            s = f.read()
        s2 = s
        for before, after in _PATCH_PAIRS:
            s2 = s2.replace(before, after)
        if s2 != s:
            with open(p, "w") as f:
                f.write(s2)
            modified += 1
            log.info("Patched %s", p)
    return modified


def _drop_hulumed_modules_from_cache() -> None:
    """Force re-import of hulumed modules so patched file contents take effect."""
    doomed = [
        name for name in sys.modules
        if any(
            tag in name
            for tag in (
                "processing_hulumed",
                "image_processing_hulumed",
                "configuration_hulumed",
                "modeling_hulumed",
            )
        )
    ]
    for name in doomed:
        del sys.modules[name]


def _is_known_hulumed_compat_error(e: BaseException) -> bool:
    s = str(e)
    return (
        "_get_arguments_from_pretrained" in s
        or "'common_kwargs'" in s
        or "takes 2 positional arguments but 3 were given" in s
    )


@register("hulumed")
class HuluMedProvider(TransformersProvider):
    """Hulu-Med-{4B,7B} — string-content chat template, manual image tokens."""

    def initialize(self) -> None:
        """Load processor + model. On first run, catches the known
        transformers-5.x incompatibilities, patches the just-downloaded
        cached file, drops the stale module from sys.modules, and retries.
        Second and subsequent runs on the same machine skip the patch-retry
        entirely (patches are idempotent, already applied)."""
        try:
            super().initialize()
            return
        except (TypeError, KeyError) as e:
            if not _is_known_hulumed_compat_error(e):
                raise
            log.warning(
                "Hulu-Med processor is incompatible with this transformers "
                "version (%s). Applying runtime patches and retrying.", e,
            )

        n = _apply_hulumed_file_patches()
        if n == 0:
            raise RuntimeError(
                "Hulu-Med compat issue detected, but no cached "
                "processing_hulumed.py could be patched. Cache layout may "
                "have changed; inspect ~/.cache/huggingface/modules/ "
                "manually."
            )

        _drop_hulumed_modules_from_cache()
        log.info("Patched %d file(s); reloading Hulu-Med processor.", n)
        # Second attempt — if the patches don't cover a new breakage, the
        # error surfaces here cleanly rather than being hidden.
        super().initialize()

    def create_session(self, system_prompt: str) -> ConversationSession:
        session = ConversationSession()
        session.provider_state = {
            # Hulu-Med's chat template expects string content, not structured parts.
            "messages": [{"role": "system", "content": system_prompt}],
            "images": [],  # PIL list attached on turn 1
        }
        return session

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

        # Prepend one <image> token per image on turn 1; subsequent turns are
        # text-only (model references the frames already in history).
        if images:
            prefix = _IMAGE_TOKEN * len(images) + "\n"
            content = prefix + user_text
        else:
            content = user_text
        messages.append({"role": "user", "content": content})

        last_error = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                reply_text = self._generate_hulumed(messages, state["images"])
                parsed = _try_parse_json(reply_text)
                messages.append({"role": "assistant", "content": reply_text})
                session.messages.append({"role": "user", "text": user_text})
                session.messages.append({"role": "assistant", "text": reply_text})
                return parsed if parsed is not None else reply_text
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
            f"Hulu-Med generate failed after {self.config.max_retries} attempts: {last_error}"
        )

    def _generate_hulumed(self, messages: list[dict], images: list) -> str:
        # Step 1: render the conversation to a plain prompt string.
        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            image_token=_IMAGE_TOKEN,
        )

        # Step 2: tokenize text + preprocess images together.
        # Hulu-Med's processor accepts images=None for text-only turns, but we
        # always pass the full image set (they live in turn-1 of the template).
        proc_kwargs = {"text": text, "return_tensors": "pt"}
        if images:
            proc_kwargs["images"] = images
        inputs = self.processor(**proc_kwargs)

        target_device = next(self.model.parameters()).device
        # Hulu-Med's BatchFeature doesn't support .to(device, dtype=...)
        # cleanly; cast manually.
        try:
            inputs = inputs.to(target_device, dtype=self.torch_dtype)
        except TypeError:
            inputs = inputs.to(target_device)
            for k, v in list(inputs.items()):
                if torch.is_tensor(v) and v.is_floating_point():
                    inputs[k] = v.to(self.torch_dtype)

        do_sample = bool(self.config.extra.get("do_sample", False))
        gen_kwargs = dict(
            max_new_tokens=self.config.max_output_tokens,
            do_sample=do_sample,
        )
        if do_sample:
            gen_kwargs["temperature"] = self.config.temperature

        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            out = self.model.generate(**inputs, **gen_kwargs)

        # Hulu-Med's model.generate returns ONLY the new tokens (not input +
        # new). Detect this: if output length is smaller than input, the
        # model already stripped the prompt for us.
        out_len = out.shape[-1]
        if out_len > input_len:
            new_tokens = out[0][input_len:]
        else:
            new_tokens = out[0]

        # Hulu-Med's decoder auto-strips <think>...</think> when use_think=False.
        return self.processor.decode(new_tokens, skip_special_tokens=True, use_think=False).strip()
