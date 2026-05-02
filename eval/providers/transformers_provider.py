"""HuggingFace `transformers` provider.

Loads pre-extracted frames (e.g. {case_id}_siglip_896/*.jpg), feeds them as
images in turn 1, and keeps subsequent turns text-only. The full message
history is re-passed on every `generate()` call — this is how multi-turn
works for local models (no server-side session).

Known caveat: MedGemma-4b-it is officially listed as "not optimized for
multi-turn" by Google. We still run it multi-turn because the benchmark is
designed that way; expect some format drift on later turns and rely on the
scorer's regex fallback.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
)

from .base import BaseProvider, ConversationSession, ProviderConfig, ProviderError
from . import register

log = logging.getLogger("benchmark.eval.transformers")


_DTYPE_MAP = {
    "auto": None,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
}


@register("transformers")
class TransformersProvider(BaseProvider):
    """Multimodal HF models (MedGemma, Qwen-VL, Hulu-Med, ...)."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.model = None
        self.processor = None
        self.device = None
        self.torch_dtype = None

    # ---------------- lifecycle ----------------

    def initialize(self) -> None:
        extra = self.config.extra
        device = extra.get("device", "auto")
        dtype_name = extra.get("torch_dtype", "auto")
        attn = extra.get("attn_implementation", "eager")
        trust_remote = bool(extra.get("trust_remote_code", False))

        if dtype_name == "auto":
            self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        else:
            if dtype_name not in _DTYPE_MAP:
                raise ValueError(f"Unknown torch_dtype: {dtype_name}")
            self.torch_dtype = _DTYPE_MAP[dtype_name]

        model_id = self.config.model_name
        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=trust_remote,
        )

        # Pick the right AutoClass: multimodal models usually register as
        # AutoModelForImageTextToText, but some custom architectures (e.g.
        # Hulu-Med, which subclasses Qwen3 CausalLM) register as
        # AutoModelForCausalLM instead. Inspect the model's auto_map.
        model_auto_cls = self._pick_auto_class(model_id, trust_remote)
        log.info("Using %s", model_auto_cls.__name__)

        # max_memory lets device_map="auto" *actually shard* across multiple
        # GPUs rather than dumping the whole model onto the GPU with the most
        # free space. Expected format: {"0": "25GiB", "1": "40GiB"} or
        # {0: "25GiB", 1: "40GiB"}. Only used when device=="auto".
        max_memory = extra.get("max_memory")

        # Load model: first try the configured device, fall back to auto.
        try:
            device_map = self._device_to_map(device)
            log.info(
                "Loading %s (dtype=%s, device_map=%s, attn=%s, max_memory=%s)",
                model_id, self.torch_dtype, device_map, attn, max_memory,
            )
            load_kwargs = dict(
                torch_dtype=self.torch_dtype,
                device_map=device_map,
                attn_implementation=attn,
                trust_remote_code=trust_remote,
            )
            if device == "auto" and max_memory:
                load_kwargs["max_memory"] = {
                    (int(k) if str(k).isdigit() else k): v for k, v in max_memory.items()
                }
            self.model = model_auto_cls.from_pretrained(model_id, **load_kwargs)
        except (RuntimeError, ValueError, AssertionError) as e:
            if device == "auto":
                raise
            log.warning(
                "Load on device=%r failed (%s). Retrying with device_map='auto'.",
                device, e,
            )
            device_map = "auto" if torch.cuda.is_available() else "cpu"
            self.model = model_auto_cls.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                device_map=device_map,
                attn_implementation=attn,
                trust_remote_code=trust_remote,
            )

        self.device = next(self.model.parameters()).device
        self.model.eval()
        log.info("Model loaded. Device: %s", self.device)

    @staticmethod
    def _pick_auto_class(model_id: str, trust_remote: bool):
        """Inspect the model's auto_map to choose the correct AutoModel class.

        Falls back to AutoModelForImageTextToText (the common case) if the
        model declares no preference.
        """
        try:
            cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote)
            auto_map = getattr(cfg, "auto_map", {}) or {}
            if "AutoModelForImageTextToText" in auto_map:
                return AutoModelForImageTextToText
            if "AutoModelForCausalLM" in auto_map:
                return AutoModelForCausalLM
        except Exception:
            pass
        return AutoModelForImageTextToText

    @staticmethod
    def _device_to_map(device: str):
        """Translate config.device into a valid device_map for from_pretrained."""
        if device == "auto":
            return "auto" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            return "cpu"
        # Validate explicit cuda indices so we fail cleanly on single-GPU hosts.
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("device=%r requested but CUDA unavailable" % device)
            if ":" in device:
                idx = int(device.split(":", 1)[1])
                if idx >= torch.cuda.device_count():
                    raise RuntimeError(
                        f"device={device!r} requested but only "
                        f"{torch.cuda.device_count()} CUDA device(s) visible"
                    )
        return {"": device}

    # ---------------- per-case ----------------

    def create_session(self, system_prompt: str) -> ConversationSession:
        session = ConversationSession()
        session.provider_state = {
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            ],
            "images": [],  # PIL.Image list, attached only in turn 1
        }
        return session

    def load_video(self, video_path: str, session: ConversationSession) -> None:
        """Load frames from session.frames_dir. Video path is ignored for
        image-based models."""
        if not session.frames_dir:
            raise ProviderError(
                f"transformers provider requires pre-extracted frames; "
                f"none found for this case (video_path={video_path})"
            )
        frame_paths = sorted(Path(session.frames_dir).glob("*.jpg"))
        if not frame_paths:
            raise ProviderError(f"No frames in {session.frames_dir}")

        sampled = self._subsample(frame_paths, self.config.frame_count)
        images = [Image.open(p).convert("RGB") for p in sampled]

        # Optional downsample: lets us trade per-frame detail for more frames
        # inside a fixed context budget. See CLAUDE.md "Frame budget per model".
        resize_to = self.config.extra.get("resize_to")
        if resize_to:
            n = int(resize_to)
            images = [img.resize((n, n), Image.LANCZOS) for img in images]
            log.info("Resized %d frames to %dx%d", len(images), n, n)

        session.provider_state["images"] = images
        session.video_loaded = True
        log.info(
            "Loaded %d/%d frames from %s",
            len(images), len(frame_paths), session.frames_dir,
        )

    @staticmethod
    def _subsample(frames: list[Path], n: int | None) -> list[Path]:
        if not n or n >= len(frames):
            return frames
        # Uniform sampling, inclusive of first and last
        step = (len(frames) - 1) / (n - 1) if n > 1 else 0
        idx = [round(i * step) for i in range(n)]
        return [frames[i] for i in idx]

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

        # Build the user turn: images first, then text.
        user_content: list[dict] = []
        for img in images:
            user_content.append({"type": "image", "image": img})
        user_content.append({"type": "text", "text": user_text})
        messages.append({"role": "user", "content": user_content})

        last_error = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                generated_text = self._generate(messages)
                parsed = _try_parse_json(generated_text)
                # Record the assistant reply regardless of parse success —
                # multi-turn context needs it.
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": generated_text}],
                })
                session.messages.append({"role": "user", "text": user_text})
                session.messages.append({"role": "assistant", "text": generated_text})
                # Return the parsed dict when available; otherwise the raw
                # string (scorer.extract_answer handles regex recovery).
                return parsed if parsed is not None else generated_text
            except torch.cuda.OutOfMemoryError as e:
                last_error = e
                log.error("CUDA OOM on attempt %d: %s", attempt, e)
                torch.cuda.empty_cache()
            except Exception as e:  # noqa: BLE001
                last_error = e
                log.warning("Generate attempt %d failed: %s", attempt, e)

            if attempt < self.config.max_retries:
                time.sleep(1.0)

        # Pop the user turn we appended so a failed turn doesn't corrupt history.
        messages.pop()
        raise ProviderError(
            f"transformers generate failed after {self.config.max_retries} attempts: {last_error}"
        )

    def _generate(self, messages: list[dict]) -> str:
        # Qwen3 / Qwen3.5 models default to reasoning mode. Pass enable_thinking
        # through the chat template when configured.
        tpl_kwargs: dict = {}
        enable_thinking = self.config.extra.get("enable_thinking")
        if enable_thinking is not None:
            tpl_kwargs["enable_thinking"] = bool(enable_thinking)

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **tpl_kwargs,
        )
        target_device = next(self.model.parameters()).device
        # Some custom processors (e.g. Hulu-Med) return a BatchEncoding whose
        # .to() doesn't accept a dtype kwarg. Fall back to device-only transfer
        # and cast floating-point tensors (pixel_values, etc.) manually.
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
        new_tokens = out[0][input_len:]
        text = self.processor.decode(new_tokens, skip_special_tokens=True)
        return text.strip()

    def cleanup(self, session: ConversationSession) -> None:
        if session.provider_state:
            # Release frames; keep model loaded across cases.
            for img in session.provider_state.get("images", []):
                try:
                    img.close()
                except Exception:
                    pass
            session.provider_state["images"] = []
            session.provider_state["messages"] = []
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------

_JSON_BRACES_RE = re.compile(r"\{[^{}]*\}")
# Qwen3 emits <think>...</think> before the answer when reasoning is enabled.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _try_parse_json(text: str) -> dict | None:
    """Try to pull the first JSON object out of a model's free-form reply."""
    # Strip any <think>...</think> block so we parse only the final answer.
    text = _THINK_BLOCK_RE.sub("", text).strip()

    # Fast path: whole reply is JSON.
    s = text
    if s.startswith("```"):
        s = s.strip("`")
        s = s.split("\n", 1)[-1] if s.startswith("json") else s
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Fallback: first {...} block.
    m = _JSON_BRACES_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None
