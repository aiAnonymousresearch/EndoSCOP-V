"""Anthropic provider — frames as base64 image blocks, structured output via tool_use."""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from pathlib import Path

from .base import BaseProvider, ConversationSession, ProviderConfig, ProviderError
from . import register

log = logging.getLogger("eval.anthropic")


def _read_api_key(extra: dict) -> str:
    env_var = extra.get("api_key_env", "ANTHROPIC_API_KEY")
    key = os.environ.get(env_var)
    if key:
        return key
    file_path = extra.get("api_key_file")
    if file_path and Path(file_path).is_file():
        return Path(file_path).read_text().strip()
    raise ProviderError(
        f"Anthropic API key not found. Set ${env_var} or configure "
        f"anthropic.api_key_file in config.yaml."
    )


def _uniform_indices(n_total: int, n_sample: int) -> list[int]:
    if n_total <= n_sample:
        return list(range(n_total))
    step = (n_total - 1) / (n_sample - 1)
    return [round(i * step) for i in range(n_sample)]


def _load_frames_b64(frames_dir: Path, n_sample: int) -> list[str]:
    frames = sorted(frames_dir.glob("*.jpg"))
    if not frames:
        raise ProviderError(f"No .jpg frames found in {frames_dir}")
    indices = _uniform_indices(len(frames), n_sample)
    out: list[str] = []
    for idx in indices:
        with open(frames[idx], "rb") as f:
            out.append(base64.b64encode(f.read()).decode("ascii"))
    return out


@register("anthropic")
class AnthropicProvider(BaseProvider):
    """Anthropic Messages API with vision and forced tool_use for structured output."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._client = None

    def initialize(self) -> None:
        try:
            import anthropic
        except ImportError as e:
            raise ProviderError(
                "anthropic package not installed. Run: pip install anthropic"
            ) from e
        api_key = _read_api_key(self.config.extra)
        self._client = anthropic.Anthropic(api_key=api_key)
        log.info("Anthropic client initialized: model=%s", self.config.model_name)

    def create_session(self, system_prompt: str) -> ConversationSession:
        session = ConversationSession()
        session.provider_state = {
            "system": system_prompt,
            "messages": [],
            "frames_b64": [],
        }
        return session

    def load_video(self, video_path: str, session: ConversationSession) -> None:
        if not session.frames_dir:
            raise ProviderError(
                f"Anthropic provider needs pre-extracted frames; "
                f"no _siglip_896 dir found alongside {video_path}"
            )
        n = self.config.frame_count or 50
        session.provider_state["frames_b64"] = _load_frames_b64(
            Path(session.frames_dir), n
        )
        session.video_loaded = True
        log.info(
            "Loaded %d frames (b64) from %s",
            len(session.provider_state["frames_b64"]), session.frames_dir,
        )

    def send_turn(
        self,
        session: ConversationSession,
        user_text: str,
        answer_schema: dict,
        is_first_turn: bool,
    ) -> dict | str:
        state = session.provider_state

        if is_first_turn:
            content: list[dict] = []
            for b64 in state["frames_b64"]:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": b64,
                    },
                })
            content.append({"type": "text", "text": user_text})
            state["messages"].append({"role": "user", "content": content})
        else:
            state["messages"].append(
                {"role": "user", "content": [{"type": "text", "text": user_text}]}
            )

        kwargs = {
            "model": self.config.model_name,
            "system": state["system"],
            "messages": state["messages"],
            "max_tokens": self.config.max_output_tokens,
            "temperature": self.config.temperature,
        }
        if answer_schema:
            kwargs["tools"] = [{
                "name": "submit_answer",
                "description": "Submit the final answer in the required schema.",
                "input_schema": answer_schema,
            }]
            kwargs["tool_choice"] = {"type": "tool", "name": "submit_answer"}

        last_error = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                resp = self._client.messages.create(**kwargs)
                # Extract structured answer from tool_use block, else text
                parsed: dict | str | None = None
                text_parts: list[str] = []
                for block in resp.content:
                    btype = getattr(block, "type", None)
                    if btype == "tool_use":
                        parsed = block.input
                    elif btype == "text":
                        text_parts.append(block.text)
                raw_text = "\n".join(text_parts)

                # Append assistant turn back to history so multi-turn context flows
                state["messages"].append(
                    {"role": "assistant", "content": [
                        {"type": "text",
                         "text": json.dumps(parsed) if parsed is not None else raw_text}
                    ]}
                )
                session.messages.append({"role": "user", "text": user_text})
                session.messages.append(
                    {"role": "assistant",
                     "text": json.dumps(parsed) if parsed is not None else raw_text}
                )

                if parsed is not None:
                    return parsed
                return raw_text

            except Exception as e:  # noqa: BLE001
                last_error = e
                log.warning("Anthropic attempt %d failed: %s", attempt, e)
                # Roll back the user turn so retries don't double-append
                if state["messages"] and state["messages"][-1]["role"] == "user":
                    state["messages"].pop()
                if attempt < self.config.max_retries:
                    wait = 2 ** attempt
                    log.info("Retrying in %ds...", wait)
                    time.sleep(wait)
                    # Re-append for the next attempt
                    if is_first_turn:
                        state["messages"].append({"role": "user", "content": content})
                    else:
                        state["messages"].append(
                            {"role": "user",
                             "content": [{"type": "text", "text": user_text}]}
                        )

        raise ProviderError(
            f"Anthropic failed after {self.config.max_retries} attempts: {last_error}"
        )

    def cleanup(self, session: ConversationSession) -> None:
        if session.provider_state:
            session.provider_state["frames_b64"] = []
