"""OpenAI provider — frames as base64 image_url, structured output via response_format."""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from pathlib import Path

from .base import BaseProvider, ConversationSession, ProviderConfig, ProviderError
from . import register

log = logging.getLogger("eval.openai")


def _read_api_key(extra: dict) -> str:
    """Resolve OPENAI_API_KEY from env-var name, then file path, then literal."""
    env_var = extra.get("api_key_env", "OPENAI_API_KEY")
    key = os.environ.get(env_var)
    if key:
        return key
    file_path = extra.get("api_key_file")
    if file_path and Path(file_path).is_file():
        return Path(file_path).read_text().strip()
    raise ProviderError(
        f"OpenAI API key not found. Set ${env_var} or configure "
        f"openai.api_key_file in config.yaml."
    )


def _uniform_indices(n_total: int, n_sample: int) -> list[int]:
    """Pick `n_sample` uniformly spaced indices from [0, n_total)."""
    if n_total <= n_sample:
        return list(range(n_total))
    step = (n_total - 1) / (n_sample - 1)
    return [round(i * step) for i in range(n_sample)]


def _load_frames_b64(frames_dir: Path, n_sample: int) -> list[str]:
    """Read up to n_sample uniformly-sampled JPEGs and base64-encode them."""
    frames = sorted(frames_dir.glob("*.jpg"))
    if not frames:
        raise ProviderError(f"No .jpg frames found in {frames_dir}")
    indices = _uniform_indices(len(frames), n_sample)
    out: list[str] = []
    for idx in indices:
        with open(frames[idx], "rb") as f:
            out.append(base64.b64encode(f.read()).decode("ascii"))
    return out


@register("openai")
class OpenAIProvider(BaseProvider):
    """OpenAI Chat Completions API with vision input."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._client = None

    def initialize(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ProviderError(
                "openai package not installed. Run: pip install openai"
            ) from e
        api_key = _read_api_key(self.config.extra)
        self._client = OpenAI(api_key=api_key)
        log.info("OpenAI client initialized: model=%s", self.config.model_name)

    def create_session(self, system_prompt: str) -> ConversationSession:
        session = ConversationSession()
        session.provider_state = {
            "messages": [{"role": "system", "content": system_prompt}],
            "frames_b64": [],
        }
        return session

    def load_video(self, video_path: str, session: ConversationSession) -> None:
        if not session.frames_dir:
            raise ProviderError(
                f"OpenAI provider needs pre-extracted frames; "
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
            content: list[dict] = [{"type": "text", "text": user_text}]
            for b64 in state["frames_b64"]:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })
            state["messages"].append({"role": "user", "content": content})
        else:
            state["messages"].append({"role": "user", "content": user_text})

        kwargs = {
            "model": self.config.model_name,
            "messages": state["messages"],
            "temperature": self.config.temperature,
            "max_completion_tokens": self.config.max_output_tokens,
        }
        if answer_schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": answer_schema,
                    "strict": False,
                },
            }

        last_error = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                content_text = resp.choices[0].message.content or ""
                state["messages"].append(
                    {"role": "assistant", "content": content_text}
                )
                session.messages.append({"role": "user", "text": user_text})
                session.messages.append({"role": "assistant", "text": content_text})
                if answer_schema:
                    try:
                        return json.loads(content_text)
                    except json.JSONDecodeError:
                        return content_text
                return content_text

            except Exception as e:  # noqa: BLE001
                last_error = e
                log.warning("OpenAI attempt %d failed: %s", attempt, e)
                if attempt < self.config.max_retries:
                    wait = 2 ** attempt
                    log.info("Retrying in %ds...", wait)
                    time.sleep(wait)
                # Roll back the unanswered user turn so retries don't append twice
                if state["messages"][-1]["role"] == "user":
                    state["messages"].pop()
                if attempt < self.config.max_retries:
                    # Re-append for next attempt
                    if is_first_turn:
                        state["messages"].append({"role": "user", "content": content})
                    else:
                        state["messages"].append({"role": "user", "content": user_text})

        raise ProviderError(
            f"OpenAI failed after {self.config.max_retries} attempts: {last_error}"
        )

    def cleanup(self, session: ConversationSession) -> None:
        if session.provider_state:
            session.provider_state["frames_b64"] = []
