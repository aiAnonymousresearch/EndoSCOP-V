"""Gemini provider — google-genai SDK, supports both AI Studio API key and Vertex AI.

Auth selection order:
  1. GOOGLE_API_KEY env var (or `gemini_api_key_file` in config) → AI Studio.
  2. GOOGLE_APPLICATION_CREDENTIALS / Application Default Credentials
     plus GOOGLE_CLOUD_PROJECT → Vertex AI.

AI Studio is simpler and has free tier; Vertex AI is the enterprise path.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from .base import BaseProvider, ConversationSession, ProviderConfig, ProviderError
from . import register

log = logging.getLogger("eval.gemini")


def _resolve_api_key(extra: dict) -> str | None:
    env_var = extra.get("api_key_env", "GOOGLE_API_KEY")
    key = os.environ.get(env_var)
    if key:
        return key
    file_path = extra.get("api_key_file")
    if file_path and Path(file_path).is_file():
        return Path(file_path).read_text().strip()
    return None


@register("gemini")
class GeminiProvider(BaseProvider):
    """Google Gemini with native mp4 input via google-genai SDK."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._client = None
        self._types = None

    def initialize(self) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise ProviderError(
                "google-genai not installed. Run: pip install google-genai"
            ) from e

        self._types = types
        api_key = _resolve_api_key(self.config.extra)
        if api_key:
            self._client = genai.Client(api_key=api_key)
            log.info(
                "Gemini AI Studio client initialized (api_key): model=%s",
                self.config.model_name,
            )
            return

        # Fall back to Vertex AI via Application Default Credentials.
        # Project resolution order: explicit config > GOOGLE_CLOUD_PROJECT env >
        # project_id auto-detected from the service-account JSON pointed at
        # by GOOGLE_APPLICATION_CREDENTIALS.
        location = self.config.extra.get("vertex_location", "global")
        project = (
            self.config.extra.get("vertex_project")
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
        )
        if not project:
            try:
                from google.auth import default as _adc_default
                _, project = _adc_default()
            except Exception as e:
                raise ProviderError(
                    "Gemini: set GOOGLE_API_KEY (Google AI Studio) or "
                    "GOOGLE_APPLICATION_CREDENTIALS pointing to a "
                    "service-account JSON, or set GOOGLE_CLOUD_PROJECT."
                ) from e
            if not project:
                raise ProviderError(
                    "Gemini: ADC succeeded but no project_id was found. "
                    "Set GOOGLE_CLOUD_PROJECT or vertex_project in config."
                )
        self._client = genai.Client(vertexai=True, project=project, location=location)
        log.info(
            "Gemini Vertex AI client initialized: project=%s, location=%s, model=%s",
            project, location, self.config.model_name,
        )

    def create_session(self, system_prompt: str) -> ConversationSession:
        session = ConversationSession()
        session.provider_state = {
            "system_prompt": system_prompt,
            "video_part": None,
            "history": [],   # list of types.Content
        }
        return session

    def load_video(self, video_path: str, session: ConversationSession) -> None:
        with open(video_path, "rb") as f:
            video_bytes = f.read()
        session.provider_state["video_part"] = self._types.Part.from_bytes(
            data=video_bytes, mime_type="video/mp4"
        )
        session.video_loaded = True
        size_mb = len(video_bytes) / (1024 * 1024)
        log.info("Loaded video: %s (%.1f MB)", video_path, size_mb)

    def send_turn(
        self,
        session: ConversationSession,
        user_text: str,
        answer_schema: dict,
        is_first_turn: bool,
    ) -> dict | str:
        state = session.provider_state
        types = self._types

        if is_first_turn:
            user_parts = [state["video_part"], types.Part.from_text(text=user_text)]
        else:
            user_parts = [types.Part.from_text(text=user_text)]

        history = state["history"]
        history.append(types.Content(role="user", parts=user_parts))

        cfg_kwargs = dict(
            system_instruction=state["system_prompt"],
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_output_tokens,
        )
        if answer_schema:
            cfg_kwargs["response_mime_type"] = "application/json"
            cfg_kwargs["response_schema"] = answer_schema
        gen_config = types.GenerateContentConfig(**cfg_kwargs)

        last_error = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=self.config.model_name,
                    contents=history,
                    config=gen_config,
                )
                text = response.text or ""
                history.append(types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=text)],
                ))
                session.messages.append({"role": "user", "text": user_text})
                session.messages.append({"role": "assistant", "text": text})

                if answer_schema:
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        return text
                return text

            except Exception as e:  # noqa: BLE001
                last_error = e
                log.warning("Gemini attempt %d failed: %s", attempt, e)
                if history and history[-1].role == "user":
                    history.pop()
                if attempt < self.config.max_retries:
                    wait = 2 ** attempt
                    log.info("Retrying in %ds...", wait)
                    time.sleep(wait)
                    history.append(types.Content(role="user", parts=user_parts))

        raise ProviderError(
            f"Gemini failed after {self.config.max_retries} attempts: {last_error}"
        )

    def cleanup(self, session: ConversationSession) -> None:
        if session.provider_state:
            session.provider_state["video_part"] = None
            session.provider_state["history"] = []
