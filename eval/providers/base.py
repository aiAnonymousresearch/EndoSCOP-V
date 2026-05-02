"""Abstract base class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProviderConfig:
    """Configuration passed to every provider."""

    model_name: str
    temperature: float = 0.0
    max_output_tokens: int = 256
    api_delay: float = 0.0
    max_retries: int = 3
    frame_count: int = 32
    frame_strategy: str = "uniform"
    extra: dict = field(default_factory=dict)


class ConversationSession:
    """Mutable state for a multi-turn conversation with one case."""

    def __init__(self):
        self.messages: list[dict] = []
        self.provider_state: Any = None
        self.video_loaded: bool = False
        self.frames_dir: str | None = None       # pre-extracted frames dir, if any
        self.case_dir: str | None = None


class BaseProvider(ABC):
    """Abstract base for all LLM providers."""

    def __init__(self, config: ProviderConfig):
        self.config = config

    @abstractmethod
    def initialize(self) -> None:
        """One-time setup: authenticate, load model, etc."""

    @abstractmethod
    def create_session(self, system_prompt: str) -> ConversationSession:
        """Create a new conversation session with the given system prompt."""

    @abstractmethod
    def load_video(self, video_path: str, session: ConversationSession) -> None:
        """Load/upload video or extract frames. Store in session.provider_state."""

    @abstractmethod
    def send_turn(
        self,
        session: ConversationSession,
        user_text: str,
        answer_schema: dict,
        is_first_turn: bool,
    ) -> dict:
        """
        Send one turn in the conversation. Returns parsed JSON response.

        The provider is responsible for:
        - Attaching video/frames on first turn
        - Enforcing structured output via its native mechanism
        - Appending to conversation history in session
        - Returning the parsed answer dict (e.g. {"answer": "C"})

        Raises ProviderError on failure after retries.
        """

    def cleanup(self, session: ConversationSession) -> None:
        """Optional per-case cleanup (delete uploaded files, etc.)."""


class ProviderError(Exception):
    """Raised when a provider fails after retries."""
