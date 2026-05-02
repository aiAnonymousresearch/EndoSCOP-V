"""Provider registry and factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseProvider, ProviderConfig

PROVIDER_REGISTRY: dict[str, type[BaseProvider]] = {}


def register(name: str):
    """Decorator to register a provider class under a short name."""
    def decorator(cls):
        PROVIDER_REGISTRY[name] = cls
        return cls
    return decorator


def create_provider(provider_name: str, config: ProviderConfig) -> BaseProvider:
    """Instantiate and initialize a provider by name."""
    # Lazy-import provider modules so they register themselves
    if provider_name == "gemini":
        from . import gemini  # noqa: F401
    elif provider_name == "openai":
        from . import openai_provider  # noqa: F401
    elif provider_name == "anthropic":
        from . import anthropic_provider  # noqa: F401
    elif provider_name == "transformers":
        from . import transformers_provider  # noqa: F401
    elif provider_name == "hulumed":
        # HuluMedProvider subclasses TransformersProvider; load both.
        from . import transformers_provider  # noqa: F401
        from . import hulumed_provider  # noqa: F401
    elif provider_name == "colonr1":
        # ColonR1Provider subclasses TransformersProvider; load both.
        from . import transformers_provider  # noqa: F401
        from . import colonr1_provider  # noqa: F401

    if provider_name not in PROVIDER_REGISTRY:
        available = ", ".join(PROVIDER_REGISTRY) or "(none loaded)"
        raise KeyError(
            f"Unknown provider '{provider_name}'. Available: {available}"
        )

    cls = PROVIDER_REGISTRY[provider_name]
    provider = cls(config)
    provider.initialize()
    return provider
