from __future__ import annotations

from dataclasses import dataclass


DEFAULT_MODEL = "anthropic/claude-sonnet-4.6"
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class ModelChoice:
    model_id: str
    label: str
    note: str


RECOMMENDED_MODELS = (
    ModelChoice("anthropic/claude-sonnet-4.6", "Claude Sonnet 4.6", "balanced coding default"),
    ModelChoice("anthropic/claude-opus-4.7", "Claude Opus 4.7", "stronger, usually slower"),
    ModelChoice("moonshotai/kimi-k2.6", "Kimi K2.6", "useful fallback route"),
    ModelChoice("openai/gpt-5.2", "GPT-5.2", "general coding model"),
)


def model_ids(models: tuple[ModelChoice, ...] = RECOMMENDED_MODELS) -> tuple[str, ...]:
    return tuple(model.model_id for model in models)


def resolve_model(explicit_model: str | None, env_model: str | None, config_model: str | None) -> str:
    return explicit_model or env_model or config_model or DEFAULT_MODEL
