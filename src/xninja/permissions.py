from __future__ import annotations

from dataclasses import dataclass

from xninja.config import XninjaConfig


@dataclass(frozen=True)
class PermissionDecision:
    allowed: bool
    remember: bool = False


def shell_command_allowed(config: XninjaConfig, command: str) -> bool:
    stripped = command.strip()
    return any(stripped == allowed or stripped.startswith(allowed + " ") for allowed in config.allowed_shell_commands)


def apply_patch_allowed(config: XninjaConfig) -> bool:
    return config.allow_apply_patch


def remember_shell_command(config: XninjaConfig, command_prefix: str) -> XninjaConfig:
    normalized = command_prefix.strip()
    if not normalized or normalized in config.allowed_shell_commands:
        return config
    return XninjaConfig(
        openrouter_api_key=config.openrouter_api_key,
        default_model=config.default_model,
        allowed_shell_commands=tuple((*config.allowed_shell_commands, normalized)),
        allow_apply_patch=config.allow_apply_patch,
    )


def remember_apply_patch(config: XninjaConfig) -> XninjaConfig:
    return XninjaConfig(
        openrouter_api_key=config.openrouter_api_key,
        default_model=config.default_model,
        allowed_shell_commands=config.allowed_shell_commands,
        allow_apply_patch=True,
    )
