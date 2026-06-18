from __future__ import annotations

import json
import os
import stat
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from xninja.models import DEFAULT_MODEL, OPENROUTER_API_BASE


@dataclass(frozen=True)
class XninjaConfig:
    openrouter_api_key: str = ""
    default_model: str = DEFAULT_MODEL
    # OpenAI-compatible inference endpoint. Defaults to OpenRouter; point it at a
    # managed proxy (e.g. via XNINJA_API_BASE) to route elsewhere — the bundled
    # agent just builds {api_base}/chat/completions with the api key as a Bearer.
    api_base: str = OPENROUTER_API_BASE
    allowed_shell_commands: tuple[str, ...] = field(default_factory=tuple)
    allow_apply_patch: bool = False


def config_dir(env: dict[str, str] | None = None) -> Path:
    values = env or os.environ
    override = values.get("XNINJA_CONFIG_DIR")
    if override:
        return Path(override).expanduser()
    xdg_config_home = values.get("XDG_CONFIG_HOME")
    if xdg_config_home:
        return Path(xdg_config_home).expanduser() / "xninja"
    return Path.home() / ".config" / "xninja"


def config_path(env: dict[str, str] | None = None) -> Path:
    return config_dir(env) / "config.json"


def cache_dir(env: dict[str, str] | None = None) -> Path:
    values = env or os.environ
    override = values.get("XNINJA_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    xdg_cache_home = values.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        return Path(xdg_cache_home).expanduser() / "xninja"
    return Path.home() / ".cache" / "xninja"


def parse_config(raw: dict[str, Any]) -> XninjaConfig:
    return XninjaConfig(
        openrouter_api_key=str(raw.get("openrouter_api_key") or ""),
        default_model=str(raw.get("default_model") or DEFAULT_MODEL),
        api_base=str(raw.get("api_base") or OPENROUTER_API_BASE),
        allowed_shell_commands=tuple(str(item) for item in raw.get("allowed_shell_commands", [])),
        allow_apply_patch=bool(raw.get("allow_apply_patch", False)),
    )


def load_config(path: Path | None = None) -> XninjaConfig:
    target = path or config_path()
    if not target.exists():
        return XninjaConfig()
    return parse_config(json.loads(target.read_text(encoding="utf-8")))


def save_config(config: XninjaConfig, path: Path | None = None) -> Path:
    target = path or config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(asdict(config), indent=2, sort_keys=True) + "\n"
    target.write_text(payload, encoding="utf-8")
    target.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return target


def config_with_env(config: XninjaConfig, env: dict[str, str] | None = None) -> XninjaConfig:
    values = env or os.environ
    return XninjaConfig(
        openrouter_api_key=values.get("OPENROUTER_API_KEY") or config.openrouter_api_key,
        default_model=values.get("XNINJA_MODEL") or config.default_model,
        api_base=values.get("XNINJA_API_BASE") or config.api_base,
        allowed_shell_commands=config.allowed_shell_commands,
        allow_apply_patch=config.allow_apply_patch,
    )


def redact_secret(value: str) -> str:
    if not value:
        return "(not set)"
    if len(value) <= 8:
        return "********"
    return value[:4] + "..." + value[-4:]
