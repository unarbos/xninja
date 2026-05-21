from __future__ import annotations

import stat

from xninja.config import (
    XninjaConfig,
    config_path,
    config_with_env,
    load_config,
    redact_secret,
    save_config,
)


def test_config_path_uses_xdg_config_home(tmp_path):
    env = {"XDG_CONFIG_HOME": str(tmp_path)}
    assert config_path(env) == tmp_path / "xninja" / "config.json"


def test_save_and_load_config_with_restrictive_permissions(tmp_path):
    path = tmp_path / "config.json"
    saved = XninjaConfig(openrouter_api_key="sk-test", default_model="model/a")

    save_config(saved, path)

    assert load_config(path) == saved
    assert stat.S_IMODE(path.stat().st_mode) == stat.S_IRUSR | stat.S_IWUSR


def test_env_overrides_secret_and_model():
    config = XninjaConfig(openrouter_api_key="stored", default_model="stored-model")
    env = {"OPENROUTER_API_KEY": "env-key", "XNINJA_MODEL": "env-model"}

    resolved = config_with_env(config, env)

    assert resolved.openrouter_api_key == "env-key"
    assert resolved.default_model == "env-model"
    assert config.openrouter_api_key == "stored"
    assert config.default_model == "stored-model"


def test_config_with_empty_env_preserves_stored_values():
    config = XninjaConfig(openrouter_api_key="stored", default_model="stored-model")

    resolved = config_with_env(config, {})

    assert resolved.openrouter_api_key == "stored"
    assert resolved.default_model == "stored-model"


def test_redact_secret():
    assert redact_secret("") == "(not set)"
    assert redact_secret("short") == "********"
    assert redact_secret("sk-1234567890") == "sk-1...7890"
