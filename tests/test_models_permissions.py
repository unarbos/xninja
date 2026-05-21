from __future__ import annotations

from xninja.config import XninjaConfig
from xninja.models import DEFAULT_MODEL, model_ids, resolve_model
from xninja.permissions import (
    apply_patch_allowed,
    remember_apply_patch,
    remember_shell_command,
    shell_command_allowed,
)


def test_resolve_model_precedence():
    assert resolve_model("flag", "env", "config") == "flag"
    assert resolve_model(None, "env", "config") == "env"
    assert resolve_model(None, None, "config") == "config"
    assert resolve_model(None, None, None) == DEFAULT_MODEL


def test_recommended_models_include_default():
    assert DEFAULT_MODEL in model_ids()


def test_shell_command_allowlist_matches_exact_or_prefix():
    config = XninjaConfig(allowed_shell_commands=("pytest", "git status"))

    assert shell_command_allowed(config, "pytest tests")
    assert shell_command_allowed(config, "git status --short")
    assert not shell_command_allowed(config, "git commit -am nope")


def test_remember_permissions_are_immutable_updates():
    config = XninjaConfig()
    with_shell = remember_shell_command(config, "pytest")
    with_apply = remember_apply_patch(with_shell)

    assert config.allowed_shell_commands == ()
    assert with_shell.allowed_shell_commands == ("pytest",)
    assert apply_patch_allowed(with_apply)
