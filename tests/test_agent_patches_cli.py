from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from xninja.agent import AgentSource, bundled_agent_source, load_agent_module, local_agent_source, run_agent
from xninja.cli import (
    brand,
    colorize_patch,
    build_parser,
    build_prompt_parser,
    color_enabled,
    commit_agent_baseline,
    copy_repo_for_agent,
    main,
    footer_hint,
    fmt_elapsed_compact,
    meta,
    parse_args,
    printable_agent_logs,
    select_agent_source,
    stream_agent_logs_enabled,
    style,
    working_status,
)
from xninja.patches import apply_patch, patch_text, repo_is_git_worktree


def init_repo(path: Path) -> None:
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    (path / "hello.txt").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "add", "hello.txt"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=path, check=True, capture_output=True)


def test_bundled_agent_metadata_loads():
    source = bundled_agent_source()

    assert source.path.exists()
    assert source.metadata["source_repo"] == "unarbos/ninja"
    assert source.metadata["commit"]


def test_bundled_agent_streaming_helpers_load():
    loaded = load_agent_module(bundled_agent_source())

    assert callable(loaded.__dict__.get("_new_logs"))
    assert callable(loaded.__dict__.get("_render_log_item"))


def test_apply_patch_changes_temp_repo(tmp_path):
    init_repo(tmp_path)
    patch = """diff --git a/hello.txt b/hello.txt
index ce01362..cc628cc 100644
--- a/hello.txt
+++ b/hello.txt
@@ -1 +1 @@
-hello
+hello ninja
"""

    result = apply_patch(tmp_path, patch)

    assert result.returncode == 0
    assert (tmp_path / "hello.txt").read_text(encoding="utf-8") == "hello ninja\n"


def test_repo_is_git_worktree(tmp_path):
    assert not repo_is_git_worktree(tmp_path)
    init_repo(tmp_path)
    assert repo_is_git_worktree(tmp_path)


def test_copy_repo_for_agent_isolates_target_repo(tmp_path):
    repo = tmp_path / "repo"
    work_root = tmp_path / "work"
    repo.mkdir()
    work_root.mkdir()
    init_repo(repo)

    copied = copy_repo_for_agent(repo, work_root)
    (copied / "hello.txt").write_text("changed in copy\n", encoding="utf-8")

    assert (repo / "hello.txt").read_text(encoding="utf-8") == "hello\n"
    assert repo_is_git_worktree(copied)


def test_dataclass_agent_module_loads(tmp_path):
    fake_agent = tmp_path / "dataclass_agent.py"
    fake_agent.write_text(
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class Thing:\n"
        "    value: str\n"
        "def solve(repo_path, issue, model, api_base, api_key):\n"
        "    return {\"patch\": Thing(\"ok\").value, \"logs\": \"\", \"steps\": 1, \"cost\": None, \"success\": True}\n",
        encoding="utf-8",
    )

    result = run_agent(AgentSource(fake_agent, {"ref": "dataclass"}), tmp_path, "task", "model", "base", "key")

    assert result["patch"] == "ok"


def test_run_agent_wraps_exception_from_solve(tmp_path):
    fake_agent = tmp_path / "failing_agent.py"
    fake_agent.write_text(
        "def solve(repo_path, issue, model, api_base, api_key):\n"
        "    raise ValueError('boom')\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match=r"Agent solve\(\.\.\.\) raised an exception: boom"):
        run_agent(AgentSource(fake_agent, {"ref": "failing"}), tmp_path, "task", "model", "base", "key")


def test_fake_agent_smoke(tmp_path):
    init_repo(tmp_path)
    fake_agent = tmp_path / "fake_agent.py"
    fake_agent.write_text(
        """
def solve(repo_path, issue, model, api_base, api_key):
    return {
        "patch": "diff --git a/hello.txt b/hello.txt\\nindex ce01362..cc628cc 100644\\n--- a/hello.txt\\n+++ b/hello.txt\\n@@ -1 +1 @@\\n-hello\\n+hello fake\\n",
        "logs": f"{issue}:{model}:{api_base}:{bool(api_key)}",
        "steps": 1,
        "cost": None,
        "success": True,
    }
""",
        encoding="utf-8",
    )

    result = run_agent(
        AgentSource(fake_agent, {"ref": "fake"}),
        tmp_path,
        "task",
        "model",
        "base",
        "key",
    )

    assert "hello fake" in patch_text(result)


def test_style_respects_no_color(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")

    assert style("hello", "green") == "hello"
    assert meta("model", "test") == "model: test"


def test_style_adds_ansi_when_enabled(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")

    assert style("hello", "green").startswith("\033[32m")


def test_stream_agent_logs_enabled_for_bundled_source(tmp_path):
    bundled = tmp_path / "bundled_agent" / "agent.py"
    cached = tmp_path / "cached" / "agent.py"

    assert stream_agent_logs_enabled(AgentSource(bundled, {}))
    assert not stream_agent_logs_enabled(AgentSource(cached, {}))


def test_printable_agent_logs_formats_none_and_text():
    assert printable_agent_logs(None) == ""
    assert printable_agent_logs("\nstep 1\n") == "step 1"


def test_cli_accepts_raw_logs_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["run", "--raw-logs", "--help"])
    captured = capsys.readouterr()

    assert exc.value.code == 0
    assert "--raw-logs" in captured.out


def test_cli_help_smoke(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    captured = capsys.readouterr()

    assert exc.value.code == 0
    assert "Run the ninja coding agent locally" in captured.out


def test_codex_like_parser_aliases():
    args = build_parser().parse_args(["exec", "-C", ".", "-m", "model/a", "--color", "never", "fix it"])

    assert args.command == "exec"
    assert args.repo == "."
    assert args.model == "model/a"
    assert args.color == "never"
    assert args.task == ["fix it"]


def test_color_mode_override(monkeypatch):
    monkeypatch.setenv("TERM", "dumb")

    assert color_enabled("always")
    assert not color_enabled("never")


def test_parse_args_routes_prompt_and_commands():
    prompt = parse_args(["--color", "never", "hello", "there"])
    agent = parse_args(["agent", "info"])
    run = parse_args(["exec", "-C", ".", "fix", "it"])

    assert prompt.command is None
    assert prompt.prompt == ["hello", "there"]
    assert agent.command == "agent"
    assert agent.agent_command == "info"
    assert run.command == "exec"
    assert run.task == ["fix", "it"]


def test_prompt_parser_accepts_one_shot_text():
    args = build_prompt_parser().parse_args(["-m", "model/a", "fix", "it"])

    assert args.model == "model/a"
    assert args.prompt == ["fix", "it"]


def test_parse_args_without_prompt_enters_interactive_shape():
    args = parse_args([])

    assert args.command is None
    assert not getattr(args, "prompt", [])
    assert args.repo == "."


def test_codex_style_helpers_are_compact(monkeypatch):
    monkeypatch.setenv("XNINJA_COLOR", "never")

    assert brand() == "xninja"
    assert fmt_elapsed_compact(0) == "0s"
    assert fmt_elapsed_compact(61) == "1m 01s"
    assert fmt_elapsed_compact(3661) == "1h 01m 01s"
    assert working_status() == "Working (Ctrl-C to interrupt)"
    assert working_status(61) == "Working (1m 01s • Ctrl-C to interrupt)"
    assert footer_hint("enter to send", "Ctrl-D to exit") == "enter to send · Ctrl-D to exit"
    assert colorize_patch("+added\n-removed") == "+added\n-removed"


def test_color_max_helpers_add_ansi_when_enabled(monkeypatch):
    monkeypatch.setenv("XNINJA_COLOR", "always")

    assert "\033[" in meta("model", "test")
    assert "\033[" in colorize_patch("+added\n-removed")


def test_failed_patch_apply_message_is_clear(monkeypatch, tmp_path, capsys):
    init_repo(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")

    def fake_resolve_agent_source(ref):
        return AgentSource(tmp_path / "bundled_agent" / "agent.py", {"source_repo": "fake", "ref": "fake"})

    def fake_run_agent(source, repo_path, task, model, api_base, api_key):
        return {
            "patch": "diff --git a/missing.txt b/missing.txt\n--- a/missing.txt\n+++ b/missing.txt\n@@ -1 +1 @@\n-old\n+new\n",
            "logs": "",
            "steps": 1,
            "cost": None,
            "success": False,
        }

    monkeypatch.setattr("xninja.cli.resolve_agent_source", fake_resolve_agent_source)
    monkeypatch.setattr("xninja.cli.run_agent", fake_run_agent)

    code = main(["--repo", str(tmp_path), "--apply", "fix missing file"])
    captured = capsys.readouterr()

    assert code != 0
    assert "Patch did not apply cleanly" in captured.err


def test_cli_version_smoke(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    captured = capsys.readouterr()

    assert exc.value.code == 0
    assert captured.out.strip().startswith("xninja ")


def test_local_agent_source_loads_path(tmp_path):
    agent = tmp_path / "agent.py"
    agent.write_text(
        "def solve(repo_path, issue, model, api_base, api_key): "
        "return {'patch': '', 'logs': '', 'steps': 0, 'cost': None, 'success': True}\n",
        encoding="utf-8",
    )

    source = local_agent_source(agent)

    assert source.path == agent.resolve()
    assert source.metadata["source_repo"] == "local"


def test_local_agent_source_rejects_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        local_agent_source(tmp_path / "missing.py")


def test_parse_args_accepts_agent_path(tmp_path):
    agent = tmp_path / "agent.py"
    args = parse_args(["--agent-path", str(agent), "fix", "it"])

    assert args.agent_path == str(agent)
    assert args.prompt == ["fix", "it"]


def test_select_agent_source_prefers_agent_path(tmp_path):
    agent = tmp_path / "agent.py"
    agent.write_text("def solve(repo_path, issue, model, api_base, api_key): return {}\n", encoding="utf-8")

    source = select_agent_source("main", str(agent))

    assert source.path == agent.resolve()
    assert source.metadata["source_repo"] == "local"


def test_custom_agent_path_smoke(tmp_path, monkeypatch):
    init_repo(tmp_path)
    agent = tmp_path / "custom_agent.py"
    agent.write_text(
        "def solve(repo_path, issue, model, api_base, api_key):\n"
        "    return {\"patch\": \"\", \"logs\": f'{issue}:{model}:{bool(api_key)}', \"steps\": 1, \"cost\": None, \"success\": True}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")

    code = main(["--repo", str(tmp_path), "--agent-path", str(agent), "custom task"])

    assert code == 1
