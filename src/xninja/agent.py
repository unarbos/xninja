from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from xninja.config import cache_dir


BUNDLED_AGENT_DIR = Path(__file__).parent / "bundled_agent"
BUNDLED_AGENT_PATH = BUNDLED_AGENT_DIR / "agent.py"
BUNDLED_METADATA_PATH = BUNDLED_AGENT_DIR / "metadata.json"
USER_AGENT_PATH = "agent.py"
USER_METADATA_PATH = "metadata.json"
NINJA_RAW_URL = "https://raw.githubusercontent.com/unarbos/ninja/{ref}/agent.py"


@dataclass(frozen=True)
class AgentSource:
    path: Path
    metadata: dict[str, Any]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def bundled_agent_source() -> AgentSource:
    return AgentSource(path=BUNDLED_AGENT_PATH, metadata=read_json(BUNDLED_METADATA_PATH))


def cached_agent_source(ref: str, env: dict[str, str] | None = None) -> AgentSource | None:
    root = cache_dir(env) / "agents" / ref.replace("/", "_")
    path = root / USER_AGENT_PATH
    if not path.exists():
        return None
    return AgentSource(path=path, metadata=read_json(root / USER_METADATA_PATH))


def resolve_agent_source(ref: str | None, env: dict[str, str] | None = None) -> AgentSource:
    if ref:
        cached = cached_agent_source(ref, env)
        if cached:
            return cached
        return fetch_agent(ref, env)
    return bundled_agent_source()


def load_agent_module(source: AgentSource) -> ModuleType:
    module_name = "xninja_loaded_agent"
    spec = importlib.util.spec_from_file_location(module_name, source.path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load agent from {source.path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_agent(
    source: AgentSource,
    repo_path: Path,
    issue: str,
    model: str,
    api_base: str,
    api_key: str,
) -> dict[str, Any]:
    module = load_agent_module(source)
    solve = getattr(module, "solve", None)
    if not callable(solve):
        raise RuntimeError(f"Agent at {source.path} does not define callable solve(...)")
    result = solve(
        repo_path=str(repo_path),
        issue=issue,
        model=model,
        api_base=api_base,
        api_key=api_key,
    )
    if not isinstance(result, dict):
        raise RuntimeError("Agent solve(...) returned a non-dict result")
    return result


def fetch_agent(ref: str, env: dict[str, str] | None = None) -> AgentSource:
    root = cache_dir(env) / "agents" / ref.replace("/", "_")
    root.mkdir(parents=True, exist_ok=True)
    agent_path = root / USER_AGENT_PATH
    metadata_path = root / USER_METADATA_PATH
    url = NINJA_RAW_URL.format(ref=ref)
    with urllib.request.urlopen(url, timeout=30) as response:
        agent_path.write_bytes(response.read())
    metadata = {"source_repo": "unarbos/ninja", "ref": ref, "path": "agent.py"}
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return AgentSource(path=agent_path, metadata=metadata)


def git_commit_for_ref(ref: str) -> str:
    output = subprocess.check_output(
        ["git", "ls-remote", "https://github.com/unarbos/ninja.git", ref],
        text=True,
        timeout=30,
    )
    first_line = output.strip().splitlines()[0] if output.strip() else ""
    return first_line.split()[0] if first_line else ""


def update_cached_agent(ref: str, env: dict[str, str] | None = None) -> AgentSource:
    source = fetch_agent(ref, env)
    commit = git_commit_for_ref(ref) or source.metadata.get("commit", "")
    metadata = {**source.metadata, "commit": commit}
    metadata_path = source.path.parent / USER_METADATA_PATH
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return AgentSource(path=source.path, metadata=metadata)


def copy_bundled_agent(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BUNDLED_AGENT_PATH, destination)
    return destination
