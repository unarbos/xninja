from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from xninja.config import cache_dir

BUNDLED_AGENT_DIR = Path(__file__).parent / "bundled_agent"
BUNDLED_AGENT_PATH = BUNDLED_AGENT_DIR / "agent.py"
BUNDLED_METADATA_PATH = BUNDLED_AGENT_DIR / "metadata.json"
ENTRYPOINT = "agent.py"
MANIFEST_FILENAME = "tau_agent_files.json"
USER_METADATA_PATH = "metadata.json"
NINJA_RAW_URL = "https://raw.githubusercontent.com/unarbos/ninja/{ref}/{path}"

# Validator-side bundle limits (ninja MINER_SUBMISSION_CHECKLIST): a submitted
# harness is agent.py plus its supporting modules, at most 32 *.py files and
# 5 MB total. We enforce the same bounds when fetching a remote bundle so an
# untrusted manifest cannot make xninja download an unbounded file set.
MAX_AGENT_FILES = 32
MAX_TOTAL_BYTES = 5_000_000

# Module name used when loading the entrypoint by file path. Mirrors the name
# the validator harness runner uses so agents that introspect __name__ behave
# identically here.
_LOADED_MODULE_NAME = "submitted_agent"


@dataclass(frozen=True)
class AgentSource:
    path: Path
    metadata: dict[str, Any]
    # Bundle root placed on sys.path before the entrypoint is imported, so that
    # `from agent.foo import ...` inside a multi-file bundle resolves. Defaults
    # to the entrypoint's directory for single-file agents.
    root: Path | None = None

    @property
    def bundle_root(self) -> Path:
        return self.root or self.path.parent


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_manifest(relative_paths: object) -> list[str]:
    """Validate a tau_agent_files.json manifest and return clean relative paths.

    Mirrors the validator's submission rules: a JSON array of clean relative
    POSIX *.py paths that includes the agent.py entrypoint, capped at 32 files.
    """
    if not isinstance(relative_paths, list) or not all(isinstance(p, str) for p in relative_paths):
        raise ValueError(f"{MANIFEST_FILENAME} must be a JSON array of relative file paths")
    if ENTRYPOINT not in relative_paths:
        raise ValueError(f"{MANIFEST_FILENAME} must list `{ENTRYPOINT}` as the entrypoint")
    if len(relative_paths) > MAX_AGENT_FILES:
        raise ValueError(f"bundle lists {len(relative_paths)} files; the maximum is {MAX_AGENT_FILES}")
    for path in relative_paths:
        if path.startswith("/") or "\\" in path or any(seg in {"", ".", ".."} for seg in path.split("/")):
            raise ValueError(f"agent file path `{path}` must be a clean relative POSIX path")
        if not path.endswith(".py"):
            raise ValueError(f"agent file `{path}` must be a Python module")
    return sorted(relative_paths)


def manifest_files(root: Path) -> list[str]:
    """Return the bundle's file list from its manifest, or just the entrypoint."""
    manifest_path = root / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return [ENTRYPOINT]
    return _validate_manifest(json.loads(manifest_path.read_text(encoding="utf-8")))


def bundled_agent_source() -> AgentSource:
    return AgentSource(
        path=BUNDLED_AGENT_PATH,
        metadata=read_json(BUNDLED_METADATA_PATH),
        root=BUNDLED_AGENT_DIR,
    )


def local_agent_source(path: str | Path) -> AgentSource:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"agent path does not exist: {resolved}")
    if resolved.is_dir():
        root = resolved
        entrypoint = root / ENTRYPOINT
        if not entrypoint.is_file():
            raise FileNotFoundError(f"bundle directory has no {ENTRYPOINT}: {root}")
    elif resolved.is_file():
        root = resolved.parent
        entrypoint = resolved
    else:
        raise ValueError(f"agent path is neither a file nor a directory: {resolved}")
    return AgentSource(
        path=entrypoint,
        metadata={"source_repo": "local", "ref": "local", "path": str(entrypoint)},
        root=root,
    )


def cached_agent_source(ref: str, env: dict[str, str] | None = None) -> AgentSource | None:
    root = cache_dir(env) / "agents" / ref.replace("/", "_")
    path = root / ENTRYPOINT
    if not path.exists():
        return None
    return AgentSource(path=path, metadata=read_json(root / USER_METADATA_PATH), root=root)


def resolve_agent_source(ref: str | None, env: dict[str, str] | None = None) -> AgentSource:
    if ref:
        cached = cached_agent_source(ref, env)
        if cached:
            return cached
        return fetch_agent(ref, env)
    return bundled_agent_source()


def load_agent_module(source: AgentSource) -> ModuleType:
    """Import the bundle entrypoint with its root on sys.path.

    Matches the validator harness runner: the bundle root goes on sys.path so a
    multi-file bundle's `agent` package imports resolve, and agent.py is loaded
    by file path. Stale `agent`/`agent.*` modules from a previous load are
    purged first so the active bundle's package is the one that imports.
    """
    root = str(source.bundle_root)
    for name in [n for n in sys.modules if n == "agent" or n.startswith("agent.") or n == _LOADED_MODULE_NAME]:
        del sys.modules[name]
    while root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    importlib.invalidate_caches()

    spec = importlib.util.spec_from_file_location(_LOADED_MODULE_NAME, source.path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load agent from {source.path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_LOADED_MODULE_NAME] = module
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
    try:
        result = solve(
            repo_path=str(repo_path),
            issue=issue,
            model=model,
            api_base=api_base,
            api_key=api_key,
        )
    except Exception as exc:
        raise RuntimeError(f"Agent solve(...) raised an exception: {exc}") from exc
    if not isinstance(result, dict):
        raise RuntimeError("Agent solve(...) returned a non-dict result")
    return result


def _fetch_text(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read()


def _fetch_manifest(ref: str) -> list[str] | None:
    """Fetch and validate the bundle manifest for a ref, or None if absent."""
    url = NINJA_RAW_URL.format(ref=ref, path=MANIFEST_FILENAME)
    try:
        raw = _fetch_text(url)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    return _validate_manifest(json.loads(raw.decode("utf-8")))


def fetch_agent(ref: str, env: dict[str, str] | None = None) -> AgentSource:
    """Download a ninja bundle at `ref` into the cache.

    Reads the manifest to fetch every bundle file (preserving the `agent/`
    package layout). Refs predating the multi-file format have no manifest, so
    we fall back to fetching the single `agent.py`.
    """
    root = cache_dir(env) / "agents" / ref.replace("/", "_")
    files = _fetch_manifest(ref) or [ENTRYPOINT]

    total = 0
    for relative in files:
        dest = root / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        content = _fetch_text(NINJA_RAW_URL.format(ref=ref, path=relative))
        total += len(content)
        if total > MAX_TOTAL_BYTES:
            raise ValueError(f"bundle exceeds {MAX_TOTAL_BYTES} bytes")
        dest.write_bytes(content)

    if len(files) > 1:
        (root / MANIFEST_FILENAME).write_text(
            json.dumps(files, indent=2) + "\n", encoding="utf-8"
        )
    metadata = {"source_repo": "unarbos/ninja", "ref": ref, "path": ENTRYPOINT, "files": files}
    (root / USER_METADATA_PATH).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return AgentSource(path=root / ENTRYPOINT, metadata=metadata, root=root)


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
    (source.bundle_root / USER_METADATA_PATH).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return AgentSource(path=source.path, metadata=metadata, root=source.bundle_root)


def copy_bundled_agent(destination: Path) -> Path:
    """Copy the full bundled harness into `destination` (a bundle directory)."""
    destination.mkdir(parents=True, exist_ok=True)
    for relative in manifest_files(BUNDLED_AGENT_DIR):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BUNDLED_AGENT_DIR / relative, target)
    shutil.copy2(BUNDLED_METADATA_PATH, destination / USER_METADATA_PATH)
    shutil.copy2(BUNDLED_AGENT_DIR / MANIFEST_FILENAME, destination / MANIFEST_FILENAME)
    return destination
