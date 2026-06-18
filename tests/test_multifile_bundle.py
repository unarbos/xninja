from __future__ import annotations

import json

import pytest

from xninja import agent as agent_module
from xninja.agent import (
    fetch_agent,
    local_agent_source,
    manifest_files,
    run_agent,
)


def write_bundle(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / "agent.py").write_text(
        "from agent.helper import VALUE\n"
        "def solve(repo_path, issue, model, api_base, api_key):\n"
        "    return {'patch': VALUE, 'logs': '', 'steps': 0, 'cost': None, 'success': True}\n",
        encoding="utf-8",
    )
    pkg = root / "agent"
    pkg.mkdir(exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "helper.py").write_text("VALUE = 'multi-file-ok'\n", encoding="utf-8")
    (root / "tau_agent_files.json").write_text(
        json.dumps(["agent.py", "agent/__init__.py", "agent/helper.py"]),
        encoding="utf-8",
    )


def test_manifest_files_defaults_to_entrypoint(tmp_path):
    (tmp_path / "agent.py").write_text("def solve(*a, **k): return {}\n", encoding="utf-8")

    assert manifest_files(tmp_path) == ["agent.py"]


def test_manifest_files_reads_and_sorts(tmp_path):
    write_bundle(tmp_path)

    assert manifest_files(tmp_path) == ["agent.py", "agent/__init__.py", "agent/helper.py"]


@pytest.mark.parametrize(
    "manifest",
    [
        ["agent/helper.py"],  # missing entrypoint
        ["agent.py", "../escape.py"],  # path traversal
        ["agent.py", "/etc/passwd"],  # absolute path
        ["agent.py", "notes.txt"],  # non-python file
        ["agent.py"] + [f"m{i}.py" for i in range(32)],  # over the 32-file cap
    ],
)
def test_manifest_files_rejects_invalid(tmp_path, manifest):
    (tmp_path / "tau_agent_files.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError):
        manifest_files(tmp_path)


def test_local_directory_bundle_loads_with_package_imports(tmp_path):
    bundle = tmp_path / "bundle"
    write_bundle(bundle)

    source = local_agent_source(bundle)
    assert source.bundle_root == bundle
    assert source.path == bundle / "agent.py"

    result = run_agent(source, tmp_path, "task", "model", "base", "key")
    assert result["patch"] == "multi-file-ok"


def test_local_single_file_uses_parent_as_root(tmp_path):
    agent_py = tmp_path / "agent.py"
    agent_py.write_text("def solve(*a, **k): return {}\n", encoding="utf-8")

    source = local_agent_source(agent_py)

    assert source.path == agent_py
    assert source.bundle_root == tmp_path


def test_fetch_agent_downloads_full_bundle(tmp_path, monkeypatch):
    monkeypatch.setenv("XNINJA_CACHE_DIR", str(tmp_path / "cache"))
    files = {
        "tau_agent_files.json": json.dumps(
            ["agent.py", "agent/__init__.py", "agent/helper.py"]
        ).encode("utf-8"),
        "agent.py": b"from agent.helper import VALUE\ndef solve(*a, **k):\n    return {'patch': VALUE}\n",
        "agent/__init__.py": b"",
        "agent/helper.py": b"VALUE = 'fetched'\n",
    }

    def fake_fetch_text(url: str) -> bytes:
        rel = url.split("/main/", 1)[1]
        return files[rel]

    monkeypatch.setattr(agent_module, "_fetch_text", fake_fetch_text)

    source = fetch_agent("main")

    assert sorted(source.metadata["files"]) == ["agent.py", "agent/__init__.py", "agent/helper.py"]
    assert (source.bundle_root / "agent" / "helper.py").is_file()
    result = run_agent(source, tmp_path, "task", "model", "base", "key")
    assert result["patch"] == "fetched"
