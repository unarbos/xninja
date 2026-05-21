from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


def patch_text(agent_result: dict[str, object]) -> str:
    patch = agent_result.get("patch", "")
    return patch if isinstance(patch, str) else ""


def patch_summary(patch: str, max_chars: int = 12000) -> str:
    if len(patch) <= max_chars:
        return patch
    omitted = len(patch) - max_chars
    return patch[:max_chars] + f"\n\n...[truncated {omitted} chars]...\n"


def apply_patch(repo_path: Path, patch: str) -> subprocess.CompletedProcess[str]:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".patch", delete=False) as handle:
        handle.write(patch)
        patch_path = Path(handle.name)
    try:
        return subprocess.run(
            ["git", "apply", str(patch_path)],
            cwd=repo_path,
            text=True,
            capture_output=True,
            check=False,
        )
    finally:
        patch_path.unlink(missing_ok=True)


def repo_is_git_worktree(repo_path: Path) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=repo_path,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"
