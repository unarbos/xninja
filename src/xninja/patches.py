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


def changed_files(patch: str) -> tuple[str, ...]:
    files: list[str] = []
    seen: set[str] = set()
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                candidate = parts[3]
                path = candidate[2:] if candidate.startswith("b/") else candidate
                if path != "/dev/null" and path not in seen:
                    seen.add(path)
                    files.append(path)
        elif line.startswith("+++ ") and not files:
            candidate = line[4:].strip()
            path = candidate[2:] if candidate.startswith("b/") else candidate
            if path != "/dev/null" and path not in seen:
                seen.add(path)
                files.append(path)
    return tuple(files)


def patch_line_counts(patch: str) -> tuple[int, int]:
    added = 0
    deleted = 0
    for line in patch.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            deleted += 1
    return added, deleted


def _git_apply(repo_path: Path, patch: str, args: list[str]) -> subprocess.CompletedProcess[str]:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".patch", delete=False) as handle:
        handle.write(patch)
        patch_path = Path(handle.name)
    try:
        return subprocess.run(
            ["git", "apply", *args, str(patch_path)],
            cwd=repo_path,
            text=True,
            capture_output=True,
            check=False,
        )
    finally:
        patch_path.unlink(missing_ok=True)


def check_patch(repo_path: Path, patch: str) -> subprocess.CompletedProcess[str]:
    return _git_apply(repo_path, patch, ["--check"])


def apply_patch(repo_path: Path, patch: str) -> subprocess.CompletedProcess[str]:
    return _git_apply(repo_path, patch, [])


def repo_is_git_worktree(repo_path: Path) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=repo_path,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"
