from __future__ import annotations
import re
from typing import Optional

_FILE_IN_ISSUE_RE = re.compile(
    r"`?([\w./-]+\.(?:py|ts|tsx|js|jsx|go|rs|java|cs|rb|php|vue|html|css|json|yaml|yml|md|R|r|cpp|h|c|hpp|toml|xml|sql|sh|txt))`?",
    re.I,
)
_MUNGE_PATH_RE = re.compile(
    r"^(?:fix|clean|cleanup|replace|update|patch|apply|munge|modify|gen|generate|"
    r"rewrite|migrate|refactor)_[\w.-]+$",
    re.I,
)
_MUNGE_FILE_RE = re.compile(
    r"^(?:fix|update|replace|refactor|patch|apply|clean|generate|rewrite|migrate|"
    r"modify)_[\w.-]+\.(?:py|sh|js|ts|rb|pl)$",
    re.I,
)
_REFACTOR_ISSUE_RE = re.compile(
    r"\b(refactor|rename|restructur|convert|migrate|reorganiz)\b",
    re.I,
)


def _changed_paths(patch_text: str) -> list[str]:
    paths: list[str] = []
    for line in patch_text.splitlines():
        if line.startswith("+++ b/"):
            path = line[len("+++ b/") :].strip()
            if path and path != "/dev/null" and path not in paths:
                paths.append(path)
    return paths


def _line_stats(patch_text: str) -> tuple[int, int]:
    added = removed = 0
    for line in patch_text.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return added, removed


def destructive_patch_reason(patch_text: str) -> Optional[str]:
    added, removed = _line_stats(patch_text)
    if removed >= 60 and added < max(5, removed // 4):
        return (
            f"the patch removes far more than it adds ({removed} deletions vs {added} additions); "
            "restore required logic instead of gutting the codebase"
        )
    return None


def munge_artifact_reason(patch_text: str) -> Optional[str]:
    for path in _changed_paths(patch_text):
        base = path.rsplit("/", 1)[-1]
        stem = base.rsplit(".", 1)[0] if "." in base else base
        if (
            _MUNGE_PATH_RE.match(stem)
            or _MUNGE_FILE_RE.match(base)
            or base.endswith((".new", ".bak", ".orig", ".tmp", ".rej"))
        ):
            return (
                f"the patch adds scratch or munge artifact `{path}`; "
                "edit source files directly and remove helper/backup files"
            )
    return None


def refactor_delete_reason(issue_text: str, patch_text: str) -> Optional[str]:
    if not _REFACTOR_ISSUE_RE.search(issue_text or ""):
        return None
    added, removed = _line_stats(patch_text)
    if removed >= 30 and added < max(8, removed // 3):
        return (
            f"refactor/rename task but patch mostly deletes code "
            f"({removed} deletions vs {added} additions); implement the change in place"
        )
    return None


def task_coverage_reason(issue_text: str, patch_text: str, repo_dir: Optional[str] = None) -> Optional[str]:
    mentioned = []
    for match in _FILE_IN_ISSUE_RE.finditer(issue_text):
        path = match.group(1).strip().lstrip("./")
        if path not in mentioned:
            mentioned.append(path)
    if not mentioned:
        return None
    touched = _changed_paths(patch_text)
    if not touched:
        return None

    # Filter mentioned files to only those that actually exist in the repo or are being touched/created.
    if repo_dir is not None:
        import os
        valid_mentioned = []
        for m in mentioned:
            # Check if the file exists on disk, or if it is a substring of any touched path,
            # or if any touched path ends with/contains it (e.g. m is "views.py" and t is "app/views.py").
            exists_on_disk = os.path.exists(os.path.join(repo_dir, m))
            is_touched = any(t == m or t.endswith("/" + m) or m.endswith("/" + t) for t in touched)
            if exists_on_disk or is_touched:
                valid_mentioned.append(m)
        mentioned = valid_mentioned

    if not mentioned:
        return None

    hit = sum(
        1
        for m in mentioned
        if any(t == m or t.endswith("/" + m) or m.endswith("/" + t) for t in touched)
    )
    if hit == 0:
        sample = ", ".join(mentioned[:6])
        return (
            f"the task names specific files ({sample}) but the patch does not touch any of them; "
            "find and edit the correct targets"
        )
    return None


def extended_repair_reason(issue_text: str, patch_text: str) -> Optional[str]:
    """Additional patch checks for the repair gate."""
    if not (patch_text or "").strip():
        return None
    for check in (
        destructive_patch_reason,
        munge_artifact_reason,
        lambda p: refactor_delete_reason(issue_text, p),
        lambda p: task_coverage_reason(issue_text, p),
    ):
        reason = check(patch_text)
        if reason:
            return reason
    return None


def patch_acceptable(patch_text: str) -> bool:
    if not patch_text.strip():
        return False
    if destructive_patch_reason(patch_text) or munge_artifact_reason(patch_text):
        return False
    return True
