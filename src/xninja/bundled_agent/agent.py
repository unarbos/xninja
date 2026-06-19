#!/usr/bin/env python3
"""
Multi-file SWE coding agent for the tau subnet.

Contract (unchanged from the public single-file base agent):
    The validator imports this file and calls:

        solve(
            repo_path="/tmp/task_repo",
            issue="Fix the bug...",
            model="validator-managed-model",
            api_base="http://validator-proxy/v1",
            api_key="per-run-proxy-token"
        )

    It returns a dict with patch, logs, steps, cost, and success.

Layout:
    agent.py             validator-owned contract + thin solve() wiring
    agent/prompts.py     system/instance templates for complete, verified fixes
    agent/model.py       stdlib OpenAI-compatible chat client with retries
    agent/environment.py fresh-subshell bash executor
    agent/agent_loop.py  the query -> act -> observe step loop
    agent/repo_diff.py   harness-compatible patch collection
"""

from __future__ import annotations

import os
import json
import re
import subprocess
import time
import traceback
from typing import Any, Dict, Optional, Tuple

from agent.agent_loop import AgentRunConfig, run_agent_loop
from agent.prompts import build_task_prompt
from agent.repo_diff import collect_repo_patch
from agent.criteria import extract_criteria, format_checklist
from agent.guards import (
    destructive_patch_reason,
    munge_artifact_reason,
    refactor_delete_reason,
    task_coverage_reason,
    patch_acceptable,
)

# -----------------------------
# Config
# -----------------------------

DEFAULT_MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", "50"))
DEFAULT_COMMAND_TIMEOUT = int(os.environ.get("AGENT_COMMAND_TIMEOUT", "40"))

DEFAULT_MODEL = os.environ.get("AGENT_MODEL") or os.environ.get("NINJA_MODEL", "")
DEFAULT_API_BASE = (
    os.environ.get("AGENT_API_BASE")
    or os.environ.get("NINJA_INFERENCE_BASE_URL")
    or os.environ.get("OPENAI_BASE_URL", "")
)
DEFAULT_API_KEY = (
    os.environ.get("AGENT_API_KEY")
    or os.environ.get("NINJA_INFERENCE_API_KEY")
    or os.environ.get("OPENAI_API_KEY", "")
)
DEFAULT_MAX_TOKENS = int(os.environ.get("AGENT_MAX_TOKENS", "8192"))

MAX_OBSERVATION_CHARS = int(os.environ.get("AGENT_MAX_OBSERVATION_CHARS", "16000"))
MAX_TOTAL_LOG_CHARS = int(os.environ.get("AGENT_MAX_TOTAL_LOG_CHARS", "260000"))


def _wall_clock_limit_seconds() -> float:
    budget = os.environ.get("TAU_AGENT_TIMEOUT_SECONDS")
    if budget:
        try:
            return max(60.0, float(int(budget)) - 20.0)
        except ValueError:
            pass
    return 280.0


WALL_CLOCK_LIMIT_SECONDS = _wall_clock_limit_seconds()
WALL_CLOCK_RESERVE_SECONDS = 10.0


def _normalize_api_base(api_base: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base[: -len("/chat/completions")]
    if base.endswith("/v1"):
        return base
    return base + "/v1"


def _resolve_inference_config(
    model: Optional[str],
    api_base: Optional[str],
    api_key: Optional[str],
) -> Tuple[str, str, str]:
    model_name = (model or DEFAULT_MODEL).strip()
    base = (api_base or DEFAULT_API_BASE).strip()
    key = (api_key if api_key is not None else DEFAULT_API_KEY).strip()

    if not model_name:
        raise ValueError("model is required; validators must pass the centrally managed model id")
    if not base:
        raise ValueError("api_base is required; validators must pass the managed inference proxy URL")
    if not key:
        raise ValueError("api_key is required; validators must pass the per-run proxy token")

    return model_name, _normalize_api_base(base), key


def build_initial_user_prompt(issue: str, repo_summary: str, preloaded_context: str = "") -> str:
    base = build_task_prompt(task_text=issue, repo_summary=repo_summary, preloaded_context=preloaded_context)
    checklist = format_checklist(extract_criteria(issue))
    return base + checklist if checklist else base


VERIFY_REPAIR_MIN_BUDGET_SECONDS = 45.0
VERIFY_REPAIR_MAX_STEPS = 14


def _changed_py_files(patch_text: str) -> list:
    paths = []
    for line in patch_text.splitlines():
        if line.startswith("+++ b/"):
            path = line[len("+++ b/"):].strip()
            if path.endswith(".py") and path not in paths:
                paths.append(path)
    return paths


def _py_syntax_errors(repo_dir: str, patch_text: str) -> list:
    broken = []
    for rel in _changed_py_files(patch_text):
        full = os.path.join(repo_dir, rel)
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as handle:
                source = handle.read()
        except OSError:
            continue
        try:
            compile(source, rel, "exec")
        except SyntaxError as exc:
            broken.append(f"{rel}: line {exc.lineno}: {exc.msg}")
        except (ValueError, TypeError):
            broken.append(f"{rel}: could not be parsed")
    return broken


def _changed_source_files(patch_text: str, exts: tuple) -> list:
    paths = []
    for line in patch_text.splitlines():
        if line.startswith("+++ b/"):
            path = line[len("+++ b/"):].strip()
            if path.endswith(exts) and path not in paths:
                paths.append(path)
    return paths


def _run_check(cmd: list, cwd: str) -> Optional[str]:
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=20)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    if proc.returncode == 0:
        return None
    msg = (proc.stderr or proc.stdout or "").strip()
    return (msg.splitlines()[0][:200] if msg else "failed syntax check")


_CS_REPEATED_BASE_RE = re.compile(
    r"\b(?:class|interface|struct|record)\s+[A-Za-z_]\w*(?:\s*<[^>]*>)?"
    r"\s*:\s*([A-Za-z_][\w.]*)(?:\s*:\s*\1\b)+"
)

_BRACE_BALANCE_EXTS = (".php", ".cs", ".kt", ".java", ".swift", ".scala")
_DELIM_OPEN = {")": "(", "]": "[", "}": "{"}


def _strip_code_noise(text: str) -> str:
    out = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            j = text.find("\n", i)
            i = n if j < 0 else j
            continue
        if c == "#":
            j = text.find("\n", i)
            i = n if j < 0 else j
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            j = text.find("*/", i + 2)
            if j < 0:
                return ""
            i = j + 2
            continue
        if c in "'\"`":
            quote = c
            i += 1
            while i < n:
                if text[i] == "\\":
                    i += 2
                    continue
                if text[i] == quote:
                    i += 1
                    break
                i += 1
            else:
                return ""
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _delimiter_balance_error(text: str, rel: str):
    if "<<<" in text:
        return None
    code = _strip_code_noise(text)
    if not code:
        return None
    stack = []
    for idx, ch in enumerate(code):
        if ch in "([{":
            stack.append(ch)
        elif ch in ")]}":
            want = _DELIM_OPEN[ch]
            if not stack:
                return f"{rel}: unexpected closing '{ch}' (extra/dangling delimiter)"
            top = stack.pop()
            if top != want:
                return f"{rel}: mismatched '{ch}' (expected close for '{top}')"
    if stack:
        return f"{rel}: {len(stack)} unclosed '{stack[-1]}' delimiter(s) (missing close brace/paren)"
    return None


_DUP_DEF_EXTS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".php", ".cs",
                 ".kt", ".java", ".go", ".swift", ".scala", ".rs")

_DUP_DEF_RE = re.compile(
    r"^[ \t]*"
    r"(?:export\s+)?(?:default\s+)?(?:public\s+|private\s+|protected\s+|internal\s+|static\s+|final\s+|abstract\s+|async\s+)*"
    r"(?:"
    r"(?:class|struct|enum|trait)\s+([A-Za-z_$][\w$]*)"
    r"|type\s+([A-Za-z_$][\w$]*)\s+(?:struct|interface)\b"
    r")",
    re.M,
)


def _duplicate_definition_error(text: str, rel: str):
    code = _strip_code_noise(text)
    if not code:
        return None
    seen = {}
    for mobj in _DUP_DEF_RE.finditer(code):
        name = mobj.group(1) or mobj.group(2)
        if not name:
            continue
        seen[name] = seen.get(name, 0) + 1
    dups = sorted(n for n, c in seen.items() if c > 1)
    if dups:
        return f"{rel}: duplicate top-level definition(s): {', '.join(dups[:4])} (defined more than once -> compile error)"
    return None


def _syntax_errors(repo_dir: str, patch_text: str) -> list:
    broken = []
    for rel in _changed_source_files(patch_text, (".py",)):
        full = os.path.join(repo_dir, rel)
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as fh:
                source = fh.read()
        except OSError:
            continue
        try:
            compile(source, rel, "exec")
        except SyntaxError as exc:
            broken.append(f"{rel}: line {exc.lineno}: {exc.msg}")
        except (ValueError, TypeError):
            broken.append(f"{rel}: could not be parsed")
    for rel in _changed_source_files(patch_text, (".json",)):
        full = os.path.join(repo_dir, rel)
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read()
        except OSError:
            continue
        try:
            json.loads(content)
        except ValueError as exc:
            broken.append(f"{rel}: invalid JSON: {str(exc)[:120]}")
    for rel in _changed_source_files(patch_text, (".js", ".mjs", ".cjs")):
        err = _run_check(["node", "--check", rel], repo_dir)
        if err:
            broken.append(f"{rel}: {err}")
    for rel in _changed_source_files(patch_text, (".go",)):
        err = _run_check(["gofmt", "-e", rel], repo_dir)
        if err:
            broken.append(f"{rel}: {err}")
    for rel in _changed_source_files(patch_text, _BRACE_BALANCE_EXTS):
        full = os.path.join(repo_dir, rel)
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        err = _delimiter_balance_error(text, rel)
        if err:
            broken.append(err)
    for rel in _changed_source_files(patch_text, _DUP_DEF_EXTS):
        full = os.path.join(repo_dir, rel)
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        err = _duplicate_definition_error(text, rel)
        if err:
            broken.append(err)
    for rel in _changed_source_files(patch_text, (".php",)):
        err = _run_check(["php", "-l", rel], repo_dir)
        if err:
            broken.append(f"{rel}: {err}")
    for rel in _changed_source_files(patch_text, (".cs",)):
        full = os.path.join(repo_dir, rel)
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        if _CS_REPEATED_BASE_RE.search(_strip_code_noise(text)):
            broken.append(f"{rel}: malformed repeated base type (e.g. ': X : X')")
    return broken


def _all_changed_files(patch_text: str) -> list:
    out = []
    for line in patch_text.splitlines():
        if line.startswith("+++ b/"):
            p = line[len("+++ b/"):].strip()
            if p and p != "/dev/null" and p not in out:
                out.append(p)
    return out


def _is_test_path(path: str) -> bool:
    p = path.lower()
    base = p.rsplit("/", 1)[-1]
    if any(seg in ("test", "tests", "spec", "specs", "__tests__") for seg in p.split("/")[:-1]):
        return True
    if base.endswith(".py") and (base.startswith("test_") or base.endswith("_test.py") or base.startswith("test")):
        return True
    if ".test." in base or ".spec." in base or base.endswith("_spec.rb") or base.endswith("_test.go"):
        return True
    return False


def _source_files(patch_text: str) -> set:
    return {p for p in _all_changed_files(patch_text) if not _is_test_path(p)}


def _added_test_files(patch_text: str) -> list:
    return [p for p in _all_changed_files(patch_text) if _is_test_path(p)]


def _python_test_outcome(repo_dir: str, patch_text: str) -> str:
    tests = [p for p in _all_changed_files(patch_text)
             if _is_test_path(p) and p.endswith(".py")
             and os.path.isfile(os.path.join(repo_dir, p))]
    if not tests:
        return "none"
    rel = tests[0]
    for exe in ("python", "python3"):
        try:
            proc = subprocess.run(
                [exe, "-m", "pytest", rel, "-x", "-q", "-p", "no:cacheprovider"],
                cwd=repo_dir, capture_output=True, text=True, timeout=25,
            )
        except (OSError, ValueError, subprocess.SubprocessError):
            continue
        if proc.returncode == 0:
            return "pass"
        if proc.returncode == 1:
            return "fail"
        return "unknown"
    return "unknown"


def _repair_reason(repo_dir: str, patch_text: str, issue_text: str = "", check_tests: bool = True):
    if not (patch_text or "").strip():
        return ("empty", "the current change set is empty; no fix was produced yet")
    if check_tests:
        outcome = _python_test_outcome(repo_dir, patch_text)
        if outcome == "fail":
            return ("test_fail", "your own regression test currently FAILS, so the fix is wrong or incomplete; correct the fix until that test passes (never weaken the test).")
        if outcome == "none" and _source_files(patch_text) and not _added_test_files(patch_text):
            return ("no_test", "the fix changes source but includes no test proving it works; ADD one focused regression test that fails on the original bug and passes with your fix, and KEEP the existing source fix in place.")
    cov = task_coverage_reason(issue_text, patch_text, repo_dir=repo_dir)
    if cov:
        return ("coverage", cov)
    broken = _syntax_errors(repo_dir, patch_text)
    if broken:
        return ("syntax", "the edited files contain syntax errors that must be fixed:\n- " + "\n- ".join(broken[:8]))
    q = (
        destructive_patch_reason(patch_text)
        or munge_artifact_reason(patch_text)
        or refactor_delete_reason(issue_text, patch_text)
    )
    if q:
        return ("quality", q)
    return None


def _build_repair_task(issue_text: str, reason: str) -> str:
    return (
        "A previous attempt to solve the task below left the repository in an "
        "incomplete or broken state. " + reason + "\n\n"
        "Inspect the current state of the repository, then finish and correct "
        "the change so it fully and correctly solves the task. Re-read each "
        "edited region to confirm it is syntactically valid before submitting.\n\n"
        "Original task:\n" + issue_text
    )


def _build_polish_task(issue_text: str, reason: str) -> str:
    return (
        "A previous attempt successfully solved the task below, passed all tests, "
        "and has no syntax errors. Now, perform a polishing and refinement pass to "
        "ensure the solution is absolutely perfect, elegant, and production-ready.\n\n"
        "Specifically:\n"
        "1. Remove any unrelated edits, debug prints, or temporary comments.\n"
        "2. Ensure the code matches the existing style perfectly (indentation, quotes).\n"
        "3. Ensure the added regression test is robust, clean, and covers all edge cases.\n"
        "4. Make the changes as concise and precise as possible to minimize churn.\n\n"
        "Original task:\n" + issue_text
    )


def solve(
    repo_path: str,
    issue: str,
    model: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    command_timeout: int = DEFAULT_COMMAND_TIMEOUT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    on_event=None,
) -> Dict[str, Any]:
    started = time.monotonic()
    try:
        model_name, base_url, proxy_token = _resolve_inference_config(model, api_base, api_key)
        run_config = AgentRunConfig(
            repo_dir=repo_path,
            model_name=model_name,
            base_url=base_url,
            auth_token=proxy_token,
            max_steps=max_steps,
            command_timeout=command_timeout,
            max_tokens=max_tokens,
            max_observation_chars=MAX_OBSERVATION_CHARS,
            max_log_chars=MAX_TOTAL_LOG_CHARS,
            wall_clock_limit=WALL_CLOCK_LIMIT_SECONDS,
        )
        outcome = run_agent_loop(
            config=run_config,
            task=build_initial_user_prompt(issue, "", ""),
            on_event=on_event,
        )

        repair_note = ""
        try:
            remaining = WALL_CLOCK_LIMIT_SECONDS - (time.monotonic() - started)
            can_repair = remaining >= VERIFY_REPAIR_MIN_BUDGET_SECONDS
            reason = _repair_reason(repo_path, outcome.patch, issue_text=issue, check_tests=can_repair)
            if reason is None and can_repair:
                reason = ("polish", "The fix is correct and passes all tests, but we must polish and refine it to ensure it is of the highest quality, contains no unrelated churn, has clean and minimal edits, and is fully complete. Review your changes and make them perfect.")
            if reason is not None and can_repair:
                kind, message = reason
                orig_sources = _source_files(outcome.patch)
                repair_config = AgentRunConfig(
                    repo_dir=repo_path,
                    model_name=model_name,
                    base_url=base_url,
                    auth_token=proxy_token,
                    max_steps=min(max_steps, VERIFY_REPAIR_MAX_STEPS),
                    command_timeout=command_timeout,
                    max_tokens=max_tokens,
                    max_observation_chars=MAX_OBSERVATION_CHARS,
                    max_log_chars=MAX_TOTAL_LOG_CHARS,
                    wall_clock_limit=remaining - WALL_CLOCK_RESERVE_SECONDS,
                )
                if kind == "polish":
                    task_prompt = build_initial_user_prompt(_build_polish_task(issue, message), "", "")
                else:
                    task_prompt = build_initial_user_prompt(_build_repair_task(issue, message), "", "")
                if on_event:
                    on_event({"type": "phase", "label": kind})
                repaired = run_agent_loop(
                    config=repair_config,
                    task=task_prompt,
                    on_event=on_event,
                )
                rp = repaired.patch
                if rp.strip() and not _syntax_errors(repo_path, rp) and patch_acceptable(rp):
                    rtest = _python_test_outcome(repo_path, rp)
                    if kind == "empty":
                        adopt = rtest != "fail"
                    elif kind == "coverage":
                        adopt = rtest != "fail"
                    elif kind in ("syntax", "test_fail", "quality", "polish"):
                        adopt = rtest != "fail" and orig_sources.issubset(_source_files(rp))
                    else:
                        gained_test = bool(_added_test_files(rp)) and not _added_test_files(outcome.patch)
                        adopt = gained_test and rtest != "fail" and orig_sources.issubset(_source_files(rp))
                    if adopt:
                        outcome = repaired
                        repair_note = " (repair adopted: %s)" % kind
        except Exception:
            repair_note = " (repair pass skipped after error)"

        elapsed = time.monotonic() - started
        return {
            "patch": outcome.patch,
            "logs": outcome.logs,
            "steps": outcome.steps,
            "cost": outcome.cost,
            "success": outcome.success,
            "message": f"{outcome.exit_status}: {outcome.message} in {elapsed:.1f}s{repair_note}",
        }
    except Exception:
        fallback_patch = collect_repo_patch(repo_path)
        return {
            "patch": fallback_patch,
            "logs": traceback.format_exc()[-8000:],
            "steps": 0,
            "cost": None,
            "success": bool(fallback_patch.strip()),
            "message": "agent crashed; returning the on-disk repository diff",
        }
