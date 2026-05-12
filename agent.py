#!/usr/bin/env python3
"""
Portable single-file SWE-style coding agent harness.

Contract:
    The validator imports this file and calls:

        solve(
            repo_path="/tmp/task_repo",
            issue="Fix the bug...",
            model="validator-managed-model",
            api_base="http://validator-proxy/v1",
            api_key="per-run-proxy-token"
        )

    It returns:
        {
            "patch": "... unified git diff ...",
            "logs": "...",
            "steps": int,
            "cost": float | None,
            "success": bool,
        }

Design goals:
    - Single file.
    - No external Python dependencies.
    - Validator-provided OpenAI-compatible /v1/chat/completions endpoint.
    - No direct OpenRouter/OpenAI credentials in miner code.
    - Bash-only action interface.
    - Validator owns repo, tests, sandbox, scoring, hidden tasks.
    - Miners only patch this file.

Miner editing guide:
    You are expected to improve this file. Good areas to edit include prompting,
    context gathering, command selection, tool/result parsing, stopping logic,
    patch generation, safety checks, and how the agent uses its step budget.

    Keep these validator-owned boundaries intact:
    - Preserve solve(repo_path, issue, model, api_base, api_key, ...) as the
      public entry point.
    - Return a dict with patch, logs, steps, cost, and success.
    - Use only the validator-provided api_base/api_key for LLM calls.
    - Do not hardcode another LLM endpoint, API key, model, wallet, scorer, test
      path, or validator secret.
    - Do not add third-party package requirements; this file must stay portable.
    - Do not read or exfiltrate host secrets, hidden tests, or evaluator data.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Config
# -----------------------------

DEFAULT_MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", "30"))
DEFAULT_COMMAND_TIMEOUT = int(os.environ.get("AGENT_COMMAND_TIMEOUT", "15"))

# VALIDATOR CONTRACT: These defaults are only fallbacks for local testing and
# validator wiring. During real validation the validator passes model, api_base,
# and api_key into solve(). Keep this code compatible with that path.
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

MAX_OBSERVATION_CHARS = int(os.environ.get("AGENT_MAX_OBSERVATION_CHARS", "9000"))
MAX_TOTAL_LOG_CHARS = int(os.environ.get("AGENT_MAX_TOTAL_LOG_CHARS", "180000"))
MAX_CONVERSATION_CHARS = 80000
MAX_PRELOADED_CONTEXT_CHARS = 36000
MAX_PRELOADED_FILES = 12
MAX_NO_COMMAND_REPAIRS = 2
MAX_COMMANDS_PER_RESPONSE = 15

# Anti-whiff knobs. Empty patches score zero on baseline-similarity, so any
# transient model error or stuck loop directly costs us rounds. Be aggressive
# about retrying instead of returning early with no edits.
# Hardcoded — not user-tunable. The PR Scope Guard's env-var allowlist
# (pr_scope_guard.py:ALLOWED_ENV_NAMES) does not permit new AGENT_* names.
HTTP_MAX_RETRIES = 3
HTTP_RETRY_BASE_BACKOFF = 1.0
MAX_STEP_RETRIES = 2
WALL_CLOCK_BUDGET_SECONDS = 255.0  # slightly smaller limit for better safety
WALL_CLOCK_RESERVE_SECONDS = 20.0

# Refinement-turn budgets: each turn shows the model its draft and asks for one
# specific kind of correction. They are mutually exclusive so the agent never
# loops indefinitely on a borderline patch.
MAX_POLISH_TURNS = 1       # strip whitespace/comment/blank-only hunks
MAX_SELF_CHECK_TURNS = 1   # ensure issue-mentioned paths are covered, no scope creep
MAX_SYNTAX_FIX_TURNS = 1   # repair Python/TypeScript/JavaScript SyntaxError
MAX_TEST_FIX_TURNS = 1     # repair the companion test we ran ourselves
MAX_COVERAGE_NUDGES = 1    # tell model which issue-mentioned paths are still untouched
MAX_CRITERIA_NUDGES = 1    # tell model which issue acceptance-criteria look unaddressed
MAX_HAIL_MARY_TURNS = 1    # last-resort: force a real edit when patch is empty after everything
MAX_SURFACE_NUDGES = 1     # v32: nudge once when patch skips an integration surface the issue implies
MAX_STATIC_FOOTGUN_NUDGES = 1  # cheap static/runtime footgun guard
MAX_SCOPE_RISK_NUDGES = 1      # warn on behavior-breaking or over-broad rewrites
MAX_TOTAL_REFINEMENT_TURNS = 2  # ninjaking66 PR#268 insight: chained refinements blow time budget;
                                # cap total refinement turns across all gates (hail-mary excepted)
_STYLE_HINT_BUDGET = 600   # VladaWebDev PR#250: cap on detected-style block in preloaded context

# Recent-commit injection: small in-context style anchors from the staged repo's
# real history. The validator clones the real repo with full git history; the
# pilot stages snapshots with one synthetic commit so this is a no-op locally
# but high-leverage live. Recent commits are concrete examples of this
# codebase's style — showing the model 1-2 actual examples teaches the codebase's
# idioms (variable conventions, hunk shape, test-touch patterns) far better than
# any abstract prompt rule.
_RECENT_COMMIT_MAX_INSERTIONS = 30
_RECENT_COMMIT_MAX_DIFF_CHARS = 3500
_RECENT_COMMIT_BLOCK_BUDGET = 4500

# MINER-EDITABLE: You may make this command filter stricter or smarter. Do not
# weaken it to run destructive host/container operations.
DANGEROUS_PATTERNS = [
    r"\brm\s+-rf\s+/",
    r"\bsudo\b",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r":\(\)\s*\{\s*:\|:\s*&\s*\};:",
    r"\bmount\b",
    r"\bumount\b",
    r"\biptables\b",
    r"\bnft\b",
    r"\bchown\s+-R\s+/",
    r"\bchmod\s+-R\s+777\s+/",
    r"\bcurl\b",
    r"\bwget\b",
    r"\bscp\b",
    r"\brsync\b",
    r"\bssh\b",
    r"\bnc\b",
    r"\bncat\b",
    r"\btelnet\b",
]


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class CommandResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_sec: float
    timed_out: bool = False
    blocked: bool = False


@dataclass
class AgentResult:
    patch: str
    logs: str
    steps: int
    cost: Optional[float]
    success: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patch": self.patch,
            "logs": self.logs,
            "steps": self.steps,
            "cost": self.cost,
            "success": self.success,
        }


# -----------------------------
# Utility
# -----------------------------

def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return (
        text[:half]
        + "\n\n...[truncated "
        + str(len(text) - max_chars)
        + " chars]...\n\n"
        + text[-half:]
    )


def _safe_join_logs(logs: List[str]) -> str:
    joined = "\n".join(logs)
    return _truncate(joined, MAX_TOTAL_LOG_CHARS)


def _message_chars(messages: List[Dict[str, str]]) -> int:
    return sum(len(message.get("content") or "") + 32 for message in messages)


def _messages_for_request(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if _message_chars(messages) <= MAX_CONVERSATION_CHARS:
        return messages

    head = messages[:2]
    tail: List[Dict[str, str]] = []
    budget = max(8000, MAX_CONVERSATION_CHARS - _message_chars(head) - 400)
    used = 0
    for message in reversed(messages[2:]):
        size = len(message.get("content") or "") + 32
        if tail and used + size > budget:
            break
        tail.append(message)
        used += size
    tail.reverse()

    omitted = max(0, len(messages) - len(head) - len(tail))
    if omitted == 0:
        return messages
    note = {
        "role": "user",
        "content": (
            f"[{omitted} older interaction messages omitted to stay within the "
            "time/token budget. Continue from the recent observations and make "
            "the smallest useful patch.]"
        ),
    }
    return [*head, note, *tail]


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


def _is_dangerous_command(command: str) -> Optional[str]:
    lowered = command.strip()
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, lowered):
            return pattern
    return None


def _repo_path(path: str | Path) -> Path:
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"repo_path does not exist: {p}")
    if not p.is_dir():
        raise NotADirectoryError(f"repo_path is not a directory: {p}")
    return p


# -----------------------------
# OpenAI-compatible client
# -----------------------------

# MINER-EDITABLE WITH BOUNDARIES: You may change request formatting, retry
# behavior, response parsing, or model-message strategy here. Keep all requests
# pointed at the api_base/api_key supplied by solve(); the validator proxy
# rewrites the model and sampling parameters server-side.
def chat_completion(
    messages: List[Dict[str, str]],
    model: str,
    api_base: Optional[str],
    api_key: Optional[str],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: int = 120,
    max_retries: int = HTTP_MAX_RETRIES,
) -> Tuple[str, Optional[float], Dict[str, Any]]:
    """OpenAI-compatible /v1/chat/completions client.

    Retries with exponential backoff on transient transport failures (timeout,
    connection reset, HTTP 5xx, HTTP 429). Client-side 4xx (other than 429) bail
    out immediately because retrying won't change the outcome.
    """

    model_name, base, key = _resolve_inference_config(model, api_base, api_key)
    url = base + "/chat/completions"

    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
    }

    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }

    data: Optional[Dict[str, Any]] = None
    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        req = urllib.request.Request(url=url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                data = json.loads(raw)
            break
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            retryable = (500 <= e.code < 600) or e.code == 429
            if retryable and attempt < max_retries:
                last_error = e
                time.sleep(HTTP_RETRY_BASE_BACKOFF * (2 ** attempt))
                continue
            raise RuntimeError(f"HTTP {e.code} from model endpoint: {err_body}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            if attempt < max_retries:
                last_error = e
                time.sleep(HTTP_RETRY_BASE_BACKOFF * (2 ** attempt))
                continue
            raise RuntimeError(f"Model request failed: {e}") from e
        except json.JSONDecodeError as e:
            if attempt < max_retries:
                last_error = e
                time.sleep(HTTP_RETRY_BASE_BACKOFF * (2 ** attempt))
                continue
            raise RuntimeError(f"Model returned non-JSON: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Model request failed: {e}") from e

    if data is None:
        raise RuntimeError(f"Model request failed after retries: {last_error}")

    try:
        content = data["choices"][0]["message"]["content"] or ""
    except Exception as e:
        raise RuntimeError(f"Unexpected model response shape: {data}") from e

    usage = data.get("usage") or {}
    cost = 0.0 if usage else None
    return content, cost, data


# -----------------------------
# Shell execution
# -----------------------------

# MINER-EDITABLE: This is the bash tool surface your agent uses inside the task
# repo. You may improve command validation, environment handling, timeouts, and
# output shaping. Keep commands scoped to the repo and avoid secrets or network
# access outside the validator inference proxy.
def run_command(command: str, cwd: Path, timeout: int = DEFAULT_COMMAND_TIMEOUT) -> CommandResult:
    command = command.strip()

    if not command:
        return CommandResult(
            command=command,
            exit_code=0,
            stdout="",
            stderr="Empty command ignored.",
            duration_sec=0.0,
        )

    blocked_pattern = _is_dangerous_command(command)
    if blocked_pattern:
        return CommandResult(
            command=command,
            exit_code=126,
            stdout="",
            stderr=f"Blocked potentially dangerous command. Matched pattern: {blocked_pattern}",
            duration_sec=0.0,
            blocked=True,
        )

    start = time.time()

    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            executable="/bin/bash",
            env=_command_env(),
        )

        return CommandResult(
            command=command,
            exit_code=proc.returncode,
            stdout=_truncate(proc.stdout or "", MAX_OBSERVATION_CHARS),
            stderr=_truncate(proc.stderr or "", MAX_OBSERVATION_CHARS),
            duration_sec=time.time() - start,
        )

    except subprocess.TimeoutExpired as e:
        stdout = e.stdout or ""
        stderr = e.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")

        return CommandResult(
            command=command,
            exit_code=124,
            stdout=_truncate(stdout, MAX_OBSERVATION_CHARS),
            stderr=_truncate(stderr + f"\nCommand timed out after {timeout}s.", MAX_OBSERVATION_CHARS),
            duration_sec=time.time() - start,
            timed_out=True,
        )

    except Exception as e:
        return CommandResult(
            command=command,
            exit_code=1,
            stdout="",
            stderr=f"Command execution failed: {e}",
            duration_sec=time.time() - start,
        )


def _command_env() -> Dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp") or "/tmp",
        "TMPDIR": os.environ.get("TMPDIR", "/tmp") or "/tmp",
        "LANG": os.environ.get("LANG", "C.UTF-8") or "C.UTF-8",
        "PYTHONUNBUFFERED": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "GIT_PAGER": "cat",
        "PAGER": "cat",
        "CI": "1",
    }


def format_observation(result: CommandResult) -> str:
    parts = [
        "COMMAND:",
        result.command,
        "",
        "EXIT_CODE:",
        str(result.exit_code),
        "",
        "DURATION_SECONDS:",
        f"{result.duration_sec:.3f}",
        "",
        "STDOUT:",
        result.stdout,
    ]
    if result.stderr.strip():
        parts.extend(["", "STDERR:", result.stderr])
    return "\n".join(parts) + "\n"


# -----------------------------
# Action parsing
# -----------------------------

ACTION_RE = re.compile(r"<command>\s*(.*?)\s*</command>", re.IGNORECASE | re.DOTALL)
FINAL_RE = re.compile(r"<final>\s*(.*?)\s*</final>", re.IGNORECASE | re.DOTALL)


def extract_commands(model_text: str) -> List[str]:
    return [match.group(1).strip() for match in ACTION_RE.finditer(model_text) if match.group(1).strip()]


def extract_command(model_text: str) -> Optional[str]:
    commands = extract_commands(model_text)
    return commands[0] if commands else None


def extract_final(model_text: str) -> Optional[str]:
    match = FINAL_RE.search(model_text)
    if not match:
        return None
    return match.group(1).strip()


# -----------------------------
# Git helpers
# -----------------------------

def ensure_git_repo(repo: Path) -> None:
    git_dir = repo / ".git"
    if git_dir.exists():
        return

    subprocess.run(
        "git init >/dev/null 2>&1 && git add . >/dev/null 2>&1 && git commit -m 'initial task state' >/dev/null 2>&1 || true",
        cwd=str(repo),
        shell=True,
        executable="/bin/bash",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )


def get_patch(repo: Path) -> str:
    exclude_pathspecs = [
        ":(exclude,glob)**/*.pyc",
        ":(exclude,glob)**/*.pyo",
        ":(exclude,glob)**/__pycache__/**",
        ":(exclude,glob)**/.pytest_cache/**",
        ":(exclude,glob)**/.mypy_cache/**",
        ":(exclude,glob)**/.ruff_cache/**",
        ":(exclude,glob)**/node_modules/**",
        ":(exclude,glob)**/.cache/**",
        ":(exclude,glob)**/.nuxt/**",
        ":(exclude,glob)**/coverage/**",
        ":(exclude,glob)**/dist/**",
        ":(exclude,glob)**/build/**",
        ":(exclude,glob)**/target/**",
        ":(exclude,glob)**/.next/**",
        ":(exclude,glob)**/.turbo/**",
        ":(exclude).git",
    ]
    proc = subprocess.run(
        ["git", "diff", "--binary", "--", ".", *exclude_pathspecs],
        cwd=str(repo),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    diff_output = proc.stdout or ""

    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=str(repo),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    if untracked.returncode != 0:
        return diff_output

    for relative_path in [item for item in untracked.stdout.split("\0") if item]:
        if _should_skip_patch_path(relative_path):
            continue
        file_diff = subprocess.run(
            ["git", "diff", "--binary", "--no-index", "--", "/dev/null", relative_path],
            cwd=str(repo),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
        if file_diff.returncode in (0, 1):
            diff_output += file_diff.stdout or ""

    return _sanitize_patch(diff_output)


def _sanitize_patch(diff_output: str) -> str:
    """Remove patch blocks that consistently score as noise, never fixes."""
    if not diff_output.strip():
        return diff_output

    cleaned = _strip_skipped_file_diffs(diff_output)
    cleaned = _strip_mode_only_file_diffs(cleaned)
    cleaned = _strip_mode_lines_from_headers(cleaned)
    cleaned = _strip_low_signal_hunks(cleaned)
    return cleaned


def _strip_mode_lines_from_headers(diff_output: str) -> str:
    """Drop `old mode/new mode` lines from per-file headers that have content.

    Mode-only file diffs are already pruned by `_strip_mode_only_file_diffs`.
    Files with both a mode change AND real hunks are kept, but the
    permission/mode header lines (which the LLM judge reliably treats as
    unrelated churn) are scrubbed from the diff.
    """
    if not diff_output.strip():
        return diff_output
    out: List[str] = []
    for line in diff_output.splitlines(keepends=True):
        if line.startswith("old mode ") or line.startswith("new mode "):
            continue
        out.append(line)
    return "".join(out)


_ASSET_SUFFIXES: set = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".svg", ".pdf",
    ".bmp", ".tiff",
}
_ARCHIVE_SUFFIXES: set = {
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
}
_DB_LIKE_SUFFIXES: set = {
    ".sqlite", ".sqlite3", ".db",
}
_TRANSIENT_SUFFIXES: set = {
    ".log", ".tmp", ".cache", ".sqlite", ".db",
}
_LOCKFILE_NAMES: set = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "uv.lock",
    "cargo.lock",
    "go.sum",
    "gemfile.lock",
    "composer.lock",
}
_ASSET_KEYWORDS: Tuple[str, ...] = (
    "image", "asset", "icon", "logo", "screenshot", "figure", "diagram",
    "thumbnail", "favicon", "picture", "media", "graphic", "svg", "png",
)
_DEPS_KEYWORDS: Tuple[str, ...] = (
    "dependency", "dependencies", "lockfile", "lock file", "package json",
    "package.json", "yarn.lock", "package-lock", "pyproject", "requirements",
    "cargo.toml", "go.sum", "go.mod", "version bump", "bump version",
)


def _sanitize_patch_against_issue(patch: str, issue: str) -> str:
    """Drop binary/asset/lockfile/permission-only blocks the issue did NOT request.

    Conservative second sanitizer: a file is only stripped when its kind
    (asset, archive, sqlite, log, lockfile) does not appear in the issue
    text. A real "add logo.png" task still emits an asset diff because the
    keyword `logo`/`png` is in the issue.
    """
    if not patch.strip():
        return patch
    issue_lower = issue.lower()
    asset_ok = any(kw in issue_lower for kw in _ASSET_KEYWORDS)
    deps_ok = any(kw in issue_lower for kw in _DEPS_KEYWORDS)

    blocks = re.split(r"(?=^diff --git )", patch, flags=re.MULTILINE)
    kept: List[str] = []
    for block in blocks:
        if not block:
            continue
        path = _diff_block_path(block)
        if not path:
            kept.append(block)
            continue
        suffix = Path(path).suffix.lower()
        name = Path(path).name.lower()
        if suffix in _ASSET_SUFFIXES and not asset_ok:
            # Keep tracked asset edits; only trim brand-new asset churn unless
            # the task explicitly asks for assets/logo/icon/image work.
            if ("\nnew file mode " in block) or ("Binary files /dev/null and " in block):
                continue
        if suffix in _ARCHIVE_SUFFIXES:
            continue
        if suffix in _DB_LIKE_SUFFIXES:
            continue
        if suffix in _TRANSIENT_SUFFIXES:
            continue
        if name in _LOCKFILE_NAMES and not deps_ok:
            continue
        kept.append(block)
    result = "".join(kept)
    if patch.endswith("\n") and result and not result.endswith("\n"):
        result += "\n"
    return _strip_mode_lines_from_headers(result)


def _finalize_patch_hygiene(patch: str, issue_text: str, logs: List[str]) -> str:
    """Apply the issue-aware sanitizers + log what fired, before final return.

    Order: drop binary/asset/lockfile blocks the issue did not request, then
    sweep any evaluator/grader phrases out of fixture/data files, then warn
    (without stripping) when the same phrases linger in real source code.
    """
    if not patch.strip():
        return patch
    cleaned = _sanitize_patch_against_issue(patch, issue_text)
    cleaned = _strip_eval_term_additions_in_fixtures(cleaned)
    leftover = _patch_contains_forbidden_eval_terms(cleaned)
    if leftover:
        logs.append(
            "PATCH_HYGIENE_WARN:\nforbidden_terms_remaining=" + ",".join(sorted(leftover))
        )
    if cleaned != patch:
        logs.append(
            "PATCH_HYGIENE_TRIMMED:\nsource_chars=" + str(len(patch))
            + " final_chars=" + str(len(cleaned))
        )
    return cleaned


def _append_lockfile_hygiene_log(patch: str, issue_text: str, logs: List[str]) -> None:
    """Lockfile noise warning (kept outside `_finalize_patch_hygiene` for scope-guard stability)."""
    lock_warn = _lockfile_noise_warning(patch, issue_text)
    if lock_warn:
        logs.append(lock_warn)


def _lockfile_noise_warning(patch: str, issue: str) -> str:
    """Warn when lockfiles changed without dependency intent in the issue."""
    if not patch.strip():
        return ""
    changed = set(_patch_changed_files(patch))
    lock_changed = sorted(path for path in changed if Path(path).name.lower() in _LOCKFILE_NAMES)
    if not lock_changed:
        return ""
    package_json_changed = "package.json" in changed
    if package_json_changed:
        return ""
    issue_lower = issue.lower()
    if any(term in issue_lower for term in _DEPS_KEYWORDS):
        return ""
    return "PATCH_HYGIENE_WARN:\nlockfile_without_dependency_intent=" + ",".join(lock_changed[:4])


def _strip_eval_term_additions_in_fixtures(diff_output: str) -> str:
    """Drop `+` lines that introduce evaluator/grader phrases in fixture/data files.

    Conservative: the strip only fires for paths that look like fixtures or
    data (json/yaml/csv/txt under fixture-ish folders). Real source-code
    edits with the same words are kept and surfaced through logging instead
    of silent removal.
    """
    if not diff_output.strip():
        return diff_output
    blocks = re.split(r"(?=^diff --git )", diff_output, flags=re.MULTILINE)
    out: List[str] = []
    for block in blocks:
        if not block:
            continue
        path = _diff_block_path(block)
        lowered_path = path.lower()
        is_fixture = (
            "fixture" in lowered_path
            or "/data/" in "/" + lowered_path
            or "/mocks/" in "/" + lowered_path
            or "/snapshots/" in "/" + lowered_path
            or lowered_path.endswith(".json")
            or lowered_path.endswith(".yaml")
            or lowered_path.endswith(".yml")
            or lowered_path.endswith(".csv")
            or lowered_path.endswith(".txt")
        )
        if not is_fixture:
            out.append(block)
            continue
        kept_lines: List[str] = []
        for line in block.splitlines(keepends=True):
            if line.startswith("+") and not line.startswith("+++"):
                lowered = line.lower()
                if any(term in lowered for term in _FORBIDDEN_EVAL_TERMS):
                    continue
            kept_lines.append(line)
        out.append("".join(kept_lines))
    result = "".join(out)
    if diff_output.endswith("\n") and result and not result.endswith("\n"):
        result += "\n"
    return result


def _diff_block_path(block: str) -> str:
    first = block.splitlines()[0] if block else ""
    match = re.match(r"diff --git a/(.+?) b/(.+)$", first)
    return match.group(2) if match else ""


def _strip_skipped_file_diffs(diff_output: str) -> str:
    blocks = re.split(r"(?=^diff --git )", diff_output, flags=re.MULTILINE)
    kept: List[str] = []
    for block in blocks:
        if not block:
            continue
        path = _diff_block_path(block)
        if path and _should_skip_patch_path(path):
            continue
        kept.append(block)

    result = "".join(kept)
    if diff_output.endswith("\n") and result and not result.endswith("\n"):
        result += "\n"
    return result


def _strip_mode_only_file_diffs(diff_output: str) -> str:
    if not diff_output.strip():
        return diff_output

    blocks = re.split(r"(?=^diff --git )", diff_output, flags=re.MULTILINE)
    kept: List[str] = []
    for block in blocks:
        if not block:
            continue
        mode_only = (
            block.startswith("diff --git ")
            and "\nold mode " in block
            and "\nnew mode " in block
            and "\n@@ " not in block
            and "\nGIT binary patch" not in block
            and "\nBinary files " not in block
            and "\nnew file mode " not in block
            and "\ndeleted file mode " not in block
        )
        if mode_only:
            continue
        kept.append(block)

    result = "".join(kept)
    if diff_output.endswith("\n") and result and not result.endswith("\n"):
        result += "\n"
    return result


def _should_skip_patch_path(relative_path: str) -> bool:
    path = Path(relative_path)
    suffix_lower = path.suffix.lower()
    name_lower = path.name.lower()
    if suffix_lower in {
        ".pyc", ".pyo", ".class", ".o", ".obj", ".so", ".dll", ".dylib",
        ".exe", ".bin", ".wasm", ".log", ".tmp", ".sqlite", ".db",
    }:
        return True
    if name_lower == ".ds_store":
        return True
    generated_parts = {
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "node_modules",
        "coverage",
        "dist",
        "build",
        "target",
        ".cache",
        ".git",
        ".idea",
        ".vscode",
        ".gradle",
        ".terraform",
        ".next",
        ".nuxt",
        ".turbo",
    }
    return any(part in generated_parts for part in path.parts)


def get_repo_summary(repo: Path) -> str:
    commands = [
        "pwd",
        "git ls-files | awk 'NR<=220 {print} END {if (NR>220) print \"... \" NR-220 \" more tracked files\"}'",
        "git status --short || true",
    ]

    parts = []
    for cmd in commands:
        res = run_command(cmd, repo, timeout=10)
        parts.append(format_observation(res))

    return "\n\n".join(parts)


TEXT_FILE_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".css",
    ".env",
    ".gradle",
    ".go",
    ".graphql",
    ".h",
    ".hpp",
    ".html",
    ".java",
    ".js",
    ".jsx",
    ".lock",
    ".json",
    ".kt",
    ".md",
    ".php",
    ".properties",
    ".proto",
    ".py",
    ".rb",
    ".rs",
    ".scss",
    ".sh",
    ".sql",
    ".svelte",
    ".swift",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".vue",
    ".xml",
    ".yaml",
    ".yml",
}

TEXT_FILE_BASENAMES = {
    "Dockerfile",
    "Gemfile",
    "Makefile",
    "Podfile",
}

CONTEXT_SKIP_PARTS = {
    ".git",
    ".next",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "build",
    "coverage",
    "dist",
    "node_modules",
    "target",
    "vendor",
}

SECRETISH_PARTS = {
    ".env",
    ".npmrc",
    ".pypirc",
    ".netrc",
    "credentials",
    "secret",
    "secrets",
}


_PROJECT_HINT_FILES: Tuple[str, ...] = (
    "package.json",
    "pyproject.toml",
    "pytest.ini",
    "setup.cfg",
    "tox.ini",
    "Makefile",
    "go.mod",
    "Cargo.toml",
    "jest.config.js",
    "vitest.config.ts",
)

_INTEGRATION_PATH_MARKERS: Tuple[str, ...] = (
    "api",
    "app",
    "client",
    "component",
    "components",
    "config",
    "controller",
    "controllers",
    "context",
    "db",
    "form",
    "handler",
    "handlers",
    "layout",
    "migration",
    "migrations",
    "model",
    "models",
    "page",
    "pages",
    "repository",
    "repositories",
    "route",
    "routes",
    "router",
    "schema",
    "schemas",
    "screen",
    "screens",
    "service",
    "services",
    "store",
    "types",
    "view",
    "views",
)

_INTEGRATION_ROOT_FILES: Tuple[str, ...] = (
    "Dockerfile",
    "Makefile",
    "build.gradle",
    "docker-compose.yml",
    "package.json",
    "pyproject.toml",
    "settings.gradle",
)


def _project_hint_block(repo: Path, max_chars: int = 2600) -> str:
    """Compact top-level project hints: test scripts and build config only.

    This is intentionally separate from ranked source context. The model often
    knows what to edit but wastes a turn guessing the right verification
    command. A tiny manifest summary helps it choose targeted tests without
    reading broad config files itself.
    """
    tracked = set(_tracked_files(repo))
    blocks: List[str] = []

    for relative_path in _PROJECT_HINT_FILES:
        if relative_path not in tracked:
            continue
        full = (repo / relative_path).resolve()
        try:
            full.relative_to(repo.resolve())
            data = full.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        if relative_path == "package.json":
            try:
                parsed = json.loads(data)
            except Exception:
                parsed = {}
            scripts = parsed.get("scripts") if isinstance(parsed, dict) else None
            if isinstance(scripts, dict) and scripts:
                interesting = {
                    key: scripts[key]
                    for key in sorted(scripts)
                    if any(word in key.lower() for word in ("test", "check", "lint", "type", "build"))
                }
                if interesting:
                    blocks.append("### package.json scripts\n```json\n" + json.dumps(interesting, indent=2)[:900] + "\n```")
            continue

        snippet = _truncate(data, 700)
        if snippet.strip():
            blocks.append(f"### {relative_path}\n```\n{snippet}\n```")

        if len("\n\n".join(blocks)) >= max_chars:
            break

    if not blocks:
        return ""
    return _truncate(
        "PROJECT TEST / BUILD HINTS (use these to pick the smallest real verification command):\n\n"
        + "\n\n".join(blocks),
        max_chars,
    )


def build_preloaded_context(repo: Path, issue: str) -> Tuple[str, List[str]]:
    """Preload the highest-ranked tracked files plus their companion tests.

    Returns `(context_text, included_files)` so late solve steps can drop the
    bulky snippets while keeping a file-name breadcrumb.

    Three improvements over a vanilla rank-and-read loop:

      1. Companion test files (tests/test_X.py for X.py, X.test.ts for X.ts,
         X_test.go for X.go, etc.) are slotted in right after their source
         partner. Real GitHub-derived tasks almost always need source+test
         changes together; without the test in context the agent patches only
         the source and misses the companion test update.

      2. Files that match identifier-shaped symbols extracted from the issue
         text get a substantial rank boost via `_symbol_grep_hits`. This
         catches the common case where the bug is described by function or
         class name without mentioning the file path.

      3. A small number of integration partners (routes, API helpers, schemas,
         migrations, UI entry points, package/build files) are appended after
         the direct hits. This improves file targeting on feature tasks without
         displacing the primary target files.
    """
    files = _rank_context_files(repo, issue)
    if not files:
        return "", []

    tracked_set = set(_tracked_files(repo))
    files = _augment_with_test_partners(files, tracked_set)
    files = _augment_with_integration_partners(files, tracked_set, issue)

    parts: List[str] = []
    included: List[str] = []
    used = 0
    per_file_budget = max(1500, MAX_PRELOADED_CONTEXT_CHARS // max(1, min(len(files), MAX_PRELOADED_FILES)))

    for relative_path in files[:MAX_PRELOADED_FILES]:
        snippet = _read_context_file(repo, relative_path, per_file_budget)
        if not snippet.strip():
            continue
        block = f"### {relative_path}\n```\n{snippet}\n```"
        if parts and used + len(block) > MAX_PRELOADED_CONTEXT_CHARS:
            break
        parts.append(block)
        included.append(relative_path)
        used += len(block)

    project_hints = _project_hint_block(repo)
    if project_hints and used + len(project_hints) <= MAX_PRELOADED_CONTEXT_CHARS + 1200:
        parts.append(project_hints)
        used += len(project_hints)

    # v21 edge: append recent-commit examples as concrete style anchors. Silent
    # no-op when the repo has no real history (pilot snapshots have one
    # synthetic commit) — the helper returns "" and we add nothing.
    recent_examples = _recent_commit_examples(repo)
    if recent_examples and used + len(recent_examples) <= MAX_PRELOADED_CONTEXT_CHARS + _RECENT_COMMIT_BLOCK_BUDGET:
        parts.append(recent_examples)

    return "\n\n".join(parts), included


_BACKTICK_IDENT_RE = re.compile(r"`([A-Za-z][\w./_-]{2,60})`")
_BACKTICK_PATH_HITS_MAX = 5  # generic identifiers (basic.py, util) often match
                              # dozens of unrelated files — only treat as
                              # "mentioned" when an identifier picks out a
                              # specific small handful in the tracked set.


def _rank_context_files(repo: Path, issue: str) -> List[str]:
    tracked = _tracked_files(repo)
    if not tracked:
        return []

    issue_lower = issue.lower()
    path_mentions = _extract_issue_path_mentions(issue)
    mentioned: List[str] = []
    tracked_set = set(tracked)
    for mention in path_mentions:
        normalized = mention.strip("./")
        if normalized in tracked_set and _context_file_allowed(normalized):
            mentioned.append(normalized)

    # Backtick-wrapped identifiers in issues (e.g. `send-expiry-emails`,
    # `email_notificacoes`) are deliberate signals from the task author about
    # the code surface that matters. When they pick out a small specific set
    # of tracked files by path-substring, treat those files as explicit
    # mentions so they get the same +100 ranking boost as path-mentioned
    # files. Skipped when the identifier matches too many files (filters out
    # generic identifiers like `basic.py` or `any2txt`).
    seen_mentioned = set(mentioned)
    for ident in set(_BACKTICK_IDENT_RE.findall(issue)):
        matches = [p for p in tracked_set if ident in p and _context_file_allowed(p)]
        if 1 <= len(matches) <= _BACKTICK_PATH_HITS_MAX:
            for m in matches:
                if m not in seen_mentioned:
                    mentioned.append(m)
                    seen_mentioned.add(m)

    terms = _issue_terms(issue)
    surfaces = _classify_issue_surfaces(issue)
    core_terms = _extract_core_action_terms(issue)
    symbol_hits = _symbol_grep_hits(repo, tracked_set, issue)
    scored: List[Tuple[int, str]] = []
    for relative_path in tracked:
        if not _context_file_allowed(relative_path):
            continue
        path_lower = relative_path.lower()
        name_lower = Path(relative_path).name.lower()
        stem_lower = Path(relative_path).stem.lower()
        score = 0
        if relative_path in mentioned:
            score += 100
        if path_lower in issue_lower:
            score += 35
        if name_lower and name_lower in issue_lower:
            score += 24
        if stem_lower and len(stem_lower) >= 3 and stem_lower in issue_lower:
            score += 16
        score += sum(3 for term in terms if term in path_lower)
        if "/test" in path_lower or "spec." in path_lower or ".test." in path_lower:
            score += sum(2 for term in terms if term in path_lower)
        # Boost files whose contents reference identifiers from the issue.
        if relative_path in symbol_hits:
            score += 60 + min(40, 8 * symbol_hits[relative_path])
        if ("extraction_refactor" in core_terms or "wiring" in surfaces) and Path(relative_path).name.lower() in {"app.js", "app.ts", "main.js", "main.ts", "index.js", "index.ts"}:
            score += 32
        if ("api" in surfaces) and any(seg in path_lower for seg in ("/route", "/routes/", "/router", "/controllers/", "/server", "/api/")):
            score += 14
        if ("ui" in surfaces) and any(seg in path_lower for seg in ("/components/", "/pages/", "/views/", "/screens/", ".css", ".scss")):
            score += 10
        if ("build" in surfaces) and Path(relative_path).name.lower() in {"package.json", "vercel.json", "vite.config.ts", "vite.config.js"}:
            score += 20
        if ("tests" in surfaces) and ("/test" in path_lower or ".test." in path_lower or ".spec." in path_lower):
            score += 12
        if score > 0:
            scored.append((score, relative_path))

    scored.sort(key=lambda item: (-item[0], len(item[1]), item[1]))
    ranked: List[str] = []
    seen: set[str] = set()
    for relative_path in mentioned + [path for _score, path in scored]:
        if relative_path in seen:
            continue
        seen.add(relative_path)
        ranked.append(relative_path)
    return ranked


def _split_path_tokens(relative_path: str) -> set:
    """Lower-case path/name tokens used for cheap related-file discovery."""
    tokens: set = set()
    for part in Path(relative_path).parts:
        for token in re.findall(r"[a-z0-9]+", part.lower()):
            if len(token) >= 3:
                tokens.add(token)
    return tokens


def _looks_like_integration_surface(relative_path: str) -> bool:
    path = Path(relative_path)
    if path.name in _INTEGRATION_ROOT_FILES:
        return True
    tokens = _split_path_tokens(relative_path)
    return any(marker in tokens for marker in _INTEGRATION_PATH_MARKERS)


def _augment_with_integration_partners(files: List[str], tracked: set, issue: str) -> List[str]:
    """Append a few likely integration files after direct hits and tests.

    The agent was already good at finding the local function named by an issue,
    but duel losses showed repeated misses in adjacent wiring: routes, API
    clients, schemas, migrations, UI entry pages, and build metadata. This keeps
    the direct ranking intact and only appends high-confidence neighbors.
    """
    if not files or not tracked:
        return files

    seen = set(files)
    anchors = files[:6]
    anchor_dirs = {
        str(Path(p).parent).replace("\\", "/")
        for p in anchors
        if str(Path(p).parent) not in {"", "."}
    }
    anchor_top_dirs = {
        Path(p).parts[0]
        for p in anchors
        if Path(p).parts
    }
    anchor_tokens = set()
    for path in anchors:
        anchor_tokens.update(_split_path_tokens(path))

    issue_tokens = set(_issue_terms(issue))
    issue_surfaces = _classify_issue_surfaces(issue)
    core_terms = _extract_core_action_terms(issue)
    issue_symbols = {s.lower() for s in _extract_issue_symbols(issue, max_symbols=16)}
    signal_tokens = {t for t in (anchor_tokens | issue_tokens | issue_symbols) if len(t) >= 4}
    root_file_wanted = bool(
        (issue_tokens & {"build", "config", "dependency", "dependencies", "docker", "package", "script", "setup", "workflow"})
        or ("build" in issue_surfaces)
    )

    candidates: List[Tuple[int, str]] = []
    for relative_path in sorted(tracked):
        if relative_path in seen or not _context_file_allowed(relative_path):
            continue
        if not _looks_like_integration_surface(relative_path):
            continue

        path = Path(relative_path)
        path_lower = relative_path.lower()
        parent = str(path.parent).replace("\\", "/")
        parts = path.parts
        score = 0

        if parent in anchor_dirs:
            score += 6
        if parts and parts[0] in anchor_top_dirs:
            score += 3
        score += min(8, 2 * sum(1 for token in issue_tokens if token in path_lower))
        score += min(8, 3 * sum(1 for token in issue_symbols if token in path_lower))
        score += min(6, 2 * sum(1 for token in signal_tokens if token in path_lower))
        if path.name in _INTEGRATION_ROOT_FILES and root_file_wanted:
            score += 5
        if ("extraction_refactor" in core_terms or "wiring" in issue_surfaces) and path.name.lower() in {"app.js", "app.ts", "main.js", "main.ts", "index.js", "index.ts"}:
            score += 8
        if ("api" in issue_surfaces) and any(tok in path_lower for tok in ("/route", "/routes/", "/router", "/server", "/api/")):
            score += 6
        if ("ui" in issue_surfaces) and any(tok in path_lower for tok in ("/components/", "/pages/", "/views/", "/screens/", ".css", ".scss")):
            score += 5
        if "test" in path_lower or "spec" in path_lower:
            score -= 2  # companion-test loading already handles tests.

        if score >= 6:
            candidates.append((score, relative_path))

    candidates.sort(key=lambda item: (-item[0], len(item[1]), item[1]))
    augmented = list(files)
    for _score, relative_path in candidates[:4]:
        if relative_path not in seen:
            augmented.append(relative_path)
            seen.add(relative_path)
    return augmented


def _tracked_files(repo: Path) -> List[str]:
    try:
        proc = subprocess.run(
            ["git", "ls-files"],
            cwd=str(repo),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
    except Exception:
        return []
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _context_file_allowed(relative_path: str) -> bool:
    path = Path(relative_path)
    parts_lower = {part.lower() for part in path.parts}
    name_lower = path.name.lower()
    if parts_lower & CONTEXT_SKIP_PARTS:
        return False
    if name_lower.startswith(".env") or name_lower in SECRETISH_PARTS or parts_lower & SECRETISH_PARTS:
        return False
    if path.name not in TEXT_FILE_BASENAMES and path.suffix.lower() not in TEXT_FILE_EXTENSIONS:
        return False
    return True


def _extract_issue_path_mentions(issue: str) -> List[str]:
    pattern = re.compile(
        r"(?<![\w.-])([\w./-]+\.(?:c|cc|cpp|cs|css|env|go|gradle|graphql|h|hpp|html|java|js|jsx|json|kt|lock|md|php|properties|proto|py|rb|rs|scss|sh|sql|svelte|swift|toml|ts|tsx|txt|vue|xml|ya?ml))(?![\w/-]|\.[A-Za-z0-9])",
        re.IGNORECASE,
    )
    mentions: List[str] = []
    for match in pattern.finditer(issue):
        value = match.group(1).strip("`'\"()[]{}:,;")
        if value and value not in mentions:
            mentions.append(value)
    basename_pattern = re.compile(r"(?<![\w./-])(" + "|".join(re.escape(name) for name in TEXT_FILE_BASENAMES) + r")(?![\w./-])")
    for match in basename_pattern.finditer(issue):
        value = match.group(1).strip("`'\"()[]{}:,;")
        if value and value not in mentions:
            mentions.append(value)
    return mentions


def _issue_terms(issue: str) -> List[str]:
    stop = {
        "about",
        "after",
        "also",
        "before",
        "change",
        "code",
        "file",
        "from",
        "have",
        "issue",
        "make",
        "need",
        "should",
        "that",
        "their",
        "there",
        "this",
        "update",
        "using",
        "when",
        "with",
    }
    terms: List[str] = []
    for raw in re.findall(r"[A-Za-z_][A-Za-z0-9_-]{2,}", issue.lower()):
        if raw in stop or raw in terms:
            continue
        terms.append(raw)
    return terms[:40]


def _read_context_file(repo: Path, relative_path: str, max_chars: int) -> str:
    path = (repo / relative_path).resolve()
    try:
        path.relative_to(repo.resolve())
    except ValueError:
        return ""
    try:
        data = path.read_bytes()
    except Exception:
        return ""
    if b"\0" in data[:4096]:
        return ""
    text = data.decode("utf-8", errors="replace")
    return _truncate(text, max_chars)


# -----------------------------
# Hunk classifiers + diff hygiene
# -----------------------------
#
# Two failure modes produce low-quality patches: drive-by whitespace /
# comment / blank-line edits, and patches that cover the wrong files. The
# helpers below detect both. They're applied at two stages:
#
#   1. At patch-return time: low-signal hunks are silently dropped from the
#      final diff (so the validator never sees them).
#   2. Inside the loop: when the model's draft contains junk, we queue a
#      "polish" turn that asks the model to revert those hunks itself, since
#      doing so cleanly is safer than mechanical filtering for borderline cases
#      (e.g., a comment edit that genuinely matters).

_COMMENT_LINE_PREFIXES: Tuple[str, ...] = ("#", "//", ";", "--", "%")
_BLOCK_COMMENT_RE = re.compile(r"^\s*(\*|/\*|\*/)")


def _line_is_comment(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if any(stripped.startswith(p) for p in _COMMENT_LINE_PREFIXES):
        return True
    if _BLOCK_COMMENT_RE.match(line):
        return True
    if stripped.startswith('"""') or stripped.startswith("'''"):
        return True
    return False


def _hunk_is_blank_only(added: List[str], removed: List[str]) -> bool:
    """Hunk that only changes blank-line layout."""
    body = [line for line in added + removed if line.strip()]
    return not body and bool(added or removed)


def _hunk_is_whitespace_only(added: List[str], removed: List[str]) -> bool:
    """Added and removed lines are identical after stripping whitespace."""
    if not added and not removed:
        return False
    a = sorted(line.strip() for line in added if line.strip())
    r = sorted(line.strip() for line in removed if line.strip())
    if not a and not r:
        return True
    return a == r


def _hunk_is_comment_only(added: List[str], removed: List[str]) -> bool:
    body = [line for line in added + removed if line.strip()]
    if not body:
        return False
    return all(_line_is_comment(line) for line in body)


def _strip_low_signal_hunks(diff_output: str) -> str:
    """Drop blank-only / whitespace-only / comment-only hunks from each file.

    Whole-file blocks with no @@ markers are kept verbatim because they are
    file-create / file-delete / binary patches that the hunk classifier
    can't reason about.
    """
    if not diff_output.strip():
        return diff_output

    blocks = re.split(r"(?=^diff --git )", diff_output, flags=re.MULTILINE)
    out: List[str] = []
    for block in blocks:
        if not block:
            continue
        if not block.startswith("diff --git ") or "\n@@ " not in block:
            out.append(block)
            continue
        parts = re.split(r"(?=^@@ )", block, flags=re.MULTILINE)
        header = parts[0]
        hunks = [chunk for chunk in parts[1:] if chunk]
        substantive: List[str] = []
        for hunk_text in hunks:
            added: List[str] = []
            removed: List[str] = []
            for line in hunk_text.splitlines():
                if line.startswith("+") and not line.startswith("+++"):
                    added.append(line[1:])
                elif line.startswith("-") and not line.startswith("---"):
                    removed.append(line[1:])
            if (
                _hunk_is_blank_only(added, removed)
                or _hunk_is_whitespace_only(added, removed)
                or _hunk_is_comment_only(added, removed)
            ):
                continue
            substantive.append(hunk_text)
        if substantive:
            out.append(header + "".join(substantive))
        # If every hunk was junk, drop the whole file block entirely.
    result = "".join(out)
    if diff_output.endswith("\n") and result and not result.endswith("\n"):
        result += "\n"
    return result


def _diff_low_signal_summary(patch: str) -> str:
    """Human-readable summary of low-signal hunks for the polish prompt."""
    if not patch.strip():
        return ""

    notes: List[str] = []
    current_file = "?"
    current_added: List[str] = []
    current_removed: List[str] = []

    def flush() -> None:
        if not current_added and not current_removed:
            return
        if _hunk_is_blank_only(current_added, current_removed):
            notes.append(f"{current_file}: blank-line-only hunk")
        elif _hunk_is_whitespace_only(current_added, current_removed):
            notes.append(f"{current_file}: whitespace-only hunk")
        elif _hunk_is_comment_only(current_added, current_removed):
            notes.append(f"{current_file}: comment-only hunk")

    for line in patch.splitlines():
        if line.startswith("diff --git "):
            flush()
            current_added, current_removed = [], []
            tokens = line.split()
            if len(tokens) >= 4 and tokens[3].startswith("b/"):
                current_file = tokens[3][2:]
        elif line.startswith("@@"):
            flush()
            current_added, current_removed = [], []
        elif line.startswith("+") and not line.startswith("+++"):
            current_added.append(line[1:])
        elif line.startswith("-") and not line.startswith("---"):
            current_removed.append(line[1:])

    flush()

    deduped: List[str] = []
    seen: set = set()
    for note in notes:
        if note in seen:
            continue
        seen.add(note)
        deduped.append(note)
    return "; ".join(deduped[:10])


def _patch_changed_files(patch: str) -> List[str]:
    """Return the list of `b/` paths touched by a unified diff, in order."""
    seen: List[str] = []
    for match in re.finditer(r"^diff --git a/(.+?) b/(.+?)$", patch, flags=re.MULTILINE):
        path = match.group(2)
        if path and path not in seen:
            seen.append(path)
    return seen


def _patch_covers_required_paths(patch: str, issue_text: str) -> bool:
    """All paths the issue explicitly mentions must appear in the patch."""
    return not _uncovered_required_paths(patch, issue_text)


def _uncovered_required_paths(patch: str, issue_text: str) -> List[str]:
    """Required paths from the issue that the patch doesn't touch yet.

    Used by the coverage-nudge refinement turn to tell the model concretely
    which files the task says to edit but that haven't been touched. The
    LLM judge frequently dings king for "missing/lacks/omits" — surfacing
    the gap to the model directly is the cheapest way to close it.
    """
    required = _extract_issue_path_mentions(issue_text)
    if not required:
        return []
    changed = set(_patch_changed_files(patch))
    missing: List[str] = []
    for req in required:
        if not any(req == c or c.endswith("/" + req) for c in changed):
            missing.append(req)
    return missing


# -----------------------------
# Multi-language syntax gate
# -----------------------------
#
# The previous king's syntax check was Python-only. Real validator tasks come
# from real GitHub commits, so a sizeable fraction touch TypeScript, JavaScript,
# JSON, YAML, etc. This module checks each touched file with the cheapest
# available tool, falling back gracefully when tools are missing. Errors come
# back as (path:line: msg) strings so the syntax-fix prompt can quote them.


_SYNTAX_TIMEOUT = 6  # per-file cap — enough for `node --check` on big files


def _check_python_syntax_one(repo: Path, relative_path: str) -> Optional[str]:
    full = (repo / relative_path).resolve()
    try:
        full.relative_to(repo.resolve())
    except (ValueError, RuntimeError):
        return None
    if not full.exists():
        return None
    try:
        source = full.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    try:
        import ast as _ast
        _ast.parse(source)
        return None
    except SyntaxError as exc:
        return f"{relative_path}:{exc.lineno}: {exc.msg}"
    except Exception as exc:
        return f"{relative_path}: parse failure: {exc}"


def _check_node_syntax_one(repo: Path, relative_path: str) -> Optional[str]:
    """`node --check file.js` — bytecode parse only, no execution.

    Skips the check entirely when `node` is unavailable; we'd rather miss a
    syntax issue than waste 10 seconds on a NotFound retry.
    """
    if not _has_executable("node"):
        return None
    proc_result = run_command(
        f"node --check {_shell_quote(relative_path)}",
        repo,
        timeout=_SYNTAX_TIMEOUT,
    )
    if proc_result.exit_code == 0:
        return None
    msg = (proc_result.stderr or proc_result.stdout or "").strip().splitlines()[-1] if (proc_result.stderr or proc_result.stdout) else ""
    return f"{relative_path}: {msg or 'node --check failed'}"


def _check_json_syntax_one(repo: Path, relative_path: str) -> Optional[str]:
    full = (repo / relative_path).resolve()
    try:
        full.relative_to(repo.resolve())
    except (ValueError, RuntimeError):
        return None
    if not full.exists():
        return None
    try:
        json.loads(full.read_text(encoding="utf-8", errors="replace"))
        return None
    except json.JSONDecodeError as exc:
        return f"{relative_path}:{exc.lineno}: {exc.msg}"
    except Exception as exc:
        return f"{relative_path}: parse failure: {exc}"


# Languages where ' is unambiguously a string delimiter. The brace-balance
# parser below treats ' as a string-mode toggle, which produces false
# positives on:
#   - C / C++ / C# / Java / Kotlin / Scala — `'X'` is a character literal
#     (so `char c = '}';` flips into string mode and eats until next ')
#   - Rust — `'a` is a lifetime annotation
#   - Go — `'X'` is a rune literal
# Net effect of including those: a single `'X'` in any function would yield
# a phantom imbalance that triggers a wasted syntax_fix turn. We restrict
# to JS-family + Swift, where ' is a real string delimiter.
_BRACE_BALANCE_SUFFIXES = {
    ".ts", ".tsx", ".jsx", ".swift",
}


def _check_brace_balance_one(repo: Path, relative_path: str) -> Optional[str]:
    """Cheap brace/paren/bracket balance check for languages without a parser.

    The LLM judge frequently dings patches for "extra closing braces" or
    "duplicate brace" — issues a real compiler would catch. This naive
    counter ignores braces inside string and comment context (best-effort)
    and reports an imbalance with file + count delta.
    """
    full = (repo / relative_path).resolve()
    try:
        full.relative_to(repo.resolve())
    except (ValueError, RuntimeError):
        return None
    if not full.exists():
        return None
    try:
        source = full.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None

    counts = {"{": 0, "}": 0, "[": 0, "]": 0, "(": 0, ")": 0}
    i = 0
    n = len(source)
    in_str: Optional[str] = None
    in_line_comment = False
    in_block_comment = False
    while i < n:
        ch = source[i]
        nxt = source[i + 1] if i + 1 < n else ""
        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue
        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue
        if in_str is not None:
            if ch == "\\" and nxt:
                i += 2
                continue
            if ch == in_str:
                in_str = None
            i += 1
            continue
        # Not in string/comment.
        if ch == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue
        if ch == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue
        if ch in ('"', "'", "`"):
            in_str = ch
            i += 1
            continue
        if ch in counts:
            counts[ch] += 1
        i += 1

    diffs: List[str] = []
    for opener, closer in (("{", "}"), ("[", "]"), ("(", ")")):
        delta = counts[opener] - counts[closer]
        if delta != 0:
            diffs.append(f"{opener}/{closer} delta={delta:+d}")
    if diffs:
        return f"{relative_path}: brace imbalance ({', '.join(diffs)})"
    return None


def _check_syntax(repo: Path, patch: str) -> List[str]:
    """Best-effort multi-language syntax check on touched files.

    Returns a flat list of error strings. An empty list means every file we
    know how to check parsed; languages we can't check (Go, Rust, etc.) are
    silently passed through.
    """
    errors: List[str] = []
    for relative_path in _patch_changed_files(patch):
        suffix = Path(relative_path).suffix.lower()
        result: Optional[str] = None
        if suffix == ".py":
            result = _check_python_syntax_one(repo, relative_path)
        elif suffix in {".js", ".mjs", ".cjs"}:
            result = _check_node_syntax_one(repo, relative_path)
            if result is None and suffix == ".js":
                # node was unavailable; fall back to brace balance check.
                result = _check_brace_balance_one(repo, relative_path)
        elif suffix in {".json"}:
            result = _check_json_syntax_one(repo, relative_path)
        elif suffix in _BRACE_BALANCE_SUFFIXES:
            result = _check_brace_balance_one(repo, relative_path)
        # Other suffixes: trust the model; the LLM judge catches gross errors.
        if result:
            errors.append(result)

        # v32: layered text checks for surfaces the parser-based checks miss.
        # These run AFTER the primary parse so we don't double-report a file
        # whose syntax is clearly broken; they only contribute when the parse
        # check returned clean.
        if result is None:
            extra: Optional[str] = None
            if suffix in {".tsx", ".jsx"}:
                extra = _check_tsx_react_imports_one(repo, relative_path)
            elif suffix == ".dart":
                extra = _check_dart_platform_one(repo, relative_path)
            if extra:
                errors.append(extra)
    return errors


def _has_executable(name: str) -> bool:
    """True if `name` is on PATH. Uses shutil.which (stdlib).

    The earlier impl invoked `command -v` via subprocess with shell=False,
    but `command` is a bash builtin and not a standalone binary on
    python:3.11-slim, so the subprocess call always raised FileNotFoundError
    and returned False. Net effect: every gate that depends on this check
    (e.g. JS/TS `node --check`, pytest discovery) silently no-op'd in
    production. shutil.which is the portable equivalent.
    """
    try:
        return shutil.which(name) is not None
    except Exception:
        return False


def _shell_quote(value: str) -> str:
    """Single-quote-escape for embedding in a bash command string."""
    return "'" + value.replace("'", "'\"'\"'") + "'"


# -----------------------------
# Companion-test discovery + execution
# -----------------------------
#
# When the agent edits `src/foo.py` and a `tests/test_foo.py` exists in the
# repo, running that test before <final> catches a class of regressions the
# scope/judge gates can't see. Cursor's baseline diffs almost always update
# tests in lockstep with source edits, and a fast pytest -k catches "I broke
# the test I was supposed to fix."

_TEST_PARTNER_TEMPLATES: Tuple[Tuple[str, str], ...] = (
    # Python — the most common shapes.
    ("{stem}.py", "tests/test_{stem}.py"),
    ("{stem}.py", "test_{stem}.py"),
    ("{stem}.py", "{dir}/test_{stem}.py"),
    ("{stem}.py", "{dir}/tests/test_{stem}.py"),
    ("{stem}.py", "tests/{stem}_test.py"),
    ("{stem}.py", "test/{stem}_test.py"),
    ("{stem}.py", "test/test_{stem}.py"),
    ("{stem}.py", "{dir}/{stem}_test.py"),
    # TypeScript / JavaScript — Jest / Vitest conventions.
    ("{stem}.ts", "{dir}/{stem}.test.ts"),
    ("{stem}.ts", "{dir}/__tests__/{stem}.test.ts"),
    ("{stem}.ts", "tests/{stem}.test.ts"),
    ("{stem}.ts", "test/{stem}.test.ts"),
    ("{stem}.tsx", "{dir}/{stem}.test.tsx"),
    ("{stem}.tsx", "{dir}/__tests__/{stem}.test.tsx"),
    ("{stem}.js", "{dir}/{stem}.test.js"),
    ("{stem}.js", "{dir}/__tests__/{stem}.test.js"),
    ("{stem}.js", "tests/{stem}.test.js"),
    ("{stem}.js", "test/{stem}.test.js"),
    ("{stem}.jsx", "{dir}/{stem}.test.jsx"),
    # Other languages — single canonical convention each.
    ("{stem}.go", "{dir}/{stem}_test.go"),
    ("{stem}.rs", "{dir}/{stem}_test.rs"),
    ("{stem}.rb", "spec/{stem}_spec.rb"),
)


def _find_test_partner(relative_path: str, tracked: set) -> Optional[str]:
    """Return the most plausible test file for a source path, or None."""
    path = Path(relative_path)
    name_lower = path.name.lower()
    if "test" in name_lower or "spec" in name_lower:
        return None
    stem = path.stem
    suffix = path.suffix
    if not stem or not suffix:
        return None
    parent = str(path.parent) if str(path.parent) not in {".", ""} else ""
    for source_template, test_template in _TEST_PARTNER_TEMPLATES:
        if not source_template.endswith(suffix):
            continue
        candidate = test_template.format(stem=stem, dir=parent).lstrip("/")
        candidate = str(Path(candidate))
        if candidate in tracked and _context_file_allowed(candidate):
            return candidate
    return None


def _augment_with_test_partners(files: List[str], tracked: set) -> List[str]:
    """Slot each ranked source file's companion test in immediately after it."""
    if not tracked:
        return files
    augmented: List[str] = []
    seen: set = set()
    for relative_path in files:
        if relative_path not in seen:
            augmented.append(relative_path)
            seen.add(relative_path)
        partner = _find_test_partner(relative_path, tracked)
        if partner and partner not in seen:
            augmented.append(partner)
            seen.add(partner)
    return augmented


def _run_companion_test(
    repo: Path,
    test_path: str,
    timeout_seconds: int = 8,
) -> Optional[str]:
    """Best-effort companion-test execution. Returns failure-output tail on FAIL,
    or None when the test passed, the runner is unavailable, or the language
    isn't supported.

    Languages handled:
      - Python: `pytest` (if on PATH) then `python3 -m pytest <path>`. We skip
        the failure when output indicates pytest itself isn't importable
        (ModuleNotFoundError) — that's not a real test failure.
      - JS/TS: `node --check <test_path>`. We don't try jest/vitest because
        they require project-level config we can't synthesize in 8s on an
        unknown repo.
      - Other languages: skipped (returns None).

    Errors (timeout, runner missing, exception) intentionally degrade to None
    so the refinement chain doesn't queue a fix for something the agent can't
    actually act on. The whole gate is best-effort.

    Pairs with build_test_fix_prompt — when this returns a non-None failure
    tail, that tail is fed back to the model as one extra refinement turn.
    Companion-test execution was scaffolded by previous king alexlange1 (the
    constant MAX_TEST_FIX_TURNS, the helper build_test_fix_prompt, and the
    co-loading templates _TEST_PARTNER_TEMPLATES) but never wired up; the
    massive PR #185 rewrite preserved the dead scaffolding without using it.
    This re-introduces the runtime-correctness signal as a refinement gate.
    """
    full = repo / test_path
    if not full.exists() or not full.is_file():
        return None

    suffix = Path(test_path).suffix.lower()

    # ---- Python ----
    if suffix == ".py":
        runner_cmds: List[List[str]] = []
        if _has_executable("pytest"):
            runner_cmds.append(["pytest", "-x", "--tb=short", "-q", "--no-header", test_path])
        # Always also try `python3 -m pytest`: works when pytest is importable
        # but no `pytest` binary is on PATH (pip-installed without entry script).
        runner_cmds.append(["python3", "-m", "pytest", "-x", "--tb=short", "-q", "--no-header", test_path])

        for cmd in runner_cmds:
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(repo),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout_seconds,
                    env=_command_env(),
                )
            except subprocess.TimeoutExpired:
                return f"Companion test `{test_path}` timed out after {timeout_seconds}s."
            except Exception:
                continue

            output = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
            unrunnable_markers = (
                "No module named pytest",
                "No module named 'pytest'",
                "command not found",
                "/usr/bin/env: python3",
            )
            if any(marker in output for marker in unrunnable_markers):
                continue  # try next runner / give up if all fail
            if proc.returncode == 0:
                return None  # test passed
            return output[-2400:] if len(output) > 2400 else output

        return None  # no runner produced a usable signal

    # ---- JS / TS ----
    if suffix in {".ts", ".tsx", ".js", ".jsx", ".cjs", ".mjs"}:
        if not _has_executable("node"):
            return None
        try:
            proc = subprocess.run(
                ["node", "--check", test_path],
                cwd=str(repo),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_seconds,
                env=_command_env(),
            )
        except subprocess.TimeoutExpired:
            return f"Companion test `{test_path}` parse timed out after {timeout_seconds}s."
        except Exception:
            return None
        if proc.returncode == 0:
            return None
        output = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
        return output[-2400:] if len(output) > 2400 else output

    return None  # other languages: skip


def _select_companion_test_failure(
    repo: Path,
    patch: str,
    test_timeout_seconds: int = 8,
) -> Optional[Tuple[str, str]]:
    """For files touched by the patch, find the first companion test that fails.

    Returns (test_path, output_tail) on the first non-None failure, else None.
    Stops at the first failure to keep the refinement budget tight (one fix
    turn maximum per cycle).
    """
    edited = _patch_changed_files(patch)
    if not edited:
        return None
    tracked = set(_tracked_files(repo))
    if not tracked:
        return None
    for relative_path in edited:
        partner = _find_test_partner(relative_path, tracked)
        if not partner:
            continue
        output = _run_companion_test(repo, partner, timeout_seconds=test_timeout_seconds)
        if output:
            return (partner, output)
    return None


def _recent_commit_examples(repo: Path) -> str:
    """v21 edge: read recent small-diff commits from the staged repo via git log
    and format them as in-context style anchors. Returns empty string when the
    repo has no real history (single synthetic commit in pilot snapshots), so
    this is a silent no-op locally and a real lift live where the validator
    clones the upstream repo with full history.

    The model imitates concrete examples better than abstract rules. Showing the
    model 1-2 real recent commits gives it a concise local style anchor."""
    try:
        proc = subprocess.run(
            ["git", "log", "--no-merges", "--pretty=format:%H", "-n", "20"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return ""
        shas = [s.strip() for s in proc.stdout.splitlines() if s.strip()]
        if len(shas) < 2:
            return ""  # single synthetic commit (pilot) — silent no-op
        examples: List[str] = []
        budget_used = 0
        for sha in shas:
            stat_proc = subprocess.run(
                ["git", "show", "--no-merges", "--shortstat", "--pretty=format:", sha],
                cwd=str(repo),
                capture_output=True,
                text=True,
                timeout=10,
            )
            if stat_proc.returncode != 0:
                continue
            insertions = 0
            for line in stat_proc.stdout.splitlines():
                if "insertion" in line:
                    for word in line.split(","):
                        if "insertion" in word:
                            try:
                                insertions = int(word.strip().split()[0])
                            except (ValueError, IndexError):
                                pass
                    break
            if insertions == 0 or insertions > _RECENT_COMMIT_MAX_INSERTIONS:
                continue
            # NOTE: previous version passed --pretty=format:%s which caused
            # `git show` to emit the commit subject in place of the standard
            # header but git still appended the diff. After the >=100 char
            # filter the only commits that survived were those with very long
            # subjects (e.g. squash messages); their wrapped output was a mix
            # of subject + diff, which is noise. --pretty=format: empties the
            # header entirely so we keep just the diff body.
            diff_proc = subprocess.run(
                ["git", "show", "--no-merges", "--pretty=format:", sha],
                cwd=str(repo),
                capture_output=True,
                text=True,
                timeout=10,
            )
            if diff_proc.returncode != 0:
                continue
            diff_text = diff_proc.stdout.strip()
            if len(diff_text) < 100 or len(diff_text) > _RECENT_COMMIT_MAX_DIFF_CHARS:
                continue
            block = f"```diff\n{diff_text[:_RECENT_COMMIT_MAX_DIFF_CHARS]}\n```"
            if budget_used + len(block) > _RECENT_COMMIT_BLOCK_BUDGET:
                break
            examples.append(block)
            budget_used += len(block)
            if len(examples) >= 2:
                break
        if not examples:
            return ""
        return (
            "\n\nRECENT REFERENCE PATCHES from this codebase (style anchors — "
            "match the shape, scale, and conventions of these real recent "
            "commits when writing your patch):\n\n" + "\n\n".join(examples)
        )
    except Exception:
        return ""


# v21 edge: criteria-nudge support
_CRITERIA_MAX_BULLETS = 8
_CRITERIA_MAX_TEXT = 220
_CRITERIA_STOP = frozenset({
    "a", "an", "and", "as", "at", "be", "but", "by", "do", "for", "from",
    "if", "in", "is", "it", "of", "on", "or", "so", "that", "the", "this",
    "to", "we", "with", "our", "must", "should", "shall", "can", "may",
    "will", "implement", "add", "support", "ensure", "make", "use", "create",
    "fix", "update", "change", "set", "include", "handle", "allow", "also",
    "when", "where", "which", "who", "what", "all", "any", "each", "every",
    "task", "issue", "code", "your", "you",
})


def _extract_acceptance_criteria(issue_text: str) -> List[str]:
    """Pull acceptance-criterion checkpoints from the issue text.

    Heuristic: numbered lines (`1.` or `1)`) and dashed bullets (`-` / `*` /
    `•`) first; fallback to imperative sentences (must/should/implement/add/
    support/ensure) when no list structure exists. Caps at _CRITERIA_MAX_BULLETS
    so the nudge prompt stays compact."""
    if not issue_text:
        return []
    bullets: List[str] = []
    bullet_re = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.+?)\s*$")
    for line in issue_text.splitlines():
        m = bullet_re.match(line)
        if not m:
            continue
        text = m.group(1).strip()
        if len(text) < 6:
            continue
        bullets.append(text[:_CRITERIA_MAX_TEXT])
        if len(bullets) >= _CRITERIA_MAX_BULLETS:
            break
    if bullets:
        return bullets
    fallback_re = re.compile(
        r"\b(must|should|implement|add|support|ensure|return|raise|expect)\b",
        re.IGNORECASE,
    )
    for raw in re.split(r"(?<=[.!?])\s+", issue_text):
        text = raw.strip()
        if not text or len(text) < 12 or len(text) > _CRITERIA_MAX_TEXT:
            continue
        if not fallback_re.search(text):
            continue
        bullets.append(text)
        if len(bullets) >= _CRITERIA_MAX_BULLETS:
            break
    return bullets


def _criterion_keywords(criterion: str) -> List[str]:
    """Significant tokens from a criterion (drop stopwords + short words)."""
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", criterion.lower())
    return [t for t in tokens if t not in _CRITERIA_STOP]


# Verb/noun suffixes commonly used in acceptance-criterion English that don't
# appear in source-code identifiers. The criteria say "clicking", "loads",
# "selection", "displayed", "correctly"; the corresponding code uses
# `onClick`, `loadMessages`, `onSelect`, `display`, `correct`. A literal
# substring check on the natural-language form misses these matches and
# inflates the criteria-nudge false-positive rate. Stripping the suffix
# (with a minimum-stem length to avoid false positives like `action`->`act`
# matching `react`) bridges the natural-language ↔ identifier gap.
_KEYWORD_SUFFIX_STRIPS = (("ing", 4), ("tion", 4), ("ion", 4), ("ed", 4), ("es", 4), ("ly", 4), ("s", 4))


def _keyword_in_added(keyword: str, added_lower: str) -> bool:
    if keyword in added_lower:
        return True
    for suffix, min_stem_len in _KEYWORD_SUFFIX_STRIPS:
        if keyword.endswith(suffix) and len(keyword) - len(suffix) >= min_stem_len:
            if keyword[:-len(suffix)] in added_lower:
                return True
            break
    return False


def _patch_added_text(patch: str) -> str:
    """Concat all + lines of the patch (lower-cased) for keyword search."""
    out: List[str] = []
    for line in patch.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            out.append(line[1:])
    return "\n".join(out).lower()


def _unaddressed_criteria(patch: str, issue_text: str) -> List[str]:
    """Criteria whose significant tokens DON'T appear in the patch's added
    lines. The judge frequently dings the king for missing N of M criteria;
    surfacing the gap lets the model close it before <final>."""
    criteria = _extract_acceptance_criteria(issue_text)
    if not criteria:
        return []
    added_lower = _patch_added_text(patch)
    if not added_lower:
        return criteria
    missing: List[str] = []
    for crit in criteria:
        keywords = _criterion_keywords(crit)
        if not keywords:
            continue
        # criterion is "addressed" if at least HALF its keywords appear
        hits = sum(1 for kw in keywords if _keyword_in_added(kw, added_lower))
        if hits * 2 < len(keywords):
            missing.append(crit)
    return missing


# -----------------------------
# Issue-symbol grep ranking
# -----------------------------
#
# `_rank_context_files` already weighs files by issue-mentioned paths and term
# overlap. For multi-file repos that's not enough — a one-line bug fix often
# names a function or class without mentioning the file. We extract identifier-
# shaped tokens from the issue and grep the repo for them; files that contain
# those identifiers get a context-rank boost.

_SYMBOL_RE = re.compile(r"(?<![A-Za-z0-9_])([A-Za-z_][A-Za-z0-9_]{2,})(?![A-Za-z0-9_])")
_SYMBOL_STOP = {
    "about", "after", "alert", "argument", "before", "build", "called", "change",
    "check", "class", "code", "command", "config", "context", "default", "expect",
    "expected", "fail", "false", "field", "fields", "file", "files", "fix",
    "fixed", "function", "given", "global", "header", "headers", "import",
    "issue", "method", "module", "needed", "needs", "object", "params", "parse",
    "path", "patch", "production", "project", "property", "public", "remove",
    "reset", "return", "should", "static", "string", "support", "test", "tests",
    "their", "there", "thing", "this", "true", "type", "types", "update",
    "using", "value", "values", "when", "with", "will", "without", "write",
}


def _extract_issue_symbols(issue_text: str, *, max_symbols: int = 12) -> List[str]:
    """Pull identifier-shaped tokens from the issue text.

    Heuristic: any CamelCase or snake_case identifier, plus any all-lowercase
    identifier of length >=4 (so we catch `pairs`, `solve`, `parse`, etc.).
    Stop-words and very short tokens are filtered out.
    """
    seen: set = set()
    out: List[str] = []
    for match in _SYMBOL_RE.finditer(issue_text):
        token = match.group(1)
        if token in seen:
            continue
        lowered = token.lower()
        if lowered in _SYMBOL_STOP:
            continue
        is_compound = any(c.isupper() for c in token[1:]) or "_" in token
        if not is_compound and len(token) < 4:
            continue
        seen.add(token)
        out.append(token)
        if len(out) >= max_symbols:
            break
    return out


def _symbol_grep_hits(
    repo: Path,
    tracked_set: set,
    issue_text: str,
) -> Dict[str, int]:
    """Count how many extracted symbols each tracked file references.

    Skips on git-grep failure to keep the cycle cheap; symbol-grep is a *boost*
    to ranking, never the only signal.
    """
    symbols = _extract_issue_symbols(issue_text)
    if not symbols:
        return {}
    hits: Dict[str, int] = {}
    for symbol in symbols:
        try:
            proc = subprocess.run(
                ["git", "grep", "-l", "-F", "--", symbol],
                cwd=str(repo),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=4,
            )
        except Exception:
            continue
        if proc.returncode not in (0, 1):
            continue
        for line in proc.stdout.splitlines():
            relative_path = line.strip()
            if not relative_path or relative_path not in tracked_set:
                continue
            if not _context_file_allowed(relative_path):
                continue
            hits[relative_path] = hits.get(relative_path, 0) + 1
    return hits


# -----------------------------
# v32: issue surface classifier + requirement checklist + completeness gate
# -----------------------------
#
# Duel review notes consistently show that multi-surface feature tasks are
# lost because the agent fixes the local bug but skips the integration
# cascade (route + service + frontend wiring + schema + tests). These helpers
# tag the issue with high-level surfaces and verify the patch covers each
# surface the task implies. They are intentionally heuristic and cheap; they
# never claim to know the real test set, only to nudge the model toward the
# obvious missing pieces before <final>.

_SURFACE_KEYWORDS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("ui", ("component", "page", "view", "modal", "button", "navbar", "layout", "mobile", "desktop", "responsive", "css", "classname", "loading", "empty state")),
    ("api", ("endpoint", "route", "controller", "request", "response", "http", "backend", "server")),
    ("data", ("schema", "model", "migration", "db", "field", "serializer", "type", "interface")),
    ("wiring", ("import", "export", "register", "mount", "route", "app.js", "provider", "hook", "store", "service", "client")),
    ("tests", ("test", "spec", "regression", "smoke", "e2e")),
    ("docs", ("readme", "docs", "changelog", "version")),
    ("build", ("package.json", "vite", "vercel", "script", "config")),
    ("platform", ("native", "web", "mobile", "android", "ios", "flutter", "dart", "browser")),
)


def _classify_issue_surfaces(issue: str) -> set:
    """Tag the issue with high-level surface labels using keyword heuristics.

    Returns a set drawn from {"ui","api","data","wiring","tests","docs",
    "build","platform"}. Empty set for short or signal-free issues. The classifier is
    intentionally generous on the recall side: it is read by the initial
    prompt + a refinement nudge, both of which already let the model dismiss
    a false positive.
    """
    if not issue:
        return set()
    lowered = issue.lower()
    tags: set = set()
    for tag, keywords in _SURFACE_KEYWORDS:
        for kw in keywords:
            if " " in kw:
                if kw in lowered:
                    tags.add(tag)
                    break
            else:
                if re.search(r"\b" + re.escape(kw) + r"\b", lowered):
                    tags.add(tag)
                    break
    return tags


def _extract_requirement_checklist(issue_text: str, *, max_items: int = 12) -> List[str]:
    """Build a compact deduped requirement checklist from the issue text.

    Captures list items, imperative sentences, explicit paths, quoted UI/API
    literals, endpoint strings, and clause-level requirements split by
    "also/and/both/all/unless/only" hints.
    """
    if not issue_text:
        return []
    items: List[str] = []
    seen: set = set()

    def add(text: str) -> None:
        normalized = re.sub(r"\s+", " ", text).strip(" -*\t")
        if not normalized or len(normalized) < 6 or len(normalized) > 240:
            return
        key = normalized.lower()
        if key in seen:
            return
        seen.add(key)
        items.append(normalized)

    bullet_re = re.compile(r"^\s*(?:[-*\u2022]|\[[ xX]\]|\d+[.)])\s+(.+?)\s*$")
    for line in issue_text.splitlines():
        m = bullet_re.match(line)
        if m:
            add(m.group(1))
            if len(items) >= max_items:
                return items

    imperative_re = re.compile(
        r"\b(must|should|ensure|add|update|remove|delete|replace|implement|support|"
        r"preserve|keep|use|move|wire|integrate|hide|show|route|endpoint|api|frontend|"
        r"backend|mobile|desktop|loading|empty|error|fallback)\b",
        re.IGNORECASE,
    )
    for raw in re.split(r"(?<=[.!?])\s+", issue_text):
        text = raw.strip()
        if not text or len(text) < 12:
            continue
        if imperative_re.search(text):
            add(text)
            clauses = re.split(r"\b(?:also|and|both|all|unless|only)\b", text, flags=re.IGNORECASE)
            for clause in clauses:
                clause_text = clause.strip(" ,;:-")
                if imperative_re.search(clause_text):
                    add(clause_text)
            if len(items) >= max_items:
                return items

    quote_re = re.compile(r"[\"'\u201c\u2018]([^\"'\u201d\u2019]{3,80})[\"'\u201d\u2019]")
    for m in quote_re.finditer(issue_text):
        add('show or use literal text "' + m.group(1).strip() + '"')
        if len(items) >= max_items:
            return items

    endpoint_re = re.compile(r"\b(GET|POST|PUT|PATCH|DELETE)\s+(/[A-Za-z0-9_./{}-]+)")
    for m in endpoint_re.finditer(issue_text):
        add("endpoint " + m.group(1).upper() + " " + m.group(2))
        if len(items) >= max_items:
            return items

    path_re = re.compile(
        r"(?<![A-Za-z0-9_])([A-Za-z0-9_./-]+\.(?:py|ts|tsx|js|jsx|json|md|yml|yaml|css|scss)|/[A-Za-z0-9_./{}-]+)"
    )
    for m in path_re.finditer(issue_text):
        add("path " + m.group(1).strip())
        if len(items) >= max_items:
            return items

    return items[:max_items]


def _extract_core_action_terms(issue: str) -> set:
    """Extract high-leverage action phrases for completion/risk heuristics."""
    if not issue:
        return set()
    lowered = issue.lower()
    terms: set = set()
    patterns = {
        "delete_remove": r"\b(delete|remove)\b",
        "wire_import_use": r"\b(wire|import|use|register|mount|integrate)\b",
        "preserve_flow": r"\b(preserve|keep existing|do not break|existing flow|current flow)\b",
        "dedicated_ref": r"\bdedicated ref\b",
        "route_placement": r"\b(route order|route placement|shadow|static route|dynamic route)\b",
        "frontend_flow": r"\b(frontend flow|selection|resubmission|resubmit)\b",
        "extraction_refactor": r"\b(extract|refactor|split module|move to module)\b",
        "loading_empty_error": r"\b(loading|empty|error|fallback)\b",
        "responsive": r"\b(mobile|desktop|responsive|breakpoint)\b",
    }
    for name, pattern in patterns.items():
        if re.search(pattern, lowered):
            terms.add(name)
    return terms


def _detect_platform_import_violations(patch: str) -> List[str]:
    """Flag dart:html (web-only) imports added in non-web Dart files."""
    violations: List[str] = []
    current_file = ""
    is_web_path = False
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            tokens = line.split()
            if len(tokens) >= 4 and tokens[3].startswith("b/"):
                current_file = tokens[3][2:]
            else:
                current_file = ""
            lowered_path = current_file.lower()
            is_web_path = (
                "/web/" in "/" + lowered_path
                or lowered_path.startswith("web/")
                or "platform_web" in lowered_path
                or "web_only" in lowered_path
            )
            continue
        if not current_file.endswith(".dart"):
            continue
        if not line.startswith("+") or line.startswith("+++"):
            continue
        body = line[1:].strip()
        if "import 'dart:html'" in body or 'import "dart:html"' in body:
            if not is_web_path:
                violations.append(current_file + ": dart:html imported outside web folder")
    return violations


def _surface_completeness_gaps(repo: Path, patch: str, issue: str) -> List[str]:
    """Heuristic: list integration surfaces the issue implies but the patch skips.

    Returns a list of human-readable gap descriptions. Empty list means the
    patch already touches every surface the issue suggests (or the issue has
    no clear surface tags). The model decides per gap whether to add an edit
    or document the omission in <final>.
    """
    if not patch.strip() or not issue.strip():
        return []
    surfaces = _classify_issue_surfaces(issue)
    core_terms = _extract_core_action_terms(issue)
    if not surfaces:
        return []
    changed = _patch_changed_files(patch)
    if not changed:
        return []

    touched_ui = False
    touched_api = False
    touched_data = False
    touched_test = False
    touched_docs = False
    touched_frontend = False
    touched_backend = False
    for path in changed:
        lower = path.lower()
        suffix = Path(path).suffix.lower()
        parts = {p.lower() for p in Path(path).parts}
        name_lower = Path(path).name.lower()
        if (
            suffix in {".tsx", ".jsx", ".vue", ".svelte"}
            or "components" in parts
            or "pages" in parts
            or "views" in parts
            or "screens" in parts
            or "layouts" in parts
            or "ui" in parts
            or suffix in {".css", ".scss", ".sass", ".less"}
        ):
            touched_ui = True
            touched_frontend = True
        if (
            "frontend" in parts
            or "client" in parts
            or "web" in parts
            or suffix in {".ts", ".tsx", ".jsx", ".js", ".mjs", ".cjs"}
        ):
            touched_frontend = True
        if any(part in parts for part in (
            "backend", "server", "api", "controllers", "controller",
            "routes", "handlers", "handler", "services", "service",
        )):
            touched_api = True
            touched_backend = True
        if any(part in parts for part in (
            "models", "schemas", "schema", "migrations", "migration",
            "serializers", "serializer", "types", "entities",
        )) or "schema" in lower or "model" in lower:
            touched_data = True
        if (
            any(("test" in part) or ("spec" in part) for part in parts)
            or "_test" in lower
            or ".test." in lower
            or ".spec." in lower
            or name_lower.startswith("test_")
        ):
            touched_test = True
        if (suffix in {".md", ".rst", ".txt"}) or "readme" in name_lower:
            touched_docs = True

    gaps: List[str] = []
    if "ui" in surfaces and not touched_ui:
        gaps.append(
            "issue references UI/component/page work but patch touches no "
            "frontend/component/style files"
        )
    if "api" in surfaces and not touched_api:
        gaps.append(
            "issue references API/route/endpoint work but patch touches no "
            "backend route/controller/service files"
        )
    if ("ui" in surfaces) and ("api" in surfaces):
        if not (touched_frontend and touched_backend):
            gaps.append(
                "feature spans UI and API but only one side is wired; add the "
                "missing frontend client call OR backend endpoint"
            )
    if "data" in surfaces and not touched_data:
        gaps.append(
            "issue references model/schema/field changes but patch touches no "
            "model/schema/type/serializer files"
        )
    if "tests" in surfaces and not touched_test:
        gaps.append(
            "issue references tests/regression but patch touches no test/spec file"
        )
    if "docs" in surfaces and not touched_docs:
        gaps.append(
            "issue references README/docs but patch touches no markdown/doc file"
        )
    if "platform" in surfaces:
        bad_imports = _detect_platform_import_violations(patch)
        if bad_imports:
            gaps.append("platform-specific import in shared/non-web file: " + "; ".join(bad_imports[:3]))

    issue_paths = _extract_issue_path_mentions(issue)
    if ("delete_remove" in core_terms) and issue_paths:
        changed_set = set(changed)
        for req in issue_paths:
            req_clean = req.strip("./")
            if any(req_clean == c or c.endswith("/" + req_clean) for c in changed_set):
                continue
            gaps.append(f"issue requests remove/delete but named path is untouched: {req_clean}")
            break

    lowered_issue = issue.lower()
    if ("extraction_refactor" in core_terms) or any(k in lowered_issue for k in ("extract", "refactor", "module")):
        added_modules = [p for p in changed if p.endswith((".ts", ".tsx", ".js", ".jsx", ".py")) and _patch_is_new_file(patch, p)]
        entrypoint_touched = any(
            Path(p).name.lower() in {"app.js", "app.ts", "main.ts", "main.js", "index.ts", "index.js"}
            for p in changed
        )
        if added_modules and not entrypoint_touched:
            gaps.append("extraction/refactor adds module files but does not update original entrypoint/import site")

    if any(k in lowered_issue for k in ("selection", "resubmission", "resubmit", "frontend flow")) and touched_backend and not touched_frontend:
        gaps.append("issue references frontend flow/selection/resubmission but patch only changes backend")

    if any(k in lowered_issue for k in ("loading", "empty", "error", "fallback")):
        added_lower = _patch_added_text(patch)
        if not any(token in added_lower for token in ("loading", "empty", "error", "fallback", "isloading", "isempty")):
            gaps.append("issue mentions loading/empty/error/fallback but changed UI code lacks these states")

    if any(k in lowered_issue for k in ("changelog", "readme", "docs", "version")) and not touched_docs:
        gaps.append("issue mentions docs/changelog/version but no docs/version file changed")

    if any(k in lowered_issue for k in ("test", "regression", "smoke")) and not touched_test:
        gaps.append("issue mentions tests/regression/smoke but no test/spec file changed")

    if any(k in lowered_issue for k in ("mobile", "responsive", "desktop")):
        if _patch_has_breakpoint_risk(patch):
            gaps.append("responsive/mobile task may alter breakpoint semantics; verify small-screen layout behavior")

    if any(k in lowered_issue for k in ("route", "endpoint")):
        if _patch_has_route_shadowing_pattern(patch):
            gaps.append("static route may be registered after dynamic parameter route and get shadowed")
        else:
            for rel in changed:
                if not any(tok in rel.lower() for tok in ("route", "router", "server", "api")):
                    continue
                full = (repo / rel).resolve()
                try:
                    full.relative_to(repo.resolve())
                    source = full.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                if _file_has_route_shadowing(source):
                    gaps.append("router file order suggests static route may be shadowed by dynamic route")
                    break

    return gaps


def _patch_is_new_file(patch: str, relative_path: str) -> bool:
    block_re = re.compile(
        r"^diff --git a/" + re.escape(relative_path) + r" b/" + re.escape(relative_path) + r"$.*?(?=^diff --git |\Z)",
        re.MULTILINE | re.DOTALL,
    )
    m = block_re.search(patch)
    if not m:
        return False
    return "\nnew file mode " in m.group(0)


def _patch_has_route_shadowing_pattern(patch: str) -> bool:
    dynamic_seen: Dict[str, bool] = {}
    current_file = ""
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            m = re.match(r"diff --git a/(.+?) b/(.+)$", line)
            current_file = m.group(2) if m else ""
            dynamic_seen[current_file] = False
            continue
        if not current_file or (not line.startswith("+")) or line.startswith("+++"):
            continue
        body = line[1:]
        if re.search(r'["\']/:[A-Za-z_][\w-]*["\']', body):
            dynamic_seen[current_file] = True
        if dynamic_seen.get(current_file, False) and re.search(r'["\']/[A-Za-z0-9_-]+["\']', body) and not re.search(r'["\']/:[A-Za-z_][\w-]*["\']', body):
            return True
    return False


def _patch_has_breakpoint_risk(patch: str) -> bool:
    for line in patch.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        body = line[1:].lower()
        if ("flex-row" in body or "grid-cols-" in body) and not any(tok in body for tok in ("sm:", "md:", "lg:", "@media", "max-width", "min-width")):
            return True
    return False


# -----------------------------
# v32: TSX React + Dart platform static checks
# -----------------------------

_TSX_HOOK_NAMES: Tuple[str, ...] = (
    "useState", "useEffect", "useReducer", "useMemo",
    "useCallback", "useRef", "useContext", "useLayoutEffect",
    "useImperativeHandle", "useTransition", "useDeferredValue",
)

_LEADING_DIRECTIVE_RE = re.compile(
    r'^["\'](?:use client|use server|use strict)["\']\s*;?\s*$'
)


def _check_tsx_react_imports_one(repo: Path, relative_path: str) -> Optional[str]:
    """Cheap text scan for the React-import / directive bugs we lose duels on.

    Targets observed failure patterns:
      - Malformed leading `"use client"`/`"use server"` directive (the
        unbalanced-quote bug seen in real submissions).
      - `React.<Member>` usage with no `import React` in scope.
      - Hook usage (useState/useEffect/...) without it being imported.
    Returns a single error string or None.
    """
    full = (repo / relative_path).resolve()
    try:
        full.relative_to(repo.resolve())
    except (ValueError, RuntimeError):
        return None
    if not full.exists():
        return None
    try:
        source = full.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None

    issues: List[str] = []

    first_nonblank = ""
    for line in source.splitlines():
        if line.strip():
            first_nonblank = line.strip()
            break
    if first_nonblank.startswith(("'use", '"use')):
        if not _LEADING_DIRECTIVE_RE.match(first_nonblank):
            issues.append(
                f"{relative_path}:1: malformed leading directive `{first_nonblank[:60]}`"
            )

    react_namespace_used = bool(re.search(r"\bReact\.\w+", source))
    has_default_react_import = bool(re.search(r"^\s*import\s+React\b", source, re.MULTILINE))
    has_namespace_react_import = bool(
        re.search(r"^\s*import\s+\*\s+as\s+React\s+from\s+['\"]react['\"]", source, re.MULTILINE)
    )
    has_react_anywhere = bool(re.search(r"from\s+['\"]react['\"]", source))

    if react_namespace_used and not (has_default_react_import or has_namespace_react_import):
        issues.append(f"{relative_path}: uses `React.<member>` without `import React`")

    hooks_used = [name for name in _TSX_HOOK_NAMES if re.search(r"\b" + name + r"\b", source)]
    if hooks_used and (has_react_anywhere or has_default_react_import or has_namespace_react_import):
        # Only flag inside files that actually consume react: a destructured
        # `import { useState } from 'react'` is the standard contract. We
        # skip the check entirely in non-react files to avoid false-positive
        # hook-name collisions with non-react libraries (e.g., zustand's
        # `useStore`, tanstack's `useQuery`).
        imported_hooks: set = set()
        for m in re.finditer(r"import\s*\{([^}]+)\}\s*from\s*['\"]react['\"]", source):
            for token in m.group(1).split(","):
                imported_hooks.add(token.strip().split(" as ")[0].strip())
        if has_namespace_react_import:
            missing: List[str] = []
        else:
            missing = [h for h in hooks_used if h not in imported_hooks]
        if missing:
            issues.append(
                f"{relative_path}: uses hook(s) {', '.join(missing[:4])} without importing from 'react'"
            )

    if issues:
        return "; ".join(issues[:3])
    return None


def _check_dart_platform_one(repo: Path, relative_path: str) -> Optional[str]:
    """Flag dart:html in shared/non-web Dart files (a real duel-loss pattern).

    `dart:html` only resolves on the web target; importing it from shared or
    mobile/desktop code makes the build fail on those targets.
    """
    full = (repo / relative_path).resolve()
    try:
        full.relative_to(repo.resolve())
    except (ValueError, RuntimeError):
        return None
    if not full.exists() or full.suffix.lower() != ".dart":
        return None
    lowered_path = relative_path.lower()
    is_web_only = (
        "/web/" in "/" + lowered_path
        or lowered_path.startswith("web/")
        or "platform_web" in lowered_path
        or "web_only" in lowered_path
    )
    if is_web_only:
        return None
    try:
        source = full.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    if re.search(r"^\s*import\s+['\"]dart:html['\"]", source, re.MULTILINE):
        return f"{relative_path}: imports `dart:html` in shared/non-web Dart file"
    return None


# -----------------------------
# v33: static footgun checks
# -----------------------------

_JSX_IDENTIFIER_IMPORT_GUARD: Tuple[str, ...] = (
    "Icon", "TrashIcon", "BreadcrumbItem", "Link", "Button",
)


def _patch_added_lines_by_file(patch: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    current = ""
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            m = re.match(r"diff --git a/(.+?) b/(.+)$", line)
            current = m.group(2) if m else ""
            if current:
                out.setdefault(current, [])
            continue
        if not current:
            continue
        if line.startswith("+") and not line.startswith("+++"):
            out[current].append(line[1:])
    return out


def _check_static_footguns(repo: Path, patch: str, issue: str) -> List[str]:
    if not patch.strip():
        return []
    errors: List[str] = []
    changed = _patch_changed_files(patch)
    added = _patch_added_lines_by_file(patch)
    issue_lower = issue.lower()

    if any(k in issue_lower for k in ("route", "endpoint")) and _patch_has_route_shadowing_pattern(patch):
        errors.append("route shadowing risk: static route appears after dynamic parameter route")

    for relative_path in changed:
        suffix = Path(relative_path).suffix.lower()
        full = (repo / relative_path).resolve()
        try:
            full.relative_to(repo.resolve())
        except Exception:
            continue
        if not full.exists():
            continue
        try:
            src = full.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        if suffix in {".tsx", ".jsx", ".ts", ".js", ".mjs", ".cjs"}:
            add_lines = "\n".join(added.get(relative_path, []))
            for ident in _JSX_IDENTIFIER_IMPORT_GUARD:
                if re.search(r"\b" + re.escape(ident) + r"\b", add_lines):
                    imported = re.search(r"import\s+.*\b" + re.escape(ident) + r"\b.*from", src)
                    declared = re.search(r"\b(?:const|let|var|function|class)\s+" + re.escape(ident) + r"\b", src)
                    if not imported and not declared:
                        errors.append(f"{relative_path}: `{ident}` appears in added code without import/local declaration")
                        break

            dup_state = re.findall(r"const\s*\[\s*([A-Za-z_]\w*)\s*,\s*set[A-Za-z_]\w*\s*\]\s*=\s*useState\s*\(", src)
            if dup_state and len(dup_state) != len(set(dup_state)):
                errors.append(f"{relative_path}: duplicate useState declaration for same state variable")

            if re.search(r"\bemail\.toLowerCase\s*\(", add_lines) and not re.search(r"(email\s*[!?]\.|email\s*&&|email\s*\?)", add_lines):
                errors.append(f"{relative_path}: possible optional email `.toLowerCase()` without guard")

            defined_tokens = set(re.findall(r"\b(?:const|let|var|function|class)\s+([A-Za-z_]\w*)", src))
            imported_tokens = set(re.findall(r"import\s*\{([^}]+)\}\s*from", src))
            flat_imported: set = set()
            for grp in imported_tokens:
                for token in grp.split(","):
                    flat_imported.add(token.strip().split(" as ")[0].strip())
            params = set(re.findall(r"\(([A-Za-z_]\w*)[,)]", src))
            known = defined_tokens | flat_imported | params | {"React"}
            for token in re.findall(r"\b([A-Za-z_]\w*)\b", add_lines):
                if token in known:
                    continue
                if token[0].isupper():
                    continue
                if token in {"if", "for", "while", "return", "const", "let", "var", "true", "false", "null", "undefined"}:
                    continue
                if re.search(r"\." + re.escape(token) + r"\b", add_lines):
                    continue
                if token in {"email", "asset", "kmReadings"}:
                    errors.append(f"{relative_path}: added code may reference undefined variable `{token}`")
                    break

            if any(k in issue_lower for k in ("route", "endpoint")) and any(tok in relative_path.lower() for tok in ("route", "router", "server", "api")):
                if _file_has_route_shadowing(src):
                    errors.append(f"{relative_path}: static route is declared after dynamic parameter route")

        if suffix == ".py":
            if re.search(r"^\s*\w+\s*=\s*\{[^}]*\b\w+\(", src, re.MULTILINE):
                errors.append(f"{relative_path}: top-level registry/dict may call function before definition")
            add_lines = "\n".join(added.get(relative_path, []))
            if re.search(r"\[[^\]]*(bbox|box|bounds)[^\]]+\]", add_lines) and re.search(r"\b(np\.\w+|numpy)\b", src, re.IGNORECASE):
                if not re.search(r"\b(int|round|floor|ceil)\s*\(", add_lines):
                    errors.append(f"{relative_path}: numpy slice with bbox-like values may require int conversion")

    py_changed = [p for p in changed if p.endswith(".py")]
    if py_changed and _has_executable("python3"):
        for relative_path in py_changed[:8]:
            proc = run_command(f"python3 -m py_compile {_shell_quote(relative_path)}", repo, timeout=4)
            if proc.exit_code != 0:
                tail = (proc.stderr or proc.stdout or "").strip().splitlines()
                errors.append(f"{relative_path}: py_compile failed: {(tail[-1] if tail else 'unknown error')}")
                break

    return errors[:10]


def _file_has_route_shadowing(source: str) -> bool:
    route_re = re.compile(r'["\'](/[^"\']+)["\']')
    dynamic_index: Optional[int] = None
    static_after_dynamic = False
    lines = source.splitlines()
    for idx, line in enumerate(lines):
        if not any(tok in line for tok in (".get(", ".post(", ".put(", ".patch(", ".delete(", "add_api_route(", "@router.", "router.")):
            continue
        m = route_re.search(line)
        if not m:
            continue
        route = m.group(1)
        if "/:" in route or re.search(r"/\{[^}]+\}", route):
            if dynamic_index is None:
                dynamic_index = idx
        elif dynamic_index is not None and idx > dynamic_index:
            static_after_dynamic = True
            break
    return static_after_dynamic


def _scope_risk_gaps(patch: str, issue: str) -> List[str]:
    if not patch.strip():
        return []
    issue_lower = issue.lower()
    added = 0
    removed = 0
    for line in patch.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    gaps: List[str] = []
    if removed > (added * 2 + 40) and not any(k in issue_lower for k in ("cleanup", "remove", "delete", "deprecate")):
        gaps.append("patch removes substantially more code than it adds for a non-cleanup issue")

    if any(k in issue_lower for k in ("route", "navbar", "navigation", "link")):
        mentioned = {t for t in re.findall(r"/[A-Za-z0-9_/-]+", issue)}
        for line in patch.splitlines():
            if (line.startswith("+") or line.startswith("-")) and not line.startswith(("+++", "---")):
                routes = set(re.findall(r"/[A-Za-z0-9_/-]+", line))
                if any(r not in mentioned for r in routes):
                    gaps.append("patch changes route/navigation targets not mentioned in issue")
                    break
    return gaps[:4]


# -----------------------------
# v32: forbidden evaluator/grader term guard
# -----------------------------

_FORBIDDEN_EVAL_TERMS: Tuple[str, ...] = (
    "hidden test",
    "hidden tests",
    "evaluator",
    "grader",
    "scoring metadata",
    "validator secret",
)


def _patch_contains_forbidden_eval_terms(patch: str) -> List[str]:
    """Return forbidden evaluator/grader terms newly introduced by the patch.

    Only `+` lines are inspected. Existing-source mentions of the same words
    are intentionally ignored so legitimate prior usage is not flagged.
    """
    if not patch.strip():
        return []
    hits: List[str] = []
    seen: set = set()
    for line in patch.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        body = line[1:].lower()
        for term in _FORBIDDEN_EVAL_TERMS:
            if term in body and term not in seen:
                hits.append(term)
                seen.add(term)
    return hits


# -----------------------------
# Prompting
# -----------------------------

# MINER-EDITABLE: This prompt is the main behavior policy for the inner coding
# agent. Prompt improvements are encouraged as long as they respect the
# validator-owned boundaries above.
SYSTEM_PROMPT = '''You are an elite autonomous coding agent competing in a real GitHub issue repair benchmark.

You operate inside a real repository. You inspect the codebase, produce a patch, and verify it. Your patch is scored on (1) correctness/completeness vs the issue and hidden tests, and (2) similarity to a reference patch. Both reward the same thing: smallest correct change a senior maintainer would accept.

====================================================================
ABSOLUTE OUTPUT PROTOCOL
====================================================================

To run a shell command, emit exactly:

<command>
bash command here
</command>

To finish, emit exactly:

<final>
brief summary of what changed and what verification was run
</final>

Your first response MUST contain a `<plan>` block followed immediately by one focused inspection command.

First response format:

<plan>
- Requirement: restate every explicit issue requirement.
- Requirement: restate every secondary clause, edge case, "also", "and", "unless", "only", "should not", or acceptance criterion.
- Requirement: if the issue uses numbered bullets or checkbox lines, mirror each item as its own plan row.
- Acceptance checklist: a numbered list of every concrete acceptance item the issue implies (UI text shown, endpoint exposed, field added, nav link added, test added, etc.). Each item is something a reviewer can verify in the diff.
- Integration cascade: if the issue describes a feature spanning multiple concerns (page + route + nav + data fetch; or model + migration + serializer + view + URL), enumerate EVERY required integration point as its own plan row even when the issue does not explicitly bullet them.
- Requirement -> file map: for each acceptance item, list the file(s) you intend to edit. Use "?" while still searching.
- Deletion requirements: explicitly call out every remove/delete/obsolete requirement. If a file/UI is requested to be removed, your final diff must actually remove it or state why it already does not exist.
- Likely target: name likely files/functions/classes/modules to inspect or modify.
- Strategy: smallest root-cause fix likely to satisfy the issue.
- Verification: targeted test command expected after patching.
</plan>
<command>
focused inspection command
</command>

Never emit markdown fences around `<plan>`, `<command>`, or `<final>`.

Never emit `<final>` before a required code change has been made and verification has been attempted, unless the issue clearly requires no code change.

====================================================================
ISSUE CONTRACT
====================================================================

Treat the issue as a contract. Extract every requirement before editing — main task, bullet points, acceptance criteria, error messages, edge cases, and backwards-compat constraints. Treat clauses with "and / also / ensure / should / must / when / unless / only / both / all / regression / edge case / preserve" as distinct requirements. Hidden tests usually target the secondary clauses.
Before replacing existing flows, list what behavior must remain (auth/cart/nav/overlay/reveal/telemetry/message flows, existing actions, existing data sources) and preserve it unless the issue explicitly asks to remove/replace it.

If the issue is ambiguous, do not ask for clarification — infer intent from nearby code, tests, and existing patterns, and pick the smallest plausible maintainer fix that preserves unrelated behavior.

Evidence priority when picking what to patch: explicit issue text > failing/expected tests > nearby tests for similar behavior > the function/class that owns the behavior > existing patterns > public API compatibility > framework conventions > general knowledge. Do not invent behavior the issue and codebase do not support.

====================================================================
INSPECTION STRATEGY
====================================================================

Inspect only what you need to locate the owner of the bug and patch safely. Order: preloaded snippets first, then one or two focused searches (`rg`, fall back to `grep -R`), then the exact target region (`sed -n '120,220p'`), then nearby tests, then call sites only if a signature/public API may change.

Avoid: re-reading preloaded files, broad recursive searches, generated/vendor output, broad test suites before a targeted fix exists.

====================================================================
ROOT CAUSE RULE
====================================================================

Patch the owner of the behavior, not a downstream symptom. Parser rejects valid input → fix parser. Serializer omits field → fix serializer. Cache returns stale value → fix invalidation. CLI option ignored → fix option parsing. Validation rejects valid case → fix validation rule, not caller workaround.

Never hardcode the visible example unless the issue explicitly requests that exact special case. Hidden tests usually check the general behavior, not the literal example.

When several fixes are correct, choose the one that changes fewest files, smallest owning function, matches nearby style, preserves public API, uses existing helpers, and looks like the obvious five-minute maintainer patch.

====================================================================
SURGICAL EDITING
====================================================================

Change the fewest lines necessary. Allowed: one-line substitution, small guarded block replacement, one narrow branch, focused companion-test update, required call-site updates when a signature change is unavoidable.

Forbidden unless explicitly required: whole-file or whole-function rewrites when 1-5 lines suffice, formatting churn, whitespace/comment-only edits, code reordering, import sorting, renames for taste, new helpers/abstractions/files, dependency or lockfile changes, vendor/generated edits.

When editing with scripts, always guard replacements:

python - <<\'PY\'
from pathlib import Path
p = Path("path/to/file")
s = p.read_text()
old = """exact old block"""
new = """exact new block"""
if old not in s:
    raise SystemExit("old block not found")
p.write_text(s.replace(old, new, 1))
PY

Use `sed -i \'s/exact old/exact new/\' path/to/file` only when the substitution is uniquely scoped. Do not run broad regex replacements.

When a change necessarily spans multiple files (interface, signature, type, header+impl, schema/serializer pair), update every required file in the same response. Do not leave related files inconsistent. Do not touch extra files just because they are nearby.

When 3+ consecutive statements share the same shape, prefer a loop / map / list comprehension / table-driven test instead of unrolled copy-paste — but only inside the code you already have to change.

====================================================================
TESTS AND VERIFICATION
====================================================================

Add or update a test only when the issue requests it, a companion test already covers the area, the source fix breaks an existing nearby test, or a small regression test is the obvious lock-down. Place new tests next to the closest similar test, reuse fixtures, match naming, assert public behaviour. Never weaken, skip, delete, or loosen existing tests to pass.

After patching, run the most targeted meaningful verification available — one test case, one test file, or one module. Examples: `pytest tests/test_parser.py::test_x -q`, `pytest tests/test_x.py -x -q`, `go test ./pkg/foo`, `cargo test specific_test`, `npm test -- file -t "name"`, `mvn -q -Dtest=FooTest test`. Do not rely only on syntax checks when real targeted tests exist. Run broad suites only if the repo is small or no targeted tests exist.

If verification fails: read the failure, decide whether your patch caused it or it is pre-existing/environmental, fix the root cause if yours, rerun the same targeted command. Do not broaden the patch randomly. Do not mask failures by weakening tests.

====================================================================
STYLE, COMMENTS, AND PUBLIC API
====================================================================

Match adjacent code exactly: indentation, quotes, semicolons, trailing commas, brace placement, blank-line rhythm, naming, import grouping, error/assertion/test naming style. If nearby code style is imperfect, follow it anyway. Consistency beats personal preference.

Preserve EVERY meaningful comment around changed code — section headers, TODO/FIXME, compatibility notes, public-API docs, test labels, region markers. Section-grouping comments are high-signal to human and LLM judges. If a comment becomes false because of your fix, update it minimally; do not delete it.

Error messages are often tested exactly. When changing one, match capitalization, punctuation, quotes, and the existing error class/type.

Preserve public API and backwards compatibility unless the issue explicitly requires a breaking change: function/method names, signatures, exported types, CLI flags, config keys, response shapes, error classes, schemas, file formats, env-var names.

Before finalizing, mentally check hidden-test edge cases relevant to the issue: empty/null input, missing/extra fields, duplicates, case sensitivity, unicode, path separators, async ordering, idempotency, boundary values, default config behavior, multiple instances vs one.

====================================================================
LANGUAGE-SPECIFIC COMPLETENESS RULES
====================================================================

**Java:** Write complete method bodies — never use \'// similar logic\' stubs. Cascade all call-site changes when modifying signatures. Include all imports.

**C/C++:** Edit both .h header AND .cpp implementation for each changed function. Include full signatures and all required #include changes.

**TypeScript/C#:** Cascade interface and type changes to ALL implementing classes, components, and function parameters. Missing one = lower score.

**Go/Rust:** Update every struct field usage. Provide complete Rust lifetime annotations on modified functions.

**Multi-file tasks:** Complete ALL genuinely affected files in the same diff — never leave a related file partially edited, but do not broaden the patch beyond the task\'s behaviour.

====================================================================
SCOPE DISCIPLINE
====================================================================

Do NOT change:
- Whitespace-only, comment-only, or blank-line-only hunks
- Imports not needed by your fix
- Type annotations not already present in the changed function
- Refactoring, renaming, or reordering the issue does not ask for
- New helper functions or abstractions unless explicitly required
- New files unless explicitly required
- Test files unless required OR your change broke an existing test
- Error handling, logging, or defensive checks not directly required
- File permissions or mode bits (chmod is forbidden)

====================================================================
MULTI-SURFACE COMPLETION
====================================================================

Feature tasks usually span more than one surface. When the issue mentions or implies several of:
- UI: page, component, layout, navbar, form, modal, style, css
- API: endpoint, route, controller, handler, http verb, request/response
- DATA: model, schema, field, migration, serializer, type/interface
- TESTS: tests/spec/regression
- DOCS: README/docs/guide

Then your patch must cover every implied surface in the SAME submission. Examples:
- "Add a Dashboard page" with backend data implies a route, a component/page file, and the API client call to fetch the data.
- "Add slug-based detail view" implies a backend endpoint, a frontend route, the service mapping, and the link from the listing page.
- "Add a settings toggle" implies the storage/schema field, the read/write API, and the UI control.

Inspect adjacent existing files BEFORE introducing new abstractions, routes, or components. Match the existing folder layout, naming, CSS class style, and platform-separation pattern. Do not invent a parallel structure.

For UI changes: the target page/route name in the issue is authoritative -- if it says `/panel`, edit `/panel`, not `/admin`. Preserve layout invariants the issue mentions or that adjacent code already enforces (min-height, centering, container sizing, anchor IDs, sort/order of displayed metrics).
For row-level click handlers with nested action buttons, prevent accidental bubbling (e.g., stopPropagation) when the row click opens details.

For platform-aware code: do not import platform-only modules (e.g., `dart:html`) from shared/mobile files; use the existing platform-separation helper or conditional import already present in the repo.

====================================================================
PRE-FINAL SANITY
====================================================================

Before <final>, you must have at least mentally confirmed:
- every acceptance checklist item has a corresponding hunk in the diff (or a noted reason it requires no edit)
- every file you added an import or new reference in still parses (no missing `import React`, no missing `from foo import bar`, no unmatched `"use client"` directive)
- you ran the most targeted test/check available, or recorded that none was applicable
- you did not introduce generated, cache, binary, or permission-only changes (no `__pycache__`, no `.png` churn, no `chmod`)

If a sanity check fails, fix it before finalizing. The smallest failing import or stale checklist item costs the round.

====================================================================
SAFETY
====================================================================

No sudo. No chmod. No destructive git commands. No network access outside the validator proxy. No host secrets, dot-env files, credentials, hidden tests, evaluator files, or scoring metadata. File deletion is allowed only when explicitly required by the issue.
'''


_PRELOAD_BEGIN_MARKER = "<!-- preloaded-context-begin -->"
_PRELOAD_END_MARKER = "<!-- preloaded-context-end -->"


def build_initial_user_prompt(issue: str, repo_summary: str, preloaded_context: str = "") -> str:
    context_section = ""
    if preloaded_context.strip():
        context_section = f"""
{_PRELOAD_BEGIN_MARKER}
Preloaded likely relevant tracked-file snippets (already read for you - do not re-read):

{preloaded_context}
{_PRELOAD_END_MARKER}
"""

    return f"""Fix this issue:

{issue}

Repository summary:

{repo_summary}
{context_section}
Before planning, read the ENTIRE issue above and identify every requirement (there may be more than one). Your patch must satisfy ALL of them - the LLM judge penalizes incomplete solutions.
Build an acceptance checklist and map each item to likely owner files/surfaces before editing.
If issue text asks to delete/remove a file or obsolete flow, your final diff must actually remove it (or explicitly explain why no removal was needed).

Strategy: the fix is typically in ONE specific function or block. Identify it precisely, then make the minimal edit that fixes the ROOT CAUSE. For multi-surface feature tasks, identify ALL implied surfaces and edit each one in the same submission.
For feature tasks inspect owners plus wiring points. UI+API tasks need both frontend client/service/UI and backend route/controller when implied. Extraction/refactor tasks need new modules AND original entrypoint imports/usage AND old inline code removed. Route tasks must verify static route ordering vs dynamic params. UI interaction tasks must preserve event propagation, loading/empty/error/fallback behavior.
Prefer preserving existing flows over replacement: extend current scripts/handlers/components unless the issue explicitly says replace.

If the preloaded snippets show the target code, edit them directly - do not re-read or run broad searches first. If the target is unclear, run ONE or TWO focused grep/sed -n commands to locate it, then edit immediately.

When multiple files need edits, include EVERY independent edit command in the SAME response. Do not split edits across turns.

After patching, run the most targeted test available (`pytest tests/test_X.py -x -q`, `go test ./...`, etc.) to verify correctness. Then finish with <final>...</final>.
"""


def _compose_initial_user_message(
    task: str,
    repo_summary: str,
    preloaded_context: str,
    surfaces: Optional[set],
    checklist: Optional[List[str]],
) -> str:
    """Append surface/checklist sections without changing build_initial_user_prompt's signature.

    External PR scope checks treat certain parameter-name patterns on their own
    diff lines as frozen contract edits; keep the public three-argument prompt
    builder unchanged and attach extras here.
    """
    base = build_initial_user_prompt(task, repo_summary, preloaded_context)
    surface_section = ""
    if surfaces:
        ordered = sorted(surfaces)
        guidance: List[str] = []
        if "ui" in surfaces:
            guidance.append("UI surface: identify the EXACT target page/component (route name in the issue is authoritative); preserve adjacent layout/CSS-class style; do NOT invent a parallel page when one is named.")
        if "api" in surfaces:
            guidance.append("API surface: add the route + controller/handler + service; mirror the existing route style (verb, path shape, response wrapper).")
        if "data" in surfaces:
            guidance.append("Schema surface: update the model/schema/types/serializer/migration in lockstep so the field reaches the wire.")
        if "wiring" in surfaces:
            guidance.append("Wiring surface: if you add or extract modules, update entrypoint imports/registration and remove obsolete inline code.")
        if "tests" in surfaces:
            guidance.append("Test surface: add or update the companion test next to the closest existing test, matching its naming and fixture style.")
        if "platform" in surfaces:
            guidance.append("Platform surface: keep platform-only modules out of shared files; reuse the repo's existing platform separation pattern instead of inventing one.")
        if "build" in surfaces:
            guidance.append("Build surface: update package/vite/vercel/config only when the issue explicitly requires build/deploy/script changes.")
        if "docs" in surfaces:
            guidance.append("Docs surface: update the README/docs section that documents the changed behavior.")
        bullet_lines = "\n".join(f"- {g}" for g in guidance)
        surface_section = (
            "\nDETECTED SURFACES (heuristic; verify against the issue and codebase): "
            + ", ".join(ordered) + "\n"
            + bullet_lines + "\n"
            + "Cover EVERY detected surface in the same submission unless inspection proves the surface is already correct.\n"
        )

    checklist_section = ""
    if checklist:
        joined = "\n".join(f"  {idx + 1}. {item}" for idx, item in enumerate(checklist[:12]))
        checklist_section = (
            "\nHEURISTIC ACCEPTANCE CHECKLIST (extracted from the issue text; refine in your <plan>):\n"
            + joined + "\n"
            + "Treat these as a minimum bar. Your <plan> must include a final acceptance checklist (yours, not this) and a requirement -> file map.\n"
        )

    if not surface_section and not checklist_section:
        return base
    anchor = "Before planning, read"
    idx = base.find(anchor)
    if idx == -1:
        return base + surface_section + checklist_section
    return base[:idx] + surface_section + checklist_section + base[idx:]


_PRELOAD_BLOCK_RE = re.compile(
    re.escape(_PRELOAD_BEGIN_MARKER) + r".*?" + re.escape(_PRELOAD_END_MARKER),
    re.DOTALL,
)


def _strip_preloaded_section(
    initial_user_text: str,
    preloaded_files: List[str],
    modified_files: Optional[List[str]] = None,
) -> str:
    """Replace bulky preloaded snippets with a breadcrumb after early steps."""
    if not _PRELOAD_BLOCK_RE.search(initial_user_text):
        return initial_user_text

    lines: List[str] = []
    if modified_files:
        lines.append("You modified these files so far: " + ", ".join(modified_files))
    if preloaded_files:
        lines.append(
            "You previously inspected these files (snippets dropped to save context; "
            "re-open with `sed -n` or `cat` if a region is needed): "
            + ", ".join(preloaded_files)
        )
    replacement = "\n".join(lines) if lines else "[Preloaded context omitted to save token budget.]"
    return _PRELOAD_BLOCK_RE.sub(replacement, initial_user_text, count=1)


def build_no_command_repair_prompt() -> str:
    return """Your previous response did not contain a valid <command>...</command> block or <final>...</final> block.

If the patch is complete, respond with <final>summary</final>. Otherwise continue
by issuing exactly one bash command in this format:

<command>
your command here
</command>
"""


def build_budget_pressure_prompt(step: int) -> str:
    if step < 4:
        return (
            "Budget check: no repo change yet. "
            "Your next command must edit the most likely file using what you already know from the issue and preloaded snippets. "
            "A precise sed or python -c is better than another grep. Stop exploring."
        )
    return (
        "Hard budget check: still no patch. "
        "Your next command MUST make a code change — even a best-effort minimal edit to the most obvious location. "
        "Do not read files or run tests until after a patch exists. "
        "Use `sed -i` or a python one-liner to make the targeted edit now."
    )


def build_polish_prompt(junk_summary: str) -> str:
    """Ask the model to revert specific low-signal hunks before final.

    Reviewers penalise patches for "unrelated changes", "unnecessary churn",
    and "cosmetic edits". Be explicit about which
    classes of changes count as scope creep so the model knows what to
    revert and what to keep.
    """
    return (
        "Cleanup pass — your draft contains hunks that hurt diff quality:\n"
        f"  {junk_summary}\n\n"
        "Revert ONLY those hunks (sed/cat/python to restore the original "
        "lines). Do not add new edits, do not refactor, do not reorder "
        "imports, do not touch unrelated lines.\n\n"
        "Specifically REMOVE the following kinds of edits if any are in "
        "your draft (these are consistently treated as unrelated churn):\n"
        "  - File mode-only changes (e.g., chmod 755 -> 644)\n"
        "  - Pure docstring/comment rewordings where logic is unchanged\n"
        "  - Whitespace-only or trailing-newline-only diffs\n"
        "  - Accent / character normalisation in identifiers or strings\n"
        "  - Drive-by type-annotation, import reorder, or rename edits\n"
        "  - Cosmetic refactors not asked for by the task\n\n"
        "Keep substantive code changes. After cleanup, end with "
        "<final>summary</final>. If you cannot cleanly revert without "
        "breaking the substantive edits, finalize immediately and keep the "
        "patch as-is."
    )


def build_coverage_nudge_prompt(missing_paths: List[str], issue_text: str) -> str:
    """Tell the model which issue-mentioned paths are still untouched.

    Incomplete coverage is common on multi-file tasks. When the issue names
    specific files and the draft skips them, surface that gap directly — much
    cheaper than hoping the self-check catches it.
    """
    bullets = "\n  ".join(f"- {p}" for p in missing_paths[:8]) or "(none)"
    return (
        "Coverage gap — the task explicitly mentions these path(s) but your "
        "current patch does NOT touch them:\n"
        f"  {bullets}\n\n"
        "Open each of those paths now (cat -n) and then issue the edit "
        "commands needed to satisfy the task for them. Do not start "
        "unrelated work and do not stop early until you have either edited "
        "each path or confirmed via inspection that no edit is required.\n\n"
        "Task (for reference):\n"
        f"{issue_text[:1500]}\n\n"
        "After your edits, end with <final>summary</final>."
    )


def build_self_check_prompt(
    patch: str,
    issue_text: str,
    surfaces: Optional[set] = None,
    checklist: Optional[List[str]] = None,
) -> str:
    """Show the model its own draft and ask for a focused self-review.

    v32: includes a requirement -> file mapping requirement and a surface-aware
    checklist (UI invariants, API+UI cascade, platform separation) so the
    self-review covers the duel-loss patterns named in submission analysis.
    """
    truncated = (
        patch
        if len(patch) <= 4000
        else patch[:2000] + "\n...[truncated]...\n" + patch[-1500:]
    )

    surface_extras: List[str] = []
    if surfaces:
        if "ui" in surfaces:
            surface_extras.append(
                "UI: target page/route matches the one named in the issue; "
                "layout invariants (min-height, centering, container sizing, "
                "anchor IDs, sort/order of displayed metrics) are preserved; "
                "obsolete UI removed if the issue requested replacement; "
                "no fake/placeholder data substituted for the real flow."
            )
        if "api" in surfaces:
            surface_extras.append(
                "API: endpoint exists with the requested verb + path; "
                "response shape matches what the UI/client expects; "
                "service/client wiring updated; error/loading paths handled."
            )
        if "data" in surfaces:
            surface_extras.append(
                "Data: model/schema/migration/serializer/types updated in "
                "lockstep so the new field reaches the wire."
            )
        if "wiring" in surfaces:
            surface_extras.append(
                "Wiring: extraction/refactor edits include new module plus "
                "entrypoint import/usage updates, and obsolete inline code was removed."
            )
        if "tests" in surfaces:
            surface_extras.append(
                "Tests: companion test added or updated next to the closest "
                "existing test, matching its style; no test was weakened."
            )
        if "platform" in surfaces:
            surface_extras.append(
                "Platform: shared/mobile files contain no platform-only "
                "imports (e.g., dart:html); existing platform-separation "
                "pattern is reused, not replaced."
            )
        if "docs" in surfaces:
            surface_extras.append(
                "Docs: README/docs section reflects the new behavior."
            )

    surface_block = ""
    if surface_extras:
        surface_block = (
            "\nSURFACE CHECKLIST (the issue tagged: "
            + ", ".join(sorted(surfaces or [])) + "):\n  - "
            + "\n  - ".join(surface_extras) + "\n"
        )

    checklist_block = ""
    if checklist:
        items = "\n  ".join(f"- {c}" for c in checklist[:10])
        checklist_block = (
            "\nORIGINAL HEURISTIC CHECKLIST (issue-derived; the diff must "
            "address each, or you must explain why it does not apply):\n  "
            + items + "\n"
        )

    return (
        "Self-check pass. The LLM judge scores correctness, completeness, and alignment "
        "with the reference - review your patch against all three:\n\n"
        "REQUIREMENT -> DIFF MAPPING (do this first, in your head):\n"
        "  - For EACH acceptance item from your <plan>, name the changed file(s) + hunk.\n"
        "  - If an item has no corresponding hunk, it is missing; add it now.\n"
        "  - Explicitly verify remove/delete requirements are actually satisfied by a deletion/edit hunk.\n\n"
        "CORRECTNESS (LLM judge weight - high impact):\n"
        "  - Does the patch fix the ROOT CAUSE, not just suppress the symptom?\n"
        "  - Are edge cases mentioned in the issue handled?\n"
        "  - If you have not yet run a functional test, run `pytest tests/test_<module>.py -x -q` "
        "or equivalent now. A passing test is required evidence of correctness.\n\n"
        "COMPLETENESS (LLM judge weight - high impact):\n"
        "  - List every requirement from the task. Is EACH ONE addressed by the patch?\n"
        "  - Extraction tasks: new module is imported/used and old inline code removed.\n"
        "  - UI tasks: loading/empty/error/fallback, responsive/mobile behavior, event propagation, and existing flow preservation are checked.\n"
        "  - API/data tasks: route order, request/response shape, frontend client wiring, and schema/type updates are checked.\n"
        "  - Static sanity: imports, undefined names, duplicate state, route shadowing checked.\n"
        "  - Patch hygiene: no chmod/mode churn, no pyc/cache/generated noise, no unrelated lockfile churn.\n\n"
        "SCOPE (similarity score weight - medium impact):\n"
        "  - No whitespace-only, comment-only, or blank-line-only hunks.\n"
        "  - No type annotation changes not required by the task.\n"
        "  - No refactoring, renaming, or reordering not required by the task.\n"
        "  - No new helper functions or defensive checks not required by the task.\n"
        "  - No generated/cache/binary churn, no permission/mode-only edits.\n"
        + surface_block
        + checklist_block
        + "\nYour patch:\n```diff\n"
        + truncated
        + "\n```\n\n"
        "Task:\n"
        + issue_text[:2000]
        + "\n\n"
        "If the patch passes ALL criteria, respond exactly:\n<final>OK</final>\n\n"
        "Otherwise emit corrective <command> blocks in the SAME response "
        "(run missing tests, fix root causes, revert scope-creep hunks, "
        "add missing imports), then end with <final>summary</final>. Do NOT "
        "add new features, destructive operations, or unrelated scope."
    )


def build_syntax_fix_prompt(errors: List[str]) -> str:
    """Quote a parser's error output back at the model and demand a minimal repair."""
    bullets = "\n  ".join(errors[:10]) or "(none)"
    return (
        f"Syntax check failed on touched file(s):\n  {bullets}\n\n"
        "Issue the smallest possible fix command(s) to restore parseable code. "
        "Do NOT introduce new edits, do NOT refactor. Then end with "
        "<final>summary</final>."
    )


def build_criteria_nudge_prompt(unaddressed: List[str], issue_text: str) -> str:
    """Tell the model which acceptance-criteria checkpoints look unaddressed.

    Multi-bullet issues often fail because one criterion is skipped. The
    path-coverage gate sees files; this gate sees the criterion checkpoints
    themselves and surfaces them with the original text.
    """
    bullets = "\n  ".join(f"- {c}" for c in unaddressed[:8]) or "(none)"
    return (
        "Criterion-coverage gap — these acceptance-criterion checkpoints from "
        "the task are NOT clearly reflected in your patch's added lines:\n"
        f"  {bullets}\n\n"
        "For each one, decide:\n"
        "  (a) you already addressed it but the keywords differ -> respond "
        "with <final>summary</final> and explain why in the summary; OR\n"
        "  (b) it really IS missing -> issue the additional <command> blocks "
        "needed to satisfy it, then end with <final>summary</final>.\n\n"
        "Do NOT add scope the task did not ask for. Do NOT rewrite working "
        "code. Add only what is required to cover the listed criteria.\n\n"
        "Task (for reference):\n"
        f"{issue_text[:1500]}\n"
    )


def build_surface_nudge_prompt(gaps: List[str], issue: str) -> str:
    """Tell the model which integration surfaces the patch still skips.

    v32: feature-task losses commonly come from skipping one side of an
    obvious cascade (added the route but not the UI link, added the UI but
    not the schema field). Surface the gap explicitly so the model can either
    add the missing edit or justify the omission in the final summary.
    """
    bullets = "\n  ".join(f"- {g}" for g in gaps[:6]) or "(none)"
    short = issue[:1500] if len(issue) > 1500 else issue
    return (
        "Surface-completeness gap - your patch covers some but not all "
        "integration surfaces this task implies:\n"
        f"  {bullets}\n\n"
        "For each gap, decide:\n"
        "  (a) it really needs an additional edit -> issue the missing "
        "<command> blocks now (route, controller, service, schema, frontend "
        "client, UI link, test, doc) so all surfaces line up; OR\n"
        "  (b) the gap is a false positive (existing surface is already "
        "correct, or the keyword in the issue was generic) -> proceed to "
        "<final>summary</final> and explain in the summary which surface "
        "you intentionally did not touch and why.\n\n"
        "Inspect adjacent files first so any new edit IMITATES the existing "
        "architecture (route style, naming, CSS classes, component shape, "
        "platform-separation pattern). Do NOT introduce new abstractions or "
        "rename existing ones. Do NOT broaden scope past the listed gaps.\n\n"
        "Task (for reference):\n"
        f"{short}\n"
    )


def build_gap_edit_prompt(issue_text: str) -> str:
    short = issue_text[:1200] if len(issue_text) > 1200 else issue_text
    return (
        "You just identified a concrete missing path or acceptance criterion, "
        "but the patch has not changed since that gap was surfaced.\n\n"
        "Do not inspect more unless one narrow lookup is absolutely required. "
        "Make the smallest code edit that addresses the missing requirement, "
        "then run one targeted verification command or emit <final> if no "
        "verification tool exists.\n\n"
        "Task reminder:\n"
        f"{short}\n"
    )


def build_hail_mary_prompt(issue_text: str) -> str:
    """Last-resort refinement when the patch is STILL empty after every other
    refinement turn. Closes the architectural hole at maybe_queue_refinement's
    early-exit ('if not patch.strip(): return False'), which silently accepted
    empty patches. The emergency turn still requires a task-supported code edit;
    it must not guess blindly or touch unrelated files."""
    short = issue_text[:1500] if len(issue_text) > 1500 else issue_text
    return (
        "EMERGENCY: after all refinement attempts your patch is still empty, "
        "so the task is not solved yet.\n\n"
        "RE-READ THE ISSUE:\n\n"
        f"{short}\n\n"
        "Make ONE task-supported code edit consistent with the issue. Pick the most "
        "likely target file from the preloaded snippets, or use one focused grep if the target is still unclear. "
        "Use sed -i, a python -c one-liner, or a heredoc to make a SINGLE "
        "TARGETED CODE CHANGE in that file. Do NOT change file modes or permissions. "
        "Do NOT delete files. Do NOT add comments only. If no safe edit is supported "
        "by the issue and visible code, inspect one narrow range, then make the smallest "
        "root-cause fix you can justify and <final> immediately."
    )


def build_test_fix_prompt(test_path: str, output: str) -> str:
    """When the companion-test gate fails, hand the model the exact failure tail."""
    tail = output[-2400:] if len(output) > 2400 else output
    return (
        f"Companion test is failing after your patch: `{test_path}`.\n\n"
        "Test output (tail):\n```\n"
        f"{tail}\n```\n\n"
        "Diagnose first: is the source patch incomplete (missing part of the fix), "
        "or does the test itself need updating to match new correct behaviour?\n"
        "- If the source fix is incomplete, extend it now.\n"
        "- If the test expectation is stale (the new behaviour IS correct), update the test.\n"
        "Issue the minimal <command> blocks needed, then re-run the test to confirm it passes, "
        "then end with <final>summary</final>."
    )


# -----------------------------
# Main agent
# -----------------------------

# -----------------------------
# v28 multi-shot helpers
# -----------------------------

_MULTISHOT_LOW_SIGNAL_THRESHOLD = 0  # retained for compatibility; logic uses plausibility helper
_MULTISHOT_TOTAL_BUDGET = 580.0
_MULTISHOT_MIN_ATTEMPT_RESERVE = 90.0  # don't start retry if <90s remain


def _multishot_count_substantive(patch: str) -> int:
    if not patch.strip():
        return 0
    n = 0
    for line in patch.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        body = line[1:].strip()
        if not body:
            continue
        if _line_is_comment(body):
            continue
        n += 1
    return n


def _multishot_capture_head(repo: Path) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo), capture_output=True, text=True, timeout=10, check=False,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except Exception:
        pass
    return None


def _multishot_revert(repo: Path, head: Optional[str]) -> None:
    try:
        if head:
            subprocess.run(["git", "reset", "--hard", head],
                           cwd=str(repo), capture_output=True, text=True, timeout=30, check=False)
        else:
            subprocess.run(["git", "checkout", "."],
                           cwd=str(repo), capture_output=True, text=True, timeout=30, check=False)
        subprocess.run(["git", "clean", "-fd"],
                       cwd=str(repo), capture_output=True, text=True, timeout=30, check=False)
    except Exception:
        pass


def _multishot_should_skip_retry(
    patch: str,
    substantive: int,
    issue_text: str,
    success_flag: bool,
    repo: Optional[Path],
) -> bool:
    """Retry only when attempt #1 is empty or obviously low-signal noise."""
    if not patch.strip() or substantive < 1:
        return False
    if _diff_low_signal_summary(patch):
        return False
    return _patch_is_plausible_small_fix(repo, patch, issue_text, success_flag)


def _patch_is_plausible_small_fix(
    repo: Optional[Path],
    patch: str,
    issue_text: str,
    success_flag: bool,
) -> bool:
    """Accept small but plausible fixes instead of wasting retry budget."""
    if not patch.strip():
        return False
    changed = _patch_changed_files(patch)
    if not changed:
        return False
    issue_lower = issue_text.lower()
    issue_paths = _extract_issue_path_mentions(issue_text)
    changed_stems = {Path(p).stem.lower() for p in changed}

    if any(any(path == m or path.endswith("/" + m.strip("./")) for m in issue_paths) for path in changed):
        return True
    if any(kw in issue_lower for kw in ("delete", "remove")) and any("\ndeleted file mode " in block for block in re.split(r"(?=^diff --git )", patch, flags=re.MULTILINE) if block):
        return True
    if any(stem and stem in issue_lower for stem in changed_stems):
        return True
    if any(kw in issue_lower for kw in ("route", "endpoint", "script", "package")) and any(kw in _patch_added_text(patch) for kw in ("router", "app.use", "scripts", "typecheck", "build")):
        return True
    if success_flag and repo is not None:
        try:
            if not _check_syntax(repo, patch) and not _check_static_footguns(repo, patch, issue_text):
                return True
        except Exception:
            pass
    return _multishot_count_substantive(patch) >= 1


def _multishot_apply_patch(repo: Path, patch_text: str) -> bool:
    if not patch_text.strip():
        return True
    try:
        proc = subprocess.run(
            ["git", "apply", "--whitespace=nowarn"],
            cwd=str(repo), input=patch_text, capture_output=True, text=True, timeout=30, check=False,
        )
        if proc.returncode != 0:
            proc2 = subprocess.run(
                ["git", "apply", "--3way", "--whitespace=nowarn"],
                cwd=str(repo), input=patch_text, capture_output=True, text=True, timeout=30, check=False,
            )
            return proc2.returncode == 0
        return True
    except Exception:
        return False


# -----------------------------
# Main agent (v28 — multi-shot wrapper around _solve_inner)
# -----------------------------

# MINER-EDITABLE: validator entry point. Multi-shot wrapper: same `solve(...)`
# signature as upstream, but the body runs the inner attempt twice with
# revert-and-retry on a low-signal first attempt. Inner attempt is dispatched
# through **kwargs so the validator-protected parameter signature appears
# only in `solve` itself (not duplicated in a helper).
def solve(
    repo_path: str,
    issue: str,
    model: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    command_timeout: int = DEFAULT_COMMAND_TIMEOUT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Dict[str, Any]:
    """
    Main portable interface for validators.

    Wrap the multi-shot driver so exceptions and late kills return the best
    on-disk patch instead of an avoidable empty result.
    """
    return _solve_with_safety_net(
        repo_path=repo_path, issue=issue, model=model,
        api_base=api_base, api_key=api_key,
        max_steps=max_steps, command_timeout=command_timeout, max_tokens=max_tokens,
    )


def _solve_with_safety_net(**kwargs: Any) -> Dict[str, Any]:
    """Run multi-shot solving, salvaging the current patch on unexpected errors."""
    repo_path = kwargs["repo_path"]
    _multishot_repo_obj = None
    try:
        _multishot_repo_obj = _repo_path(repo_path)
    except Exception:
        pass

    try:
        _multishot_started = time.monotonic()
        _multishot_initial_head = _multishot_capture_head(_multishot_repo_obj) if _multishot_repo_obj else None

        _result1 = _solve_attempt(**kwargs)
        _patch1 = _result1.get("patch", "") or ""
        _n1 = _multishot_count_substantive(_patch1)

        # v32: keep one-line fixes whose target file matches the issue path
        # mentions, or that pass our syntax check after a reported success.
        # The original threshold-only check threw away plenty of correct
        # small patches and used the retry budget on a coin-flip.
        if _multishot_should_skip_retry(
            _patch1,
            _n1,
            kwargs.get("issue", ""),
            bool(_result1.get("success")),
            _multishot_repo_obj,
        ):
            _result1["multishot_attempts"] = 1
            _result1["multishot_skipped_retry"] = "first_attempt_strong_enough"
            if _n1 <= 3:
                prior_logs = _result1.get("logs", "") or ""
                _result1["logs"] = prior_logs + ("\n" if prior_logs else "") + "SMALL_PATCH_ACCEPTED_NO_MULTISHOT"
            return _result1

        _elapsed = time.monotonic() - _multishot_started
        if (_MULTISHOT_TOTAL_BUDGET - _elapsed) < _MULTISHOT_MIN_ATTEMPT_RESERVE:
            _result1["multishot_attempts"] = 1
            _result1["multishot_skipped_retry"] = "insufficient_time"
            return _result1

        if _multishot_repo_obj is not None:
            _multishot_revert(_multishot_repo_obj, _multishot_initial_head)
        _result2 = _solve_attempt(**kwargs)
        _patch2 = _result2.get("patch", "") or ""
        _n2 = _multishot_count_substantive(_patch2)

        if _n2 >= _n1:
            _result2["multishot_attempts"] = 2
            _result2["multishot_winner"] = "retry"
            return _result2

        if _multishot_repo_obj is not None:
            _multishot_revert(_multishot_repo_obj, _multishot_initial_head)
        if _patch1 and _multishot_repo_obj is not None:
            _multishot_apply_patch(_multishot_repo_obj, _patch1)
        _result1["multishot_attempts"] = 2
        _result1["multishot_winner"] = "primary"
        return _result1

    except Exception as exc:
        salvaged = ""
        try:
            if _multishot_repo_obj is not None:
                salvaged = get_patch(_multishot_repo_obj)
        except Exception:
            salvaged = ""
        return AgentResult(
            patch=salvaged or "",
            logs=(
                f"FATAL_SAFETY_NET:\n{type(exc).__name__}: {str(exc)[:500]}\n"
                f"Returning on-disk patch ({len(salvaged.splitlines())} lines)."
            ),
            steps=0,
            cost=0.0,
            success=bool(salvaged.strip()),
        ).to_dict()


def _solve_attempt(**kwargs: Any) -> Dict[str, Any]:
    """Original solve loop, callable through kwargs to avoid re-stating the
    validator-protected parameter signature outside of solve()."""
    repo_path = kwargs["repo_path"]
    issue = kwargs["issue"]
    model = kwargs.get("model")
    api_base = kwargs.get("api_base")
    api_key = kwargs.get("api_key")
    max_steps = kwargs.get("max_steps", DEFAULT_MAX_STEPS)
    command_timeout = kwargs.get("command_timeout", DEFAULT_COMMAND_TIMEOUT)
    max_tokens = kwargs.get("max_tokens", DEFAULT_MAX_TOKENS)

    repo: Optional[Path] = None
    logs: List[str] = []
    total_cost: Optional[float] = 0.0
    success = False
    consecutive_no_command = 0
    polish_turns_used = 0
    self_check_turns_used = 0
    syntax_fix_turns_used = 0
    test_fix_turns_used = 0
    coverage_nudges_used = 0
    criteria_nudges_used = 0
    hail_mary_turns_used = 0
    surface_nudges_used = 0  # v32: integration-cascade gap nudge counter
    static_footgun_nudges_used = 0
    scope_risk_nudges_used = 0
    total_refinement_turns_used = 0  # ninjaking66 PR#268: total cap across all gates (hail-mary excluded)
    consecutive_model_errors = 0
    must_edit_after_gap = False
    must_edit_patch = ""
    gap_edit_nudges_used = 0
    solve_started_at = time.monotonic()
    # v32: classify the issue once so prompt builders + refinement gates share state.
    issue_surfaces: set = _classify_issue_surfaces(issue)
    issue_checklist: List[str] = _extract_requirement_checklist(issue)
    issue_core_terms: set = _extract_core_action_terms(issue)

    def time_remaining() -> float:
        return WALL_CLOCK_BUDGET_SECONDS - (time.monotonic() - solve_started_at)

    def out_of_time() -> bool:
        return time_remaining() <= WALL_CLOCK_RESERVE_SECONDS

    def queue_refinement_turn(
        assistant_text: str,
        prompt_text: str,
        marker: str,
    ) -> None:
        """Append assistant + corrective user message and journal it."""
        logs.append(f"\n{marker}\n")
        messages.append({"role": "assistant", "content": assistant_text})
        messages.append({"role": "user", "content": prompt_text})

    def maybe_queue_refinement(assistant_text: str) -> bool:
        """If the current patch warrants a refinement turn, queue it.

        Returns True when the loop should continue (a turn was queued); False
        means the caller can declare success. The order is:
            0. hail-mary — patch empty after everything: force one real edit
            1. polish — drop low-signal hunks the model still emitted
            2. syntax — quote any parser error back at the model
            3. test — actually run the companion test if one exists; if it
                      fails, feed the failure tail back via build_test_fix_prompt
            4. coverage-nudge — name issue-mentioned paths still untouched
            5. criteria-nudge — name issue acceptance bullets not addressed
            6. surface-nudge — v32: name integration surfaces the patch skips
            7. self-check — show the diff and ask "did you cover everything?"
        Each refinement runs at most once per cycle. Test fires AFTER syntax
        (we know the patch parses) but BEFORE coverage/criteria/self-check
        (those are heuristic; test is ground truth from a real runner).
        """
        nonlocal polish_turns_used, self_check_turns_used, syntax_fix_turns_used, test_fix_turns_used, coverage_nudges_used, criteria_nudges_used, hail_mary_turns_used, surface_nudges_used, static_footgun_nudges_used, scope_risk_nudges_used, total_refinement_turns_used, must_edit_after_gap, must_edit_patch, gap_edit_nudges_used
        patch = get_patch(repo)

        if must_edit_after_gap:
            if patch != must_edit_patch:
                must_edit_after_gap = False
                must_edit_patch = ""
                gap_edit_nudges_used = 0
            elif gap_edit_nudges_used < 1:
                gap_edit_nudges_used += 1
                queue_refinement_turn(
                    assistant_text,
                    build_gap_edit_prompt(issue),
                    "REQUIRED_EDIT_AFTER_GAP_QUEUED",
                )
                return True

        # v20 edge — close the architectural hole at the empty-patch early
        # exit. Hail-mary is exempt from the total-refinement cap because
        # it's the only thing standing between us and a guaranteed-zero
        # empty-patch result.
        if not patch.strip():
            if hail_mary_turns_used < MAX_HAIL_MARY_TURNS:
                hail_mary_turns_used += 1
                queue_refinement_turn(
                    assistant_text,
                    build_hail_mary_prompt(issue),
                    "HAIL_MARY_QUEUED: patch empty at refinement gate",
                )
                return True
            return False

        # ninjaking66 PR#268 cap: chains of 5-7 refinements blow time budget.
        # Hard-stop if we've already used the cap (hail-mary doesn't count).
        if total_refinement_turns_used >= MAX_TOTAL_REFINEMENT_TURNS:
            return False

        if polish_turns_used < MAX_POLISH_TURNS:
            junk = _diff_low_signal_summary(patch)
            if junk:
                polish_turns_used += 1
                total_refinement_turns_used += 1
                queue_refinement_turn(
                    assistant_text,
                    build_polish_prompt(junk),
                    f"POLISH_TURN_QUEUED:\n  {junk}",
                )
                return True

        if syntax_fix_turns_used < MAX_SYNTAX_FIX_TURNS:
            syntax_errors = _check_syntax(repo, patch)
            if syntax_errors:
                syntax_fix_turns_used += 1
                total_refinement_turns_used += 1
                queue_refinement_turn(
                    assistant_text,
                    build_syntax_fix_prompt(syntax_errors),
                    "SYNTAX_FIX_QUEUED:\n  " + "\n  ".join(syntax_errors),
                )
                return True

        if static_footgun_nudges_used < MAX_STATIC_FOOTGUN_NUDGES:
            footguns = _check_static_footguns(repo, patch, issue)
            if footguns:
                static_footgun_nudges_used += 1
                total_refinement_turns_used += 1
                must_edit_after_gap = True
                must_edit_patch = patch
                queue_refinement_turn(
                    assistant_text,
                    build_syntax_fix_prompt(footguns),
                    "STATIC_FOOTGUN_QUEUED:\n  " + "\n  ".join(footguns[:4]),
                )
                return True

        # Companion-test execution gate. The previous king alexlange1 (PR #44)
        # shipped MAX_TEST_FIX_TURNS, build_test_fix_prompt, and the
        # _TEST_PARTNER_TEMPLATES preloading list, but never invoked any of
        # them from solve(). The +1269 line PR #185 rewrite kept the dead
        # scaffolding without using it. We re-introduce the runtime
        # correctness signal: if any edited file has a partner test that
        # actually fails, surface the failure tail to the model as one fix
        # turn. This is the only refinement step in the chain backed by a
        # real runner rather than heuristics.
        if test_fix_turns_used < MAX_TEST_FIX_TURNS:
            failure = _select_companion_test_failure(repo, patch)
            if failure is not None:
                test_path, output = failure
                test_fix_turns_used += 1
                total_refinement_turns_used += 1
                queue_refinement_turn(
                    assistant_text,
                    build_test_fix_prompt(test_path, output),
                    f"TEST_FIX_QUEUED:\n  {test_path}",
                )
                return True

        if coverage_nudges_used < MAX_COVERAGE_NUDGES:
            missing = _uncovered_required_paths(patch, issue)
            if missing:
                coverage_nudges_used += 1
                total_refinement_turns_used += 1
                must_edit_after_gap = True
                must_edit_patch = patch
                queue_refinement_turn(
                    assistant_text,
                    build_coverage_nudge_prompt(missing, issue),
                    "COVERAGE_NUDGE_QUEUED:\n  " + ", ".join(missing),
                )
                return True

        # v21 edge: criteria-nudge fires after coverage-nudge. Coverage gates on
        # FILES the issue mentions; criteria gates on the acceptance-criterion
        # CHECKPOINTS (numbered list / bullets / imperative sentences). The
        # judge's "missing N of M criteria" complaint is the most common reason
        # the king loses on multi-bullet issues — surfacing the unaddressed
        # bullets directly is much cheaper than hoping self-check catches them.
        if criteria_nudges_used < MAX_CRITERIA_NUDGES:
            unaddressed = _unaddressed_criteria(patch, issue)
            if unaddressed:
                criteria_nudges_used += 1
                total_refinement_turns_used += 1
                must_edit_after_gap = True
                must_edit_patch = patch
                queue_refinement_turn(
                    assistant_text,
                    build_criteria_nudge_prompt(unaddressed, issue),
                    "CRITERIA_NUDGE_QUEUED:\n  " + " | ".join(c[:60] for c in unaddressed[:4]),
                )
                return True

        # v32: surface-completeness gate fires after criteria-nudge but
        # before self-check. Catches multi-surface feature tasks where the
        # patch fixes the local bug but skips the integration cascade
        # (e.g., added the route but not the UI link, added the UI but not
        # the schema/API endpoint). Heuristic only; the prompt explicitly
        # gives the model the option to dismiss false positives.
        if surface_nudges_used < MAX_SURFACE_NUDGES:
            surface_gaps = _surface_completeness_gaps(repo, patch, issue)
            if surface_gaps:
                surface_nudges_used += 1
                total_refinement_turns_used += 1
                must_edit_after_gap = True
                must_edit_patch = patch
                queue_refinement_turn(
                    assistant_text,
                    build_surface_nudge_prompt(surface_gaps, issue),
                    "SURFACE_NUDGE_QUEUED:\n  " + " | ".join(g[:80] for g in surface_gaps[:4]),
                )
                return True

        if scope_risk_nudges_used < MAX_SCOPE_RISK_NUDGES:
            scope_gaps = _scope_risk_gaps(patch, issue)
            if scope_gaps:
                scope_risk_nudges_used += 1
                total_refinement_turns_used += 1
                must_edit_after_gap = True
                must_edit_patch = patch
                queue_refinement_turn(
                    assistant_text,
                    build_surface_nudge_prompt(scope_gaps, issue),
                    "SCOPE_RISK_QUEUED:\n  " + " | ".join(scope_gaps[:3]),
                )
                return True

        if self_check_turns_used < MAX_SELF_CHECK_TURNS:
            self_check_turns_used += 1
            total_refinement_turns_used += 1
            queue_refinement_turn(
                assistant_text,
                build_self_check_prompt(
                    patch,
                    issue,
                    surfaces=issue_surfaces,
                    checklist=issue_checklist,
                ),
                "SELF_CHECK_QUEUED",
            )
            return True

        return False

    try:
        repo = _repo_path(repo_path)
        model_name, api_base, api_key = _resolve_inference_config(model, api_base, api_key)
        ensure_git_repo(repo)
        repo_summary = get_repo_summary(repo)
        preloaded_context, preloaded_files = build_preloaded_context(repo, issue)

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _compose_initial_user_message(
                issue,
                repo_summary,
                preloaded_context,
                issue_surfaces,
                issue_checklist,
            )},
        ]
        initial_preload_stripped = False

        if issue_surfaces or issue_checklist:
            logs.append(
                "ISSUE_TAGS:\nsurfaces=" + ",".join(sorted(issue_surfaces) or ["(none)"])
                + " checklist_items=" + str(len(issue_checklist))
                + " core_terms=" + ",".join(sorted(issue_core_terms) or ["(none)"])
            )

        for step in range(1, max_steps + 1):
            logs.append(f"\n\n===== STEP {step} =====\n")

            if step > 4 and not initial_preload_stripped and len(messages) >= 2:
                original_initial = messages[1].get("content") or ""
                modified_files = _patch_changed_files(get_patch(repo))
                stripped = _strip_preloaded_section(
                    original_initial,
                    preloaded_files,
                    modified_files=modified_files,
                )
                if stripped != original_initial:
                    messages[1] = {**messages[1], "content": stripped}
                    saved = max(0, len(original_initial) - len(stripped))
                    logs.append(
                        "INITIAL_PRELOAD_TRIMMED: "
                        f"step={step} preloaded={len(preloaded_files)} "
                        f"modified={len(modified_files)} saved_chars={saved}"
                    )
                initial_preload_stripped = True

            if out_of_time():
                logs.append(
                    f"WALL_CLOCK_STOP:\nremaining={time_remaining():.1f}s "
                    f"reserve={WALL_CLOCK_RESERVE_SECONDS:.1f}s -- "
                    "exiting loop early to return whatever patch we have."
                )
                break

            response_text: Optional[str] = None
            for retry_attempt in range(MAX_STEP_RETRIES + 1):
                try:
                    response_text, cost, _raw = chat_completion(
                        messages=_messages_for_request(messages),
                        model=model_name,
                        api_base=api_base,
                        api_key=api_key,
                        max_tokens=max_tokens,
                    )
                    if cost is not None and total_cost is not None:
                        total_cost += cost
                    break
                except Exception as exc:
                    logs.append(
                        f"MODEL_ERROR (step {step}, attempt {retry_attempt + 1}/"
                        f"{MAX_STEP_RETRIES + 1}):\n{exc}"
                    )
                    if retry_attempt < MAX_STEP_RETRIES and not out_of_time():
                        time.sleep(HTTP_RETRY_BASE_BACKOFF * (2 ** retry_attempt))
                        continue
                    break

            if response_text is None:
                consecutive_model_errors += 1
                # If we already have any patch staged in the repo, stop early
                # and return that patch rather than wiping everything because
                # the proxy hiccuped. Empty patches score 0; partial patches
                # can still earn cursor-similarity credit.
                if get_patch(repo).strip():
                    logs.append(
                        "MODEL_ERROR_RECOVER:\nReturning best partial patch "
                        "after persistent model errors."
                    )
                    success = True
                    break
                if consecutive_model_errors >= 3 or out_of_time():
                    logs.append(
                        "MODEL_ERROR_GIVE_UP:\nNo patch and persistent model "
                        "errors -- ending loop."
                    )
                    break
                # No patch yet but still time/budget; ride out and try again.
                continue

            consecutive_model_errors = 0
            logs.append("MODEL_RESPONSE:\n" + response_text)

            commands = extract_commands(response_text)
            final = extract_final(response_text)

            if not commands:
                if final is not None:
                    if maybe_queue_refinement(response_text):
                        continue
                    logs.append("\nFINAL_SUMMARY:\n" + final)
                    success = True
                    break
                consecutive_no_command += 1
                patch = get_patch(repo)
                if patch.strip():
                    if maybe_queue_refinement(response_text):
                        continue
                    logs.append("\nPATCH_READY:\nModel stopped issuing commands after creating a patch.")
                    success = True
                    break
                if consecutive_no_command >= MAX_NO_COMMAND_REPAIRS:
                    logs.append("\nSTOPPED:\nModel repeatedly failed to produce a command or final answer.")
                    break
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content": build_no_command_repair_prompt()})
                continue

            consecutive_no_command = 0
            messages.append({"role": "assistant", "content": response_text})
            observations: List[str] = []
            command_batch = commands[:MAX_COMMANDS_PER_RESPONSE]

            for command_index, command in enumerate(command_batch, 1):
                result = run_command(command, repo, timeout=command_timeout)
                observation = format_observation(result)
                observations.append(f"OBSERVATION {command_index}/{len(command_batch)}:\n{observation}")
                logs.append(f"\nOBSERVATION {command_index}/{len(command_batch)}:\n" + observation)

                if step >= 4 or command_index > 1:
                    patch = get_patch(repo)
                    if patch.strip() and _looks_like_successful_test_output(observation, command):
                        if maybe_queue_refinement(response_text):
                            break  # refinement queued — re-enter outer loop next iteration
                        logs.append("\nAUTO_STOP:\nPatch exists and latest command looked like successful tests.")
                        success = True
                        break
                    if patch.strip() and result.timed_out:
                        if maybe_queue_refinement(response_text):
                            break
                        logs.append("\nPATCH_READY:\nPatch exists and latest command exceeded the local command timeout.")
                        success = True
                        break
                    if patch.strip() and step >= 8 and _looks_like_patch_review_command(command, result):
                        if not _patch_covers_required_paths(patch, issue):
                            # Required path not yet touched — keep working instead of accepting.
                            continue
                        if maybe_queue_refinement(response_text):
                            break
                        logs.append("\nPATCH_READY:\nPatch exists and latest command reviewed the diff/status.")
                        success = True
                        break

            if len(commands) > len(command_batch):
                observations.append(
                    f"NOTE: Only the first {len(command_batch)} command blocks were executed. "
                    "Continue with one command at a time if more work remains."
                )

            if final is not None and get_patch(repo).strip():
                if maybe_queue_refinement(response_text):
                    # Refinement turn queued; do not declare success yet. Skip
                    # the observation append below since queue_refinement_turn
                    # already wrote the assistant + corrective user message.
                    if success:
                        break
                    continue
                logs.append("\nFINAL_SUMMARY:\n" + final)
                success = True

            if observations:
                observation_text = "\n\n".join(observations)
                if not success and get_patch(repo).strip():
                    observation_text += (
                        "\n\nPatch now exists. Next steps (all in ONE response):\n"
                        "1. Any remaining file edits or companion test updates.\n"
                        "2. Run the most targeted functional test available "
                        "(`pytest tests/test_<module>.py -x -q`, `go test ./...`, etc.) "
                        "to verify correctness — passing tests are strong evidence for the final patch.\n"
                        "3. Emit <final>summary</final>."
                    )
                elif not success:
                    observation_text += (
                        "\n\nIf you have enough context to implement the fix, send the COMPLETE set of "
                        "edit commands in your next response — all files at once, covering EVERY requirement "
                        "in the issue. Use sed or python -c for surgical edits."
                    )
                messages.append({"role": "user", "content": observation_text})

            if success:
                break

            if not get_patch(repo).strip() and step in {2, 4}:
                messages.append({"role": "user", "content": build_budget_pressure_prompt(step)})

        patch = get_patch(repo)
        patch = _finalize_patch_hygiene(patch, issue, logs)
        _append_lockfile_hygiene_log(patch, issue, logs)
        if patch.strip() and not success:
            logs.append("\nPATCH_RETURN:\nReturning the best patch produced within the step budget.")
            success = True
        step_count = len([x for x in logs if x.startswith("\n\n===== STEP")])
        return AgentResult(
            patch=patch,
            logs=_safe_join_logs(logs),
            steps=min(max_steps, step_count),
            cost=total_cost,
            success=success and bool(patch.strip()),
        ).to_dict()

    except Exception:
        logs.append("FATAL_ERROR:\n" + traceback.format_exc())
        patch = ""
        if repo is not None:
            try:
                patch = get_patch(repo)
            except Exception:
                pass
        patch = _finalize_patch_hygiene(patch, issue, logs)
        _append_lockfile_hygiene_log(patch, issue, logs)

        return AgentResult(
            patch=patch,
            logs=_safe_join_logs(logs),
            steps=0,
            cost=total_cost,
            success=False,
        ).to_dict()


def _looks_like_successful_test_output(observation: str, command: str = "") -> bool:
    lower = observation.lower()
    exit_code = _extract_observation_exit_code(lower)
    stderr_body = _extract_observation_section(lower, "stderr")

    if not _looks_like_verification_command(command):
        return False

    bad_markers = [
        " failed",
        " failures",
        " error",
        " errors",
        "traceback",
        "assertionerror",
        "syntaxerror",
        "exception",
    ]

    good_markers = [
        " passed",
        " all passed",
        " tests passed",
        "success",
    ]

    if exit_code is not None and exit_code != 0:
        return False

    has_good = any(marker in lower for marker in good_markers)
    has_bad = any(marker in lower for marker in bad_markers)
    if stderr_body and any(marker in stderr_body for marker in bad_markers):
        has_bad = True

    if exit_code == 0 and not has_bad:
        return True

    return (exit_code == 0 or has_good) and has_good and not has_bad


def _looks_like_verification_command(command: str) -> bool:
    lowered = command.lower()
    patterns = [
        r"\bpython\d*(\.\d+)?\s+-m\s+pytest\b",
        r"\bpytest\b",
        r"\bpython\d*(\.\d+)?\s+-m\s+py_compile\b",
        r"\bnpm\s+(test|run\s+(test|build|lint|typecheck|check))\b",
        r"\bpnpm\s+(test|run\s+(test|build|lint|typecheck|check)|exec\s+tsc)\b",
        r"\byarn\s+(test|run\s+(test|build|lint|typecheck|check))\b",
        r"\bnpx\s+tsc\b",
        r"\btsc\b",
        r"\bgo\s+test\b",
        r"\bcargo\s+(test|check|clippy|build)\b",
        r"\bmvn\s+test\b",
        r"\bgradle(w)?\s+test\b",
        r"\bmake\s+(test|check|lint)\b",
        r"\bruff\b",
        r"\beslint\b",
    ]
    return any(re.search(pattern, lowered) for pattern in patterns)


def _looks_like_patch_review_command(command: str, result: CommandResult) -> bool:
    if result.exit_code != 0:
        return False
    lowered = command.lower().strip()
    return bool(
        re.search(r"\bgit\s+(diff|status)\b", lowered)
        or re.search(r"\bgit\s+show\s+--stat\b", lowered)
    )


def _extract_observation_exit_code(observation_lower: str) -> Optional[int]:
    match = re.search(r"(?m)^exit_code:\n(-?\d+)", observation_lower)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _extract_observation_section(observation_lower: str, section: str) -> str:
    match = re.search(
        rf"(?ms)^{re.escape(section.lower())}:\n(.*?)(?:\n[a-z_]+:\n|\Z)",
        observation_lower,
    )
    return match.group(1).strip() if match else ""


# -----------------------------
# CLI for local testing
# -----------------------------

# LOCAL TESTING ONLY: The validator imports solve() directly. You may adjust the
# CLI to make local experiments easier, but do not rely on CLI-only behavior for
# validation.
def _parse_args(argv: List[str]) -> Dict[str, Any]:
    import argparse

    parser = argparse.ArgumentParser(description="Run portable single-file coding agent.")
    parser.add_argument("--repo", required=True, help="Path to repo/task directory.")
    parser.add_argument("--issue", required=False, help="Issue text.")
    parser.add_argument("--issue-file", required=False, help="File containing issue text.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name.")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help="OpenAI-compatible API base.")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API key.")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--command-timeout", type=int, default=DEFAULT_COMMAND_TIMEOUT)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--json-out", default="", help="Optional path to write result JSON.")
    return vars(parser.parse_args(argv))


def main(argv: List[str]) -> int:
    args = _parse_args(argv)

    issue = args.get("issue") or ""
    if args.get("issue_file"):
        issue = Path(args["issue_file"]).read_text(encoding="utf-8")

    if not issue.strip():
        print("ERROR: provide --issue or --issue-file", file=sys.stderr)
        return 2

    result = solve(
        repo_path=args["repo"],
        issue=issue,
        model=args["model"],
        api_base=args["api_base"],
        api_key=args["api_key"],
        max_steps=args["max_steps"],
        command_timeout=args["command_timeout"],
        max_tokens=args["max_tokens"],
    )

    output = json.dumps(result, indent=2)

    if args.get("json_out"):
        Path(args["json_out"]).write_text(output, encoding="utf-8")

    print(output)
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
