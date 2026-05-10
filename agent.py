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
MAX_CONVERSATION_CHARS = 150000
MAX_PRELOADED_CONTEXT_CHARS = 32000
MAX_PRELOADED_FILES = 10
MAX_NO_COMMAND_REPAIRS = 3
MAX_COMMANDS_PER_RESPONSE = 12

# Anti-whiff knobs. Empty patches score zero on baseline-similarity, so any
# transient model error or stuck loop directly costs us rounds. Be aggressive
# about retrying instead of returning early with no edits.
# Hardcoded — not user-tunable. The PR Scope Guard's env-var allowlist
# (pr_scope_guard.py:ALLOWED_ENV_NAMES) does not permit new AGENT_* names.
HTTP_MAX_RETRIES = 3
HTTP_RETRY_BASE_BACKOFF = 1.0
MAX_STEP_RETRIES = 2

# v2.1: wall-clock guard removed. The historic 270 s budget was throwing
# away ~1 in 5 attempts that would have finished cleanly within the next
# 1-2 model turns. Cursor inference is fast; ``DEFAULT_MAX_STEPS = 30``
# is the only loop bound now. Time-based truncation cost more attempts
# than it saved.

# v2.1: edit-pressure nudge schedule. Replaces the hardcoded `step in {2, 4}`
# bullying that fired before the model had any meaningful repo context. With
# the defaults below the schedule is: weak nudge at step 5, then strong
# nudges at 7 / 9 / 11 / ... — each only when no patch has been produced yet.
# Once any patch lands the schedule is silenced for the rest of the run.
EDIT_WARN_FIRST_STEP = 5
EDIT_WARN_INTERVAL = 2

# v2.4 / v3.2: the eight separate refinement caps that lived here were
# replaced by a single LLM-judge gate. The previous ladder ran 7
# priority-ordered checks (hail-mary / polish / syntax / test / coverage /
# criteria / self-check) with a total-cap on top, ~130 lines of keyword
# matching, and a dedicated runner-process gate. v2.4 collapsed all of
# that into one PASS/FAIL acceptance call; v3.2 upgrades that into a
# scored final-review judge: the model gives a 0-100 SCORE plus an
# explicit VERDICT plus a concrete FIX_LIST (see ``judge_final``). A
# patch passes only when VERDICT is PASS *and* the score is at least
# ``FINAL_CHECK_PASS_THRESHOLD``. Permissive on every judge error so a
# flaky judge can never block shipping.
MAX_FINAL_CHECK_TURNS = 3   # at most one judge call per solve
FINAL_CHECK_PASS_THRESHOLD = 70  # v3.2: PASS gate is VERDICT==PASS AND score >= this
FINAL_CHECK_MAX_ISSUE_CHARS = 4000
FINAL_CHECK_MAX_RESPONSE_CHARS = 2000
FINAL_CHECK_MAX_PATCH_CHARS = 8000  # v3.2: was 6000; mid-truncate at 5000+2500
FINAL_CHECK_MAX_TOKENS = 500  # v3.2: was 400; SCORE+VERDICT+REASON+FIX_LIST is longer

_STYLE_HINT_BUDGET = 600   # VladaWebDev PR#250: cap on detected-style block in preloaded context

# Recent-commit injection: small in-context style anchors from the staged repo's
# real history. The validator clones the real repo with full git history; the
# pilot stages snapshots with one synthetic commit so this is a no-op locally
# but high-leverage live. Cursor's reference patches ARE recent commits in this
# codebase's style — showing the model 1-2 actual examples teaches the codebase's
# idioms (variable conventions, hunk shape, test-touch patterns) far better than
# any abstract prompt rule.
_RECENT_COMMIT_MAX_INSERTIONS = 30
# v1.2: tighter caps. With the new (A) lockfile-only filter and (B)
# issue-aware ranking each surviving commit is targeted, so doubling its
# diff length doesn't double its signal — keep them lean.
_RECENT_COMMIT_MAX_DIFF_CHARS = 1800   # v1.1 was 3500
_RECENT_COMMIT_BLOCK_BUDGET = 2800     # v1.1 was 4500
_RECENT_COMMIT_POOL_SIZE = 6           # v1.2: candidate pool for issue-aware re-rank

# repo_summary safety net: GitHub-mined tasks always start from a clean
# snapshot so `git status --short` is empty and was producing 11 lines of
# pure boilerplate per run. Off by default; flip on if a future harness
# stages partially-applied state and we want it surfaced to the model.
ENABLE_UNCOMMITTED_STATUS = False

# v1.2 prompt-section toggles. The user prompt is now assembled section by
# section (file snippets / recent commits) instead of one fat preloaded blob;
# each section can be turned off without touching code.
ENABLE_PRELOADED_CONTEXT = True
ENABLE_RECENT_COMMITS = True

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
        ":(exclude,glob)**/__pycache__/**",
        ":(exclude,glob)**/.pytest_cache/**",
        ":(exclude,glob)**/node_modules/**",
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

    cleaned = _strip_mode_only_file_diffs(diff_output)
    return _strip_low_signal_hunks(cleaned)


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
    if path.suffix == ".pyc":
        return True
    return any(part in {"__pycache__", ".pytest_cache", "node_modules", ".git"} for part in path.parts)


def get_repo_summary(repo: Path) -> str:
    """Plain-markdown description of the repo state at agent start.

    Replaces the previous ``format_observation``-based output which produced
    ~250 lines of ``COMMAND: / EXIT_CODE: / DURATION_SECONDS:`` shell-log
    boilerplate per run. Concretely:

      * cwd is taken from ``repo`` directly - no ``pwd`` shell call.
      * The file list is filtered through ``_context_file_allowed`` so the
        agent sees only readable source/text/config files. Build artifacts
        (``.class``, ``.jar``, ``.pyc``, ...) and binary assets (``.jpg``,
        ``.gif``, ``.eps``, ...) are excluded; they were prompt-budget
        noise.
      * ``git status --short`` is dropped by default (empty for mined
        tasks); the ``ENABLE_UNCOMMITTED_STATUS`` toggle re-adds a
        one-line note when the working tree is not clean.

    Cap of 250 listed entries keeps the summary bounded on huge repos.
    """
    tracked = _tracked_files(repo)
    readable = [p for p in tracked if _context_file_allowed(p)]

    lines: List[str] = [
        f"Working directory: {repo}",
        (
            f"Tracked files: {len(tracked)} total, {len(readable)} "
            "readable (filtered to source/text/config)"
        ),
        "",
    ]

    cap = 250
    if len(readable) <= cap:
        lines.extend(readable)
    else:
        lines.extend(readable[:cap])
        lines.append(f"... and {len(readable) - cap} more readable files")

    if ENABLE_UNCOMMITTED_STATUS:
        status = _uncommitted_status_short(repo)
        if status:
            lines.append("")
            lines.append("Uncommitted changes (working tree is not clean):")
            lines.extend(status.splitlines()[:20])

    return "\n".join(lines)


def _uncommitted_status_short(repo: Path) -> str:
    """Return ``git status --short`` output, stripped, or ``""`` on error.

    Used only when ``ENABLE_UNCOMMITTED_STATUS`` is set. Kept as a tiny
    dedicated helper so ``get_repo_summary()`` doesn't mix shell I/O with
    formatting logic.
    """
    try:
        proc = subprocess.run(
            ["git", "status", "--short"],
            cwd=str(repo),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
    except Exception:
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


TEXT_FILE_EXTENSIONS = {
    # Web / script
    ".css", ".html", ".js", ".jsx", ".scss", ".svelte", ".ts", ".tsx", ".vue",
    # Compiled / typed
    ".c", ".cc", ".cpp", ".cs", ".cr", ".dart", ".go", ".h", ".hpp", ".java",
    ".kt", ".rs", ".scala", ".swift", ".v", ".zig", ".nim",
    # Functional / academic
    ".clj", ".cljs", ".ex", ".exs", ".erl", ".hs", ".ml", ".mli",
    # Scripting / glue
    ".bash", ".lua", ".pl", ".pm", ".php", ".py", ".r", ".rb", ".sh", ".zsh",
    # Native-adjacent
    ".m", ".mm",
    # Data / docs / config
    ".json", ".md", ".sql", ".toml", ".txt", ".xml", ".yaml", ".yml",
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


# v1.2: project-hint block removed. The previous version dumped
# package.json / pyproject.toml / Makefile / etc. as a "project hints"
# section. The agent runs in a Docker container with Python only and no
# network — it cannot run JS / Ruby / Rust / Go test runners regardless
# of what their config files say. The block was paying ~2 600 tokens per
# run for hints the agent could not act on; removed.


@dataclass
class PreloadedSections:
    """v1.2: named sections of the user prompt instead of one fat string.

    ``build_preloaded_context`` used to concatenate file snippets, project
    hints, and recent-commit examples into a single ``preloaded_context``
    blob. That made each part hard to A/B-toggle, hard to budget
    independently, and produced double-headers when the assembler re-wrapped
    the bundle. v1.2 returns these named fields so the prompt assembler
    renders each section selectively, with its own header.
    """
    file_snippets: str = ""
    recent_commits: str = ""

    def has_any(self) -> bool:
        return bool(self.file_snippets.strip() or self.recent_commits.strip())


def build_preloaded_context(repo: Path, issue: str) -> PreloadedSections:
    """Preload the highest-ranked tracked files plus their companion tests.

    Two improvements over a vanilla rank-and-read loop:

      1. Companion test files (tests/test_X.py for X.py, X.test.ts for X.ts,
         X_test.go for X.go, etc.) are slotted in right after their source
         partner. Real GitHub-derived tasks almost always need source+test
         changes together; without the test in context the agent patches only
         the source and misses the companion test update.

      2. Files that match identifier-shaped symbols extracted from the issue
         text get a substantial rank boost via `_weighted_grep_score`. v1.3
         splits extraction into five typed pools (paren / string / backtick
         / pascal / snake) and weights each pool's contribution by its
         empirical precision, so a file matched by a quoted UI string
         outscores a file matched only by a bare PascalCase noun in prose.

    v1.2: returns a ``PreloadedSections`` instead of one concatenated string,
    so the prompt assembler can render file snippets and recent commits as
    independent sections with their own headers and budgets. The
    ``MAX_PRELOADED_CONTEXT_CHARS`` budget now applies only to file snippets;
    ``_RECENT_COMMIT_BLOCK_BUDGET`` already existed and now actually applies
    independently — previously a fat snippets section could starve the
    recent-commits section.
    """
    sections = PreloadedSections()

    if ENABLE_PRELOADED_CONTEXT:
        files = _rank_context_files(repo, issue)
        if files:
            tracked_set = set(_tracked_files(repo))
            files = _augment_with_test_partners(files, tracked_set)

            parts: List[str] = []
            used = 0
            per_file_budget = max(
                1500,
                MAX_PRELOADED_CONTEXT_CHARS // max(1, min(len(files), MAX_PRELOADED_FILES)),
            )

            for relative_path in files[:MAX_PRELOADED_FILES]:
                snippet = _read_context_file(repo, relative_path, per_file_budget)
                if not snippet.strip():
                    continue
                block = f"### {relative_path}\n```\n{snippet}\n```"
                if parts and used + len(block) > MAX_PRELOADED_CONTEXT_CHARS:
                    break
                parts.append(block)
                used += len(block)

            sections.file_snippets = "\n\n".join(parts)

    if ENABLE_RECENT_COMMITS:
        # Silent no-op when the repo has no real history (pilot snapshots
        # have one synthetic commit) — the helper returns "" and we add
        # nothing. v1.2 passes the issue so ranking can prefer commits that
        # touch the same files / symbols the agent is being asked to edit.
        sections.recent_commits = _recent_commit_examples(repo, issue)

    return sections


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
    # v1.3: weighted multi-pool grep replaces the single ``_symbol_grep_hits``
    # pass. Each typed pool (paren / string / backtick / pascal / snake)
    # contributes its own (base, per-extra, cap) bonus so a file matched by
    # a quoted UI string outscores a file matched only by a bare PascalCase
    # noun in prose.
    pool_score = _weighted_grep_score(repo, tracked_set, issue)
    # v1.4: path-atom overlap (prose-only recall) + recent-modification
    # tiebreaker. ``issue_atoms_set`` decomposes every issue signal into
    # ≥3-char atoms; ``_path_atom_overlap_score`` rewards files whose path
    # parts overlap with that vocabulary. ``recent_touch`` surfaces files
    # the developer was actively editing in the last 30 commits — only a
    # tiebreaker (capped at +20).
    issue_atoms_set = _issue_atoms(issue)
    recent_touch = _recently_touched_files(repo)
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
        # v1.3: drop the legacy ``+2 per term on test-shaped paths``
        # double-count. With ``terms`` capped at 20 the bonus was just
        # amplifying noise on tasks whose issue happens to mention generic
        # verbs ("render", "submit").
        score += pool_score.get(relative_path, 0)
        # v1.4: two new signals.
        score += _path_atom_overlap_score(relative_path, issue_atoms_set)
        score += _recent_modification_score(recent_touch.get(relative_path, 0))
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
    if path.suffix.lower() not in TEXT_FILE_EXTENSIONS:
        return False
    return True


def _extract_issue_path_mentions(issue: str) -> List[str]:
    pattern = re.compile(
        r"(?<![\w.-])([\w./-]+\.(?:c|cc|cpp|cs|css|go|h|hpp|html|java|js|jsx|json|kt|md|php|py|rb|rs|scss|sh|sql|svelte|swift|toml|ts|tsx|txt|vue|xml|ya?ml))(?![\w.-])",
        re.IGNORECASE,
    )
    mentions: List[str] = []
    for match in pattern.finditer(issue):
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
    # v1.3: 40 → 20. Halves the noise floor on path-substring scoring
    # without losing the high-signal top tokens. The old bottom 39 % were
    # pure English noise that gave every file mentioning common verbs a
    # faint score bump.
    return terms[:20]


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


# v2.4: ``_diff_low_signal_summary`` removed (only consumer was the
# now-deleted polish refinement turn). ``_strip_low_signal_hunks`` above
# still runs at patch-return time, so junk hunks are silently dropped
# before the validator sees them — we just no longer surface them as a
# refinement nudge. ``_patch_covers_required_paths`` /
# ``_uncovered_required_paths`` removed: the coverage refinement turn
# they fed is gone and the auto-stop heuristic that also called them
# was already removed in v2.3.


def _patch_changed_files(patch: str) -> List[str]:
    """Return the list of `b/` paths touched by a unified diff, in order."""
    seen: List[str] = []
    for match in re.finditer(r"^diff --git a/(.+?) b/(.+?)$", patch, flags=re.MULTILINE):
        path = match.group(2)
        if path and path not in seen:
            seen.append(path)
    return seen


# v2.4: multi-language syntax gate removed. The previous version ran
# ``_check_python_syntax_one`` / ``_check_node_syntax_one`` /
# ``_check_json_syntax_one`` / ``_check_brace_balance_one`` after every
# draft and queued a syntax_fix refinement turn on any error. The LLM
# judge added in v2.4 catches semantic failures the deterministic
# syntax checks couldn't, and treats real syntax errors as one of many
# correctness signals. The deterministic gate's only unique value was
# false-positive-free verdicts on `.py` / `.js` — but it also missed
# every other language. We accept the trade-off: fewer ground-truth
# checks, broader coverage via the judge.
#
# Removed: ``_SYNTAX_TIMEOUT``, ``_BRACE_BALANCE_SUFFIXES``,
# ``_check_python_syntax_one``, ``_check_node_syntax_one``,
# ``_check_json_syntax_one``, ``_check_brace_balance_one``,
# ``_check_syntax``.
#
# Also removed: ``_has_executable``, ``_shell_quote`` (only consumed by
# the syntax / companion-test runners).


# -----------------------------
# Companion-test partner templates (context preloading only — v2.4)
# -----------------------------
#
# v2.4 keeps these because ``build_preloaded_context`` slots companion
# tests into the prompt as context (LLM benefits from seeing the test
# alongside the source it is asked to fix). The runtime
# ``_run_companion_test`` / ``_select_companion_test_failure`` gate has
# been removed — see the ``_legacy_companion_test_marker`` comment further
# down for rationale.

_TEST_PARTNER_TEMPLATES: Tuple[Tuple[str, str], ...] = (
    # Python — the most common shapes.
    ("{stem}.py", "tests/test_{stem}.py"),
    ("{stem}.py", "test_{stem}.py"),
    ("{stem}.py", "{dir}/test_{stem}.py"),
    ("{stem}.py", "{dir}/tests/test_{stem}.py"),
    ("{stem}.py", "tests/{stem}_test.py"),
    # TypeScript / JavaScript — Jest / Vitest conventions.
    ("{stem}.ts", "{dir}/{stem}.test.ts"),
    ("{stem}.ts", "{dir}/__tests__/{stem}.test.ts"),
    ("{stem}.ts", "tests/{stem}.test.ts"),
    ("{stem}.tsx", "{dir}/{stem}.test.tsx"),
    ("{stem}.tsx", "{dir}/__tests__/{stem}.test.tsx"),
    ("{stem}.js", "{dir}/{stem}.test.js"),
    ("{stem}.js", "{dir}/__tests__/{stem}.test.js"),
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


# v2.4: ``_run_companion_test`` and ``_select_companion_test_failure``
# removed. They powered the test_fix refinement turn, which forked
# ``pytest`` / ``node --check`` against a partner test inside the per-step
# loop. The judge added in v2.4 now decides whether the patch is
# acceptable; if the judge wants verification it can ask the agent to run
# the test inside the agent's own bash loop. Removing the in-loop runner
# saves a subprocess + timeout per refinement cycle and removes a class of
# false positives where the partner test was intrinsically broken (failing
# on master) and the model wasted a turn "fixing" it.


def _parse_shortstat_insertions(stat_output: str) -> int:
    """Extract the ``N insertions(+)`` count from ``git show --shortstat`` output.

    Module-level helper so the recent-commit loop stays readable; previously
    inlined as a nested for/try.
    """
    for line in stat_output.splitlines():
        if "insertion" not in line:
            continue
        for word in line.split(","):
            if "insertion" in word:
                try:
                    return int(word.strip().split()[0])
                except (ValueError, IndexError):
                    return 0
    return 0


def _score_commit_for_issue(touched: List[str], diff_text: str, issue: str) -> int:
    """v1.2 (B): score a candidate commit for issue-relevance.

    Used to re-rank the recent-commit candidate pool so we surface
    examples that touch the same files / symbols the agent is being asked
    to edit. When the issue is empty (unit tests / bare CLI) every score
    is 0 and ordering collapses back to pure recency — fully
    backwards-compatible.

      * +50 per touched-path that exactly matches an ``issue`` path mention
      * +2 per ``issue`` term substring hit on a touched path
      * +3 per extracted ``issue`` symbol occurrence in the diff body
    """
    if not issue:
        return 0
    score = 0
    path_mentions = {m.strip("./") for m in _extract_issue_path_mentions(issue)}
    if path_mentions:
        for path in touched:
            if path in path_mentions:
                score += 50
    terms = _issue_terms(issue)
    if terms:
        joined_paths_lower = "\n".join(touched).lower()
        for term in terms:
            score += 2 * joined_paths_lower.count(term)
    symbols = _extract_issue_symbols(issue)
    if symbols and diff_text:
        for symbol in symbols:
            score += 3 * diff_text.count(symbol)
    return score


def _recent_commit_examples(repo: Path, issue: str = "") -> str:
    """v21 edge: read recent small-diff commits from the staged repo via git log
    and format them as in-context style anchors. Returns empty string when the
    repo has no real history (single synthetic commit in pilot snapshots), so
    this is a silent no-op locally and a real lift live where the validator
    clones the upstream repo with full history.

    The model imitates concrete examples better than abstract rules. Cursor's
    reference patch IS a one-off commit in this codebase's style; showing the
    model 1-2 real recent commits gives it the same anchor.

    v1.2 changes:
      * Returns body-only output — the prompt assembler owns the section
        header (``## Recent reference patches``) so we no longer emit a
        leading "RECENT REFERENCE PATCHES from this codebase..." prefix
        that produced double-headers.
      * (A) Relevance filter: each candidate's diff is scanned for touched
        paths via ``_patch_changed_files``; if no touched file passes
        ``_context_file_allowed`` (i.e. the commit is purely lockfile /
        generated / binary / CI-config churn), the commit is dropped.
        Lockfile-only commits were the single biggest source of useless
        ```diff blocks in v1.1.
      * (B) Issue-aware ranking: build a small candidate pool of size
        ``_RECENT_COMMIT_POOL_SIZE`` that pass per-commit filters, score
        each via ``_score_commit_for_issue``, and pick the top 2.
    """
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

        # Phase 1: build a candidate pool that survives per-commit filters.
        # Each entry is (sha, diff_text, touched_allowed_paths).
        pool: List[Tuple[str, str, List[str]]] = []
        for sha in shas:
            if len(pool) >= _RECENT_COMMIT_POOL_SIZE:
                break
            stat_proc = subprocess.run(
                ["git", "show", "--no-merges", "--shortstat", "--pretty=format:", sha],
                cwd=str(repo),
                capture_output=True,
                text=True,
                timeout=10,
            )
            if stat_proc.returncode != 0:
                continue
            insertions = _parse_shortstat_insertions(stat_proc.stdout)
            if insertions == 0 or insertions > _RECENT_COMMIT_MAX_INSERTIONS:
                continue
            # NOTE: previous version passed --pretty=format:%s which caused
            # `git show` to emit the commit subject in place of the standard
            # header but git still appended the diff. --pretty=format:
            # empties the header entirely so we keep just the diff body.
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
            # (A) Relevance filter: drop commits with no source-file touches.
            touched = _patch_changed_files(diff_text)
            allowed_touched = [p for p in touched if _context_file_allowed(p)]
            if not allowed_touched:
                continue
            pool.append((sha, diff_text, allowed_touched))

        if not pool:
            return ""

        # Phase 2: (B) issue-aware ranking. Stable: ties keep recency order
        # because Python sort is stable and the pool was built in recency
        # order.
        pool.sort(
            key=lambda entry: -_score_commit_for_issue(entry[2], entry[1], issue),
        )

        examples: List[str] = []
        budget_used = 0
        for _sha, diff_text, _touched in pool:
            block = f"```diff\n{diff_text[:_RECENT_COMMIT_MAX_DIFF_CHARS]}\n```"
            if budget_used + len(block) > _RECENT_COMMIT_BLOCK_BUDGET:
                continue
            examples.append(block)
            budget_used += len(block)
            if len(examples) >= 2:
                break

        if not examples:
            return ""
        # Body-only — assembler owns the section header now.
        return "\n\n".join(examples)
    except Exception:
        return ""


# v2.4: criteria-nudge support removed. The previous version extracted
# acceptance-criterion bullets from the issue (``_extract_acceptance_criteria``),
# tokenised each (``_criterion_keywords`` + ``_KEYWORD_SUFFIX_STRIPS``),
# then keyword-searched the patch's added lines (``_keyword_in_added`` +
# ``_patch_added_text``) and surfaced the unmatched ones via a
# criteria_nudge refinement turn. The judge added in v2.4 evaluates
# coverage holistically (it sees the full patch + the full issue and
# decides whether every requirement is addressed), making the
# keyword-overlap heuristic redundant. Removed:
# ``_CRITERIA_MAX_BULLETS`` / ``_CRITERIA_MAX_TEXT`` / ``_CRITERIA_STOP``,
# ``_extract_acceptance_criteria``, ``_criterion_keywords``,
# ``_KEYWORD_SUFFIX_STRIPS``, ``_keyword_in_added``,
# ``_patch_added_text``, ``_unaddressed_criteria``.


# -----------------------------
# v1.4: path-atom inverted index + recent-modification boost
# -----------------------------
#
# v1.3 already extracts identifier-shaped tokens cleanly, but tasks where
# the LLM described changes in *prose* (no quoted symbols) still landed
# only 1–7 of their changed files in the top-30 because there was nothing
# to extract. v1.4 attacks the same problem from a different angle:
# instead of trying to extract richer tokens from prose, score tracked-
# file paths by overlap with the *domain words the issue uses
# repeatedly* — and add a small "recently-edited code is hotter"
# tiebreaker for tasks where even that fails (e.g. non-English issue
# text against English path stems).

_PATH_ATOM_STOP = frozenset({
    # Universal noise atoms — would inflate overlap counts on every repo.
    "src", "app", "lib", "main", "index", "public", "private", "static",
    "core", "common", "shared", "util", "utils", "helper", "helpers",
    "base", "default", "config", "configs", "settings",
    "test", "tests", "spec", "specs", "fixture", "fixtures", "mock", "mocks",
    "build", "dist", "out", "bin", "obj", "vendor", "node_modules",
    "doc", "docs", "readme", "license",
    "new", "old", "tmp", "temp", "data", "file", "files", "code",
    "the", "and", "for", "with", "from", "this", "that",
})

_PATH_CAMEL_SPLIT_RE = re.compile(
    r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+"
)


def _split_path_parts(text: str) -> List[str]:
    """Split an arbitrary identifier or path segment into lowercased atoms.

    Step 1: split on non-alphanumerics (snake / kebab / dot / slash collapse).
    Step 2: CamelCase split inside each chunk so ``XMLHttpRequest`` →
    ``xml`` / ``http`` / ``request``.
    """
    parts: List[str] = []
    for chunk in re.split(r"[^A-Za-z0-9]+", text):
        if not chunk:
            continue
        for sub in _PATH_CAMEL_SPLIT_RE.findall(chunk):
            if sub:
                parts.append(sub.lower())
    return parts


def _atomize(token: str) -> List[str]:
    """Decompose ``token`` into ≥3-char lowercased atoms, filtering stopwords."""
    out: List[str] = []
    seen: set = set()
    for atom in _split_path_parts(token):
        if len(atom) < 3:
            continue
        if atom in _PATH_ATOM_STOP:
            continue
        if atom in seen:
            continue
        seen.add(atom)
        out.append(atom)
    return out


def _atoms_match(a: str, b: str) -> bool:
    """Equality OR substring-overlap when the shorter atom is ≥4 chars.

    Handles plurals (``service`` ↔ ``services``) and stem prefixes
    (``auth`` ↔ ``authentication``) without a real stemmer. The 4-char
    floor blocks accidental matches like ``pos`` ↔ ``position``.
    """
    if a == b:
        return True
    if len(a) > len(b):
        a, b = b, a  # ensure ``a`` is the shorter atom
    if len(a) < 4:
        return False
    return a in b


def _issue_atoms(issue_text: str) -> frozenset:
    """Deduped union of ``_atomize(token)`` over every signal source the
    issue produces: path mentions, all five typed symbol pools, and terms.

    Returned as a ``frozenset`` so set operations against ``path_atoms``
    are O(1) per atom.
    """
    if not issue_text:
        return frozenset()
    sources: List[str] = []
    sources.extend(_extract_issue_path_mentions(issue_text))
    pools = _extract_symbol_pools(issue_text)
    for pool_tokens in pools.values():
        sources.extend(pool_tokens)
    sources.extend(_issue_terms(issue_text))
    out: set = set()
    for tok in sources:
        for atom in _atomize(tok):
            out.add(atom)
    return frozenset(out)


def _path_atoms(relative_path: str) -> List[str]:
    """Atom decomposition of a tracked-file path, with stem CamelCase split.

    Cached at module scope via lru_cache wouldn't help here (we'd just
    re-iterate the tracked-file list once), so the function stays a plain
    helper.
    """
    return _atomize(relative_path)


def _recently_touched_files(repo: Path, commit_limit: int = 30) -> Dict[str, int]:
    """v1.4: ``{path: touch_count}`` for tracked, context-allowed files
    appearing in the last ``commit_limit`` non-merge commits.

    Returns ``{}`` silently on git failure or empty history. 4-second
    timeout so a slow repo never wedges ranking. Used by
    ``_recent_modification_score`` as a tiebreaker — never as the only
    signal.
    """
    try:
        proc = subprocess.run(
            [
                "git", "log", "--no-merges",
                "-n", str(commit_limit),
                "--name-only", "--pretty=format:",
            ],
            cwd=str(repo),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=4,
        )
    except Exception:
        return {}
    if proc.returncode != 0 or not proc.stdout.strip():
        return {}
    counts: Dict[str, int] = {}
    for raw in proc.stdout.splitlines():
        path = raw.strip()
        if not path:
            continue
        if not _context_file_allowed(path):
            continue
        counts[path] = counts.get(path, 0) + 1
    return counts


# Per-file path-atom overlap curve. Single-atom matches stay mild (a
# generic word coincidence); multi-atom matches get steep (genuine phrase
# coherence). Stays under the weakest grep-pool reward (snake base +35)
# so atom overlap never dominates a real symbol match.
_ATOM_OVERLAP_BONUS: Dict[int, int] = {
    1: 8,
    2: 22,
    3: 38,
    4: 50,
}
_ATOM_OVERLAP_BONUS_MAX = 70


def _path_atom_overlap_score(
    relative_path: str,
    issue_atoms_set: frozenset,
) -> int:
    """v1.4: bonus from ``issue_atoms ∩ path_atoms`` for a single file."""
    if not issue_atoms_set:
        return 0
    path_atoms = _path_atoms(relative_path)
    if not path_atoms:
        return 0
    matched: set = set()
    for p_atom in path_atoms:
        for i_atom in issue_atoms_set:
            if _atoms_match(p_atom, i_atom):
                matched.add(p_atom)
                break
    overlap = len(matched)
    if overlap == 0:
        return 0
    if overlap >= 5:
        return _ATOM_OVERLAP_BONUS_MAX
    return _ATOM_OVERLAP_BONUS[overlap]


def _recent_modification_score(touched_count: int) -> int:
    """v1.4: modest tiebreaker. Any file touched in the last 30 commits
    gets ``+5``, plus ``+6`` per additional appearance, capped at ``+20``.

    Below the weakest grep-pool reward (``snake_symbols`` base ``+35``)
    so recency never overrides semantic match — it only breaks ties
    between two domain-equivalent files.
    """
    if touched_count <= 0:
        return 0
    return min(20, 5 + 6 * max(0, touched_count - 1))


# -----------------------------
# Issue-symbol grep ranking — v1.3 typed pools
# -----------------------------
#
# `_rank_context_files` already weighs files by issue-mentioned paths and term
# overlap. For multi-file repos that's not enough — a one-line bug fix often
# names a function or class without mentioning the file. We extract identifier-
# shaped tokens from the issue and grep the repo for them; files that contain
# those identifiers get a context-rank boost.
#
# v1.3 attacks three problems with the v1.2 single-list approach:
#
#   1. The single ``_extract_issue_symbols`` regex was capped at 12 tokens
#      and treated every identifier the same regardless of its typographic
#      envelope (backtick? paren? quoted string? bare PascalCase?).
#   2. ``_issue_terms`` was a flat 40-token list whose bottom ~40 % was
#      pure English noise. (Now capped at 20 — see ``_issue_terms``.)
#   3. ``_rank_context_files`` ran one ``_symbol_grep_hits`` pass over the
#      whole 12-symbol set and added the same flat bonus regardless of
#      whether the matching symbol came from a high-confidence source
#      (`` `Picture` ``, ``(EntityList.vue)``) or a low-confidence one
#      (a bare PascalCase noun in prose).
#
# Solution: split extraction into five typed pools (highest-precision
# first), then weight each pool's grep contribution differently in
# ``_weighted_grep_score``.

# `\`...\`` spans — split the body of a backtick span into identifier-
# shaped sub-tokens (``\`mirrorVertical(Picture source)\``` →
# ``mirrorVertical``, ``Picture``, ``source``).
_BACKTICK_SPAN_RE = re.compile(r"`([^`\n]{2,80})`")
_BACKTICK_SUBTOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}")

# Whitespace-free ``(token)`` payloads only. The whitespace-allowing
# version pulls in English filler like ``which``, ``uses``, ``interface``;
# restricting to no-whitespace recovers exactly the
# ``(EntityList.vue)`` / ``(CropModal)`` / ``(INTRO_LIMIT)`` shape that
# the corpus study showed as ~25 % directly file-matching.
_PAREN_SPAN_RE = re.compile(r"\(([A-Za-z_][\w./_-]{1,60})\)")

# CamelCase + single-cap-noun PascalCase (``TestPicture17``, ``Picture``,
# ``Pixel``, ``Restore``). Bare lowercase identifiers are not in this
# pool — they go through ``_SNAKE_RE`` only when they have an underscore.
_PASCAL_RE = re.compile(r"(?<![A-Za-z0-9_])([A-Z][A-Za-z0-9]{2,})(?![A-Za-z0-9_])")
_PASCAL_STOP = frozenset({
    "The", "This", "That", "These", "Those", "There", "Their", "They",
    "Implement", "Currently", "Acceptance", "Major", "Minor", "Note",
    "Otherwise", "Furthermore", "Moreover", "However", "Also", "And",
    "When", "While", "Where", "What", "Which", "Who", "How", "Why",
    "Should", "Must", "Could", "Would", "Will", "Can", "May",
    "Add", "Update", "Fix", "Remove", "Make", "Use", "Create", "Ensure",
    "Support", "Allow", "Handle", "Change", "Set", "Include", "Issue",
    "Task", "File", "Code", "Function", "Class", "Method", "Module",
    "Type", "Value", "Field", "Header", "Default", "Expected", "Required",
})

# `snake_case` tokens with at least one underscore-separated segment
# (Python functions, DB columns, CLI subcommand names like
# ``match_interactions``).
_SNAKE_RE = re.compile(r"(?<![A-Za-z0-9_])([a-z][a-z0-9]*(?:_[a-z0-9]+)+)(?![A-Za-z0-9_])")

# Quoted strings (double + single) and ``--kebab`` flags, in one pool.
# Captures the test3 ``"Restore"`` / ``"Send Intro"`` UI-string case and
# the test10 ``--storage-gb`` / ``--region`` CLI-flag case.
_DOUBLE_QUOTED_RE = re.compile(r'"([^"\n]{2,40})"')
_SINGLE_QUOTED_RE = re.compile(r"'([^'\n]{2,40})'")
_KEBAB_FLAG_RE = re.compile(r"(?<![A-Za-z0-9_])(--[a-z][a-z0-9-]{1,40})(?![A-Za-z0-9_])")

# Per-pool caps. Worst-case unique-symbol count is the sum of these
# (~86); in practice each task populates only two or three pools.
_POOL_CAPS = {
    "paren": 16,
    "string": 16,
    "backtick": 18,
    "pascal": 24,
    "snake": 12,
}

# Pool order = highest-precision first. ``_extract_symbol_pools`` claims
# tokens greedily so each unique token is git-grep'd at most once per
# ``_rank_context_files`` call.
_POOL_ORDER: Tuple[str, ...] = ("paren", "string", "backtick", "pascal", "snake")

# Weight tuples: (base, per-extra-hit, per-file cap).
_POOL_WEIGHTS: Dict[str, Tuple[int, int, int]] = {
    "paren":    (95, 12, 45),
    "string":   (80, 10, 40),
    "backtick": (70,  9, 35),
    "pascal":   (45,  6, 24),
    "snake":    (35,  5, 20),
}

# Universal noise words to drop from every pool. Same set the legacy
# ``_SYMBOL_STOP`` filtered, kept here so it still applies to PascalCase /
# snake_case extractions.
_TYPED_POOL_STOP = frozenset({
    "about", "after", "alert", "argument", "before", "build", "called", "change",
    "check", "class", "code", "command", "config", "context", "default", "expect",
    "expected", "fail", "false", "field", "fields", "file", "files", "fix",
    "fixed", "function", "given", "global", "header", "headers", "import",
    "issue", "method", "module", "needed", "needs", "object", "params", "parse",
    "path", "patch", "production", "project", "property", "public", "remove",
    "reset", "return", "should", "static", "string", "support", "test", "tests",
    "their", "there", "thing", "this", "true", "type", "types", "update",
    "using", "value", "values", "when", "with", "will", "without", "write",
})


def _pool_token_ok(token: str) -> bool:
    """Universal token-level filter applied to every pool extraction."""
    if not token:
        return False
    if len(token) < 3:
        return False
    if token.lower() in _TYPED_POOL_STOP:
        return False
    return True


def _extract_backtick_symbols(issue_text: str) -> List[str]:
    out: List[str] = []
    seen: set = set()
    for span_match in _BACKTICK_SPAN_RE.finditer(issue_text):
        body = span_match.group(1)
        for sub_match in _BACKTICK_SUBTOKEN_RE.finditer(body):
            tok = sub_match.group(0)
            if not _pool_token_ok(tok) or tok in seen:
                continue
            seen.add(tok)
            out.append(tok)
            if len(out) >= _POOL_CAPS["backtick"]:
                return out
    return out


def _extract_paren_symbols(issue_text: str) -> List[str]:
    out: List[str] = []
    seen: set = set()
    for match in _PAREN_SPAN_RE.finditer(issue_text):
        tok = match.group(1)
        if not _pool_token_ok(tok) or tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
        if len(out) >= _POOL_CAPS["paren"]:
            break
    return out


def _extract_pascal_symbols(issue_text: str) -> List[str]:
    out: List[str] = []
    seen: set = set()
    for match in _PASCAL_RE.finditer(issue_text):
        tok = match.group(1)
        if tok in _PASCAL_STOP:
            continue
        if not _pool_token_ok(tok) or tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
        if len(out) >= _POOL_CAPS["pascal"]:
            break
    return out


def _extract_snake_symbols(issue_text: str) -> List[str]:
    out: List[str] = []
    seen: set = set()
    for match in _SNAKE_RE.finditer(issue_text):
        tok = match.group(1)
        if not _pool_token_ok(tok) or tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
        if len(out) >= _POOL_CAPS["snake"]:
            break
    return out


def _extract_string_literals(issue_text: str) -> List[str]:
    """Combined pool: double-quoted spans + single-quoted spans + ``--kebab``
    flags. All three carry the same "user wrote this exact string for a
    reason" signal."""
    out: List[str] = []
    seen: set = set()

    def _add(tok: str) -> bool:
        if not _pool_token_ok(tok) or tok in seen:
            return False
        seen.add(tok)
        out.append(tok)
        return len(out) >= _POOL_CAPS["string"]

    for match in _DOUBLE_QUOTED_RE.finditer(issue_text):
        if _add(match.group(1).strip()):
            return out
    for match in _SINGLE_QUOTED_RE.finditer(issue_text):
        if _add(match.group(1).strip()):
            return out
    for match in _KEBAB_FLAG_RE.finditer(issue_text):
        if _add(match.group(1)):
            return out
    return out


def _extract_symbol_pools(issue_text: str) -> Dict[str, List[str]]:
    """Build all five typed pools, claiming each unique token greedily by the
    higher-precision pool first.

    Returned dict has keys in ``_POOL_ORDER`` (highest-precision first).
    """
    pools: Dict[str, List[str]] = {
        "backtick": _extract_backtick_symbols(issue_text),
        "paren":    _extract_paren_symbols(issue_text),
        "pascal":   _extract_pascal_symbols(issue_text),
        "snake":    _extract_snake_symbols(issue_text),
        "string":   _extract_string_literals(issue_text),
    }
    claimed: set = set()
    deduped: Dict[str, List[str]] = {name: [] for name in _POOL_ORDER}
    for name in _POOL_ORDER:
        for tok in pools[name]:
            if tok in claimed:
                continue
            claimed.add(tok)
            deduped[name].append(tok)
    return deduped


def _extract_issue_symbols(issue_text: str, *, max_symbols: int = 12) -> List[str]:
    """Backwards-compat: deduped union of typed pools, capped at
    ``max_symbols``, in highest-precision-first order.

    Used by ``_score_commit_for_issue`` (recent-commit ranking) — that
    consumer still wants a flat list and gets richer extraction for free
    without any code changes.
    """
    pools = _extract_symbol_pools(issue_text)
    out: List[str] = []
    seen: set = set()
    for name in _POOL_ORDER:
        for tok in pools[name]:
            if tok in seen:
                continue
            seen.add(tok)
            out.append(tok)
            if len(out) >= max_symbols:
                return out
    return out


def _symbol_grep_hits(
    repo: Path,
    tracked_set: set,
    symbols: List[str],
) -> Dict[str, int]:
    """Count how many of ``symbols`` each tracked file references.

    Returns ``{path: hit_count}``. Skips on git-grep failure to keep the
    cycle cheap; symbol-grep is a *boost* to ranking, never the only signal.

    v1.3: signature changed from ``(repo, tracked_set, issue_text)`` to
    ``(repo, tracked_set, symbols)`` so the multi-pool weighting in
    ``_weighted_grep_score`` can call this once per pool with that pool's
    tokens. (External callers that previously passed an issue string
    should now call ``_extract_issue_symbols(issue)`` themselves.)
    """
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


def _weighted_grep_score(
    repo: Path,
    tracked_set: set,
    issue_text: str,
) -> Dict[str, int]:
    """v1.3: per-pool weighted grep score for each tracked file.

    For each typed pool (``paren`` / ``string`` / ``backtick`` / ``pascal``
    / ``snake``) we git-grep the pool's tokens against the repo and add
    ``base + min(cap, per_extra * (count - 1))`` to that file's score for
    every file that matches at least one pool token. Symbols claimed by
    a higher-precision pool are not re-grepped at lower precision (greedy
    dedup happens in ``_extract_symbol_pools``), so each unique token is
    grepped at most once.
    """
    pools = _extract_symbol_pools(issue_text)
    out: Dict[str, int] = {}
    for pool_name, pool_tokens in pools.items():
        if not pool_tokens:
            continue
        base, per_extra, cap = _POOL_WEIGHTS[pool_name]
        hits = _symbol_grep_hits(repo, tracked_set, pool_tokens)
        for path, count in hits.items():
            bonus = base + min(cap, per_extra * max(0, count - 1))
            out[path] = out.get(path, 0) + bonus
    return out


# -----------------------------
# Prompting
# -----------------------------

# MINER-EDITABLE: This prompt is the main behavior policy for the inner coding
# agent. Prompt improvements are encouraged as long as they respect the
# validator-owned boundaries above.
#
# v3.1 rewrite — re-aligned with the v2.x architecture:
#   * v2.3: ``<final>`` is the only stop signal; loop observes one command
#     per response (so multi-file "ALL files in ONE response" rule was
#     misleading and is gone).
#   * v2.4: companion-test runner deleted; the sandbox has no working
#     test runner — running tests just burns a step on a useless error.
#     ``## TESTS AND VERIFICATION`` is replaced with an explicit
#     "do not run tests" clause folded into ``## FINAL ANSWER``.
#   * The ``<plan>`` block instruction was unparsed (no validator reads
#     it); removed from the protocol section.
#   * New ``## ACCEPTANCE CRITERIA`` block targets the most common silent
#     failure mode the agent still produces — declaring ``<final>`` after
#     addressing only some of the listed criteria because the model
#     assumed related-looking code already implements them.
#   * ``## SCOPE`` consolidated with explicit ``Do:`` / ``Don't:`` lists.
#   * ``## STYLE`` extended to call out control-flow shape (loop vs
#     unrolled, comprehension vs map) and to name the preloaded recent-
#     reference patches as the best style anchor.
SYSTEM_PROMPT = '''You are an elite autonomous coding agent competing in a real GitHub issue repair benchmark.

You operate inside a real repository. You inspect the codebase, produce a patch, and finish. Your patch is scored on (1) correctness/completeness vs the issue and an LLM judge, and (2) similarity to a reference patch. Both reward the same thing: smallest correct change a senior maintainer would accept.

====================================================================
ABSOLUTE OUTPUT PROTOCOL
====================================================================

To run a shell command, emit exactly:

<command>
bash command here
</command>

The loop runs ONE command per response. Multi-file work proceeds across consecutive turns, one command per turn.

To finish, emit exactly:

<final>
brief summary of what changed
</final>

Never emit markdown fences around `<command>` or `<final>`.

Never emit `<final>` before a required code change has actually been made (the diff must be non-empty), unless the issue clearly requires no code change.

====================================================================
ISSUE CONTRACT
====================================================================

Treat the issue as a contract. Extract every requirement before editing — main task, bullet points, acceptance criteria, error messages, edge cases, and backwards-compat constraints. Treat clauses with "and / also / ensure / should / must / when / unless / only / both / all / regression / edge case / preserve" as distinct requirements. Hidden tests usually target the secondary clauses.

If the issue is ambiguous, do not ask for clarification — infer intent from nearby code, tests, and existing patterns, and pick the smallest plausible maintainer fix that preserves unrelated behavior.

Evidence priority when picking what to patch: explicit issue text > failing/expected tests > nearby tests for similar behavior > the function/class that owns the behavior > existing patterns > public API compatibility > framework conventions > general knowledge. Do not invent behavior the issue and codebase do not support.

====================================================================
ACCEPTANCE CRITERIA
====================================================================

Every requirement the issue lists is unmet by definition. The criterion is listed because its implementation is missing, incomplete, or wrong. This is the most common silent failure of this agent: assuming a listed criterion is already satisfied because related-looking code exists, then shipping an incomplete patch.

Even if you see related or similar code in the repo, do NOT assume a criterion is already implemented. Verify each one in the actual code before you decide it needs no edit.

Before emitting `<final>`:
  1. Re-read the issue. Enumerate every distinct requirement (numbered bullets, checkbox lines, "and / also / ensure / must / should" clauses).
  2. For each requirement, name the file/function/symbol that now implements it AS A RESULT OF YOUR PATCH (not pre-existing code that happens to be related).
  3. If you cannot point to a concrete change in your patch for a listed requirement, the patch is not done — keep editing.

====================================================================
INSPECTION STRATEGY
====================================================================

Inspect only what you need to locate the owner of the bug and patch safely. Order: preloaded snippets first, then one or two focused searches (`rg`, fall back to `grep -R`), then the exact target region (`sed -n '120,220p'`), then nearby tests, then call sites only if a signature/public API may change.

Avoid: re-reading preloaded files, broad recursive searches, generated/vendor output, broad test suites before a targeted fix exists.

====================================================================
ROOT CAUSE RULE
====================================================================

Patch the owner of the behavior, not a downstream symptom. Parser rejects valid input -> fix parser. Serializer omits field -> fix serializer. Cache returns stale value -> fix invalidation. CLI option ignored -> fix option parsing. Validation rejects valid case -> fix validation rule, not caller workaround.

Never hardcode the visible example unless the issue explicitly requests that exact special case. Hidden tests usually check the general behavior, not the literal example.

When several fixes are correct, choose the one that changes fewest files, smallest owning function, matches nearby style, preserves public API, uses existing helpers, and looks like the obvious five-minute maintainer patch.

====================================================================
SURGICAL EDITING
====================================================================

Change the fewest lines necessary per turn. One location per turn; multi-file work proceeds across consecutive turns. Preloaded companion tests (when present) are just another file in that sequence — edit them in their own turn when the source change requires it.

When editing with scripts, always guard replacements:

python - <<'PY'
from pathlib import Path
p = Path("path/to/file")
s = p.read_text()
old = """exact old block"""
new = """exact new block"""
if old not in s:
    raise SystemExit("old block not found")
p.write_text(s.replace(old, new, 1))
PY

Use `sed -i 's/exact old/exact new/' path/to/file` only when the substitution is uniquely scoped. Do not run broad regex replacements.

When 3+ consecutive statements share the same shape, prefer a loop / map / list comprehension / table-driven test instead of unrolled copy-paste — but only inside the code you already have to change.

====================================================================
SCOPE
====================================================================

Do:
- Edit the file(s) the issue's acceptance criteria require, plus any call site that becomes inconsistent because of an unavoidable signature change.
- Update a test file only when the issue explicitly requires editing it.
- Match the surrounding code's shape, names, and patterns (see ## STYLE).

Don't:
- Whole-file or whole-function rewrites when 1–5 lines suffice.
- Formatting churn, whitespace-only edits, blank-line shuffles, comment-only rewordings.
- Code reordering, import sorting, renames for taste, drive-by type-annotation additions.
- New helpers / abstractions / files unless the issue explicitly requires them.
- Dependency or lockfile changes, vendor/generated edits, CI config edits.
- Touching extra files just because they are nearby in the diff or in the same directory.

====================================================================
STYLE
====================================================================

Match adjacent code exactly: indentation, quotes, semicolons, trailing commas, brace placement, blank-line rhythm, naming, import grouping, error/assertion/test naming style, AND control flow (loop vs unrolled, list comprehension vs map, single-return vs early-return). If nearby code style is imperfect, follow it anyway — consistency beats personal preference.

Use the EXACT identifier names already in the codebase. If the existing code is `loadMessages`, do not introduce `load_messages`. If the existing code uses `snake_case` for fields, do not introduce `camelCase`.

The "Recent reference patches" section in your preloaded context is the best style anchor available — it shows real recent commits in this codebase's idiom (variable conventions, hunk shape, test-touch patterns). Imitate them when in doubt.

Preserve meaningful comments around changed code — section headers, TODO/FIXME, compatibility notes, public-API docs, test labels, region markers. If a comment becomes false because of your fix, update it minimally; do not delete it.

Error messages are often tested exactly. When changing one, match capitalization, punctuation, quotes, and the existing error class/type. Use the exact message from the issue if provided.

Preserve public API and backwards compatibility unless the issue explicitly requires a breaking change: function/method names, signatures, exported types, CLI flags, config keys, response shapes, error classes, schemas, file formats, env-var names.

Before finalizing, mentally check hidden-test edge cases relevant to the issue: empty/null input, missing/extra fields, duplicates, case sensitivity, unicode, path separators, async ordering, idempotency, boundary values, default config behavior, multiple instances vs one. Patch the general root behavior, not only the visible case.

====================================================================
LANGUAGE NOTES (only the load-bearing items)
====================================================================

- TypeScript/C#/Java: cascade interface/type changes to every implementer and call site; write complete method bodies (no `// similar logic` stubs); include required imports.
- C/C++: update header + implementation together; preserve const-correctness and ownership style.
- Go: keep error wrapping style; update all impacted struct literals; run `gofmt` on changed Go files.
- Rust: preserve ownership/lifetime style; do not clone just to silence borrow errors; update all struct initializers and pattern matches.
- Python: preserve existing typing style; do not add annotations to untyped code unless required; avoid broad `except Exception`; reuse existing exceptions and fixtures.
- JS/TS: preserve CJS vs ESM and async style; avoid `any` unless nearby code uses it; do not change package-manager files unless required.
- Shell/SQL: preserve POSIX/bash compatibility, quoting style, naming conventions; minimal reversible migrations only.

====================================================================
SAFETY AND ENVIRONMENT
====================================================================

Never: use sudo, delete repo files, access host secrets, modify hidden tests/evaluator files, install packages, use network outside the validator proxy, modify lockfiles or CI unless required, hardcode visible-example outputs, add sleeps to hide races. Avoid editing generated files unless the issue explicitly targets them.

====================================================================
FAILURE RECOVERY AND COMMAND ECONOMY
====================================================================

If a command fails: use the error message, run at most one focused follow-up inspection, fix the direct cause, avoid thrashing. If an edit script fails: inspect only the intended target region and correct the edit, do not rewrite the file. Do not keep running broad commands hoping something changes.

A strong solve usually shapes up as: (1) one focused inspection, (2) inspect target region, (3) apply edits across consecutive turns (one location per turn), (4) optional focused `git diff`, (5) concise `<final>`. Do not over-inspect; do not under-inspect when public APIs or hidden edge cases are at risk.

====================================================================
FINAL ANSWER
====================================================================

Do NOT run tests. The sandbox does not provide a working test runner — `pytest` / `go test` / `npm test` invocations will fail in the sandbox and produce noise the downstream LLM judge cannot act on. Decide correctness from inspection and the surrounding code's intent.

When done, emit only:

<final>
Changed [file/function] to [brief root-cause fix]. Added/updated [other file] if applicable.
</final>

Keep it short. No diffs, markdown, speculation, or extra commands after `<final>`.

You are producing the smallest complete patch most likely to match the hidden reference and satisfy every listed acceptance criterion. Find the owner. Fix the root cause. Address every criterion. Preserve everything else. Finish.'''

def build_initial_user_prompt(issue: str, repo_summary: str, preloaded_context: str = "") -> str:
    preloaded = preloaded_context if isinstance(preloaded_context, PreloadedSections) else None
    blocks: List[str] = []
    blocks.append(f"## Issue\n\n{issue.strip()}")

    if repo_summary.strip():
        blocks.append(f"## Repository summary\n\n{repo_summary.strip()}")

    if preloaded is not None:
        if preloaded.file_snippets.strip():
            blocks.append(
                "## Preloaded file snippets\n\n"
                "Likely relevant tracked-file snippets (already read for you — do not re-read):\n\n"
                + preloaded.file_snippets.strip()
            )
        if preloaded.recent_commits.strip():
            blocks.append(
                "## Recent reference patches\n\n"
                "Real recent commits from this codebase — match the shape, scale, "
                "and conventions of these examples when writing your patch:\n\n"
                + preloaded.recent_commits.strip()
            )

    blocks.append(
        "## Strategy\n\n"
        "Before planning, read the ENTIRE issue above and identify every "
        "requirement (there may be more than one). Your patch must satisfy "
        "ALL of them — the LLM judge penalizes incomplete solutions.\n\n"
        "The fix is typically in ONE specific function or block. Identify "
        "it precisely, then make the minimal edit that fixes the ROOT "
        "CAUSE.\n\n"
        "If the preloaded snippets show the target code, edit them "
        "directly — do not re-read or run broad searches first. If the "
        "target is unclear, run ONE or TWO focused grep/sed -n commands "
        "to locate it, then edit immediately.\n\n"
        "When multiple files need edits, include EVERY independent edit "
        "command in the SAME response. Do not split edits across turns.\n\n"
        "After patching, run the most targeted test available "
        "(`pytest tests/test_X.py -x -q`, `go test ./...`, etc.) to "
        "verify correctness. Then finish with <final>...</final>."
    )

    return "Fix this issue.\n\n" + "\n\n".join(blocks) + "\n"


def build_no_command_repair_prompt() -> str:
    return """Your previous response did not contain a valid <command>...</command> block or <final>...</final> block.

If the patch is complete, respond with <final>summary</final>. Otherwise continue
by issuing exactly one bash command in this format:

<command>
your command here
</command>
"""


def build_budget_pressure_weak() -> str:
    """v2.1: first edit-pressure nudge. Fires once at ``EDIT_WARN_FIRST_STEP``
    when no patch has been produced yet — soft escalation."""
    return (
        "Budget check: no repo change yet. "
        "Your next command must edit the most likely file using what you already know from the issue and preloaded snippets. "
        "A precise sed or python -c is better than another grep. Stop exploring."
    )


def build_budget_pressure_strong() -> str:
    """v2.1: subsequent edit-pressure nudges. Fires every
    ``EDIT_WARN_INTERVAL`` steps after the first weak nudge — hard escalation."""
    return (
        "Hard budget check: still no patch. "
        "Your next command MUST make a code change — even a best-effort minimal edit to the most obvious location. "
        "Do not read files or run tests until after a patch exists. "
        "Use `sed -i` or a python one-liner to make the targeted edit now."
    )


def _should_nudge_for_edit(step: int, has_patch_ever: bool) -> Optional[str]:
    """v2.1: compute whether to inject an edit-pressure nudge this step.

    Returns the nudge prompt text or ``None``. Schedule:
      * any step where ``has_patch_ever`` is True → silent (the run already
        made progress; no point bullying the model);
      * ``step < EDIT_WARN_FIRST_STEP`` → silent (too early — model still
        grounding on issue and preloaded snippets);
      * ``step == EDIT_WARN_FIRST_STEP`` → weak nudge;
      * ``step > EDIT_WARN_FIRST_STEP`` and
        ``(step - EDIT_WARN_FIRST_STEP) % EDIT_WARN_INTERVAL == 0``
        → strong nudge;
      * else → silent.
    """
    if has_patch_ever:
        return None
    if step < EDIT_WARN_FIRST_STEP:
        return None
    if step == EDIT_WARN_FIRST_STEP:
        return build_budget_pressure_weak()
    if (step - EDIT_WARN_FIRST_STEP) % EDIT_WARN_INTERVAL == 0:
        return build_budget_pressure_strong()
    return None


# v2.4: ``build_polish_prompt`` / ``build_coverage_nudge_prompt`` /
# ``build_self_check_prompt`` / ``build_syntax_fix_prompt`` /
# ``build_criteria_nudge_prompt`` / ``build_hail_mary_prompt`` /
# ``build_test_fix_prompt`` removed. Each fed one branch of the 7-check
# refinement ladder; the ladder is now a single LLM-judge gate. Hail-mary,
# polish, coverage, criteria, self-check are all subsumed into what the
# judge is asked to evaluate.
#
# v3.2: the v2.4 binary PASS/FAIL judge (``judge_acceptance``) is replaced
# with a scored final-review judge (``judge_final``). The judge now emits
# a 0-100 SCORE, an explicit VERDICT, a prose REASON, and a concrete
# FIX_LIST of bullets the agent should address. Two structural problems
# the binary judge had:
#   * no nuance: a 95-point patch and a 10-point patch both produced
#     "PASS"; one missing-comma diff and one wrong-target diff both
#     produced "FAIL". One bit of information per retry meant the retry
#     often fixed the wrong thing.
#   * acceptance criteria only: style violations and obvious correctness
#     bugs had no explicit weight in the judge prompt and got picked up
#     unevenly.
# The new rubric scores three dimensions in priority order (acceptance >
# correctness > style+scope) with explicit score anchors, and the
# FIX_LIST gives the retry-turn agent something concrete to act on.


# -----------------------------
# v3.2: LLM final-review judge (scored, multi-dimensional rubric)
# -----------------------------


@dataclass
class FinalVerdict:
    """Structured judge result. ``passed`` already incorporates the
    explicit VERDICT plus the score-vs-threshold check, so callers only
    need to read this one bool to decide whether to ship.

    ``score`` is 0-100 (clamped). ``reason`` is the judge's prose summary.
    ``fix_list`` is the list of concrete bullets the judge asked the
    agent to address; empty on PASS, normally non-empty on FAIL (an
    empty fix_list on FAIL is treated as PASS upstream to avoid a wasted
    retry turn).
    """

    passed: bool
    score: int
    reason: str
    fix_list: List[str]


FINAL_CHECK_SYSTEM = (
    "You are a strict, concise final-review judge. You receive:\n"
    "  - the original task (issue text),\n"
    "  - the agent's last message,\n"
    "  - the agent's proposed patch as a unified diff.\n\n"
    "Score the patch 0-100 using this rubric, in priority order:\n\n"
    "1. ACCEPTANCE CRITERIA (highest weight). Does the patch contain a\n"
    "   concrete edit that addresses every requirement / criterion in\n"
    "   the issue? Missing or partial coverage of any criterion is the\n"
    "   heaviest penalty. Do NOT assume a criterion is already\n"
    "   satisfied just because related code exists in the repo - if a\n"
    "   criterion is listed it was unmet at the start of the task.\n\n"
    "2. CORRECTNESS. Does the patch edit the right file / function /\n"
    "   symbol, without obvious bugs (off-by-one, wrong condition,\n"
    "   swapped args, broken call signature, missing import)? Obvious\n"
    "   syntax breaks fall here too.\n\n"
    "3. STYLE + SCOPE. Does the patch match the surrounding code's\n"
    "   indentation / quoting / brace / control-flow conventions, use\n"
    "   the EXACT names already in the codebase, and avoid unnecessary\n"
    "   changes (whitespace-only edits, gratuitous refactoring, new\n"
    "   helpers, error handling not asked for)?\n\n"
    "Score anchors:\n"
    "  - 90-100: every criterion addressed correctly, clean style, minimal scope.\n"
    "  - 70-89:  every criterion addressed but with minor style or scope issues.\n"
    "  - 50-69:  most criteria addressed; one missing/partial OR a real bug.\n"
    "  - 0-49:   multiple missing criteria OR wrong-target edits OR broken syntax.\n\n"
    "Be precise and concrete. Prefer FAIL when in doubt about ACCEPTANCE\n"
    "CRITERIA coverage; STYLE issues alone are rarely a FAIL.\n\n"
    "Respond with exactly the format requested by the user message and\n"
    "nothing else."
)


def build_final_check_prompt(
    issue_text: str,
    last_response: str,
    patch: str,
) -> str:
    """User message for the final-review judge. Quotes the issue, the
    agent's final assistant turn, and the patch — each truncated to a
    per-section budget so the judge call stays bounded regardless of
    how verbose the agent was. Demands the SCORE / VERDICT / REASON /
    FIX_LIST format the judge system prompt requires.

    The patch is mid-truncated (5000 head + 2500 tail) so the judge sees
    both the start of the diff and the trailing edits even on sprawling
    multi-file changes.
    """
    issue_short = issue_text[:FINAL_CHECK_MAX_ISSUE_CHARS]
    response_short = last_response[:FINAL_CHECK_MAX_RESPONSE_CHARS]
    if len(patch) <= FINAL_CHECK_MAX_PATCH_CHARS:
        diff_view = patch
    else:
        diff_view = patch[:5000] + "\n...[diff truncated]...\n" + patch[-2500:]
    return (
        "TASK / ISSUE:\n"
        f"{issue_short}\n\n"
        "AGENT'S LAST MESSAGE:\n"
        f"{response_short}\n\n"
        "PATCH (unified diff):\n"
        "```diff\n"
        f"{diff_view}\n"
        "```\n\n"
        "Score and judge the patch using the rubric in your system\n"
        "prompt. Reply using EXACTLY this format and nothing else:\n\n"
        "SCORE: <integer 0-100>\n"
        "VERDICT: PASS|FAIL\n"
        "REASON: <one or two sentences; what is right and what is wrong>\n"
        "FIX_LIST:\n"
        "- <concrete bullet naming the file / symbol / criterion to fix>\n"
        "- <bullet>\n"
        "- ...\n\n"
        "Rules:\n"
        "- Use VERDICT: PASS only when every requirement is addressed\n"
        "  correctly and there are no obvious bugs or wrong-target edits.\n"
        "- On VERDICT: FAIL the FIX_LIST must contain at least one\n"
        "  concrete bullet that names the file / symbol / criterion to\n"
        "  fix; vague bullets (\"improve quality\") are not actionable.\n"
        "- On VERDICT: PASS the FIX_LIST may be empty (write a single\n"
        '  "- (none)" bullet).\n'
    )


def build_final_fix_prompt(
    score: int,
    reason: str,
    fix_list: List[str],
    issue_text: str,
) -> str:
    """User message queued back to the agent when the judge fails.
    Quotes the score, prose reason, AND the concrete fix bullets verbatim
    into the retry turn — LLMs act on bullet lists much more reliably
    than on prose paragraphs."""
    issue_short = issue_text[:1500]
    if fix_list:
        bullets = "\n".join(f"- {item}" for item in fix_list)
    else:
        bullets = "- (judge gave no concrete bullets; use REASON above)"
    return (
        f"Final check FAILED (score {score}/100). Judge said:\n\n"
        f"{reason}\n\n"
        "Things to fix:\n"
        f"{bullets}\n\n"
        "Address every item above with one or more <command> blocks "
        "(use sed or `python -c` for surgical edits) and then end with "
        "<final>summary</final>. Do NOT add scope the task did not "
        "ask for; fix only the named gaps and bugs.\n\n"
        "Task (for reference):\n"
        f"{issue_short}\n"
    )


# v3.2: parser regexes — pre-compiled so judging stays cheap on retries.
_FINAL_SCORE_RE = re.compile(r"(?im)^\s*SCORE\s*:\s*(\d{1,3})\s*$")
_FINAL_VERDICT_RE = re.compile(r"(?im)^\s*VERDICT\s*:\s*(PASS|FAIL)\s*$")
_FINAL_REASON_RE = re.compile(
    r"(?ims)^\s*REASON\s*:\s*(.+?)(?:\n\s*FIX[_ ]LIST\s*:|\n\s*VERDICT\s*:|\n\s*SCORE\s*:|\Z)"
)
_FINAL_FIX_LIST_RE = re.compile(r"(?ims)^\s*FIX[_ ]LIST\s*:\s*\n(.+?)\Z")
_FINAL_BULLET_RE = re.compile(r"(?m)^\s*-\s+(.+?)\s*$")
_FINAL_EMPTY_BULLETS = frozenset({"(none)", "none", "n/a", "na", "-"})


def _parse_final_verdict(text: str) -> FinalVerdict:
    """Extract SCORE / VERDICT / REASON / FIX_LIST from the judge response.

    Permissive on every parse failure (each maps to ``passed=True`` so a
    flaky judge can never block shipping):
      * empty / whitespace-only response
      * response missing SCORE or VERDICT
      * SCORE not parseable as an integer in [0, 100]
      * VERDICT: FAIL with empty FIX_LIST after dedup
        (nothing actionable, retry would be wasted)
    """
    if not text or not text.strip():
        return FinalVerdict(True, 100, "judge returned empty response", [])

    score_match = _FINAL_SCORE_RE.search(text)
    verdict_match = _FINAL_VERDICT_RE.search(text)
    if score_match is None or verdict_match is None:
        return FinalVerdict(
            True, 100, "judge produced no parseable score/verdict", []
        )

    try:
        score = max(0, min(100, int(score_match.group(1))))
    except ValueError:
        return FinalVerdict(True, 100, "judge score was not an integer", [])

    verdict_pass = verdict_match.group(1).upper() == "PASS"
    threshold_pass = score >= FINAL_CHECK_PASS_THRESHOLD

    reason_match = _FINAL_REASON_RE.search(text)
    reason = reason_match.group(1).strip() if reason_match else text.strip()[:300]

    fix_list: List[str] = []
    fix_list_match = _FINAL_FIX_LIST_RE.search(text)
    if fix_list_match is not None:
        for bullet_match in _FINAL_BULLET_RE.finditer(fix_list_match.group(1)):
            bullet = bullet_match.group(1).strip()
            if bullet and bullet.lower() not in _FINAL_EMPTY_BULLETS:
                fix_list.append(bullet)

    passed = verdict_pass and threshold_pass

    # Failed but nothing concrete to act on -> ship to avoid a wasted
    # retry turn. Same spirit as the other permissive fallbacks above.
    if not passed and not fix_list:
        return FinalVerdict(
            True,
            score,
            f"{reason} [no actionable fix bullets, shipping]",
            [],
        )

    return FinalVerdict(passed, score, reason, fix_list)


def judge_final(
    issue_text: str,
    last_response: str,
    patch: str,
    *,
    model: str,
    api_base: str,
    api_key: str,
) -> FinalVerdict:
    """v3.2: ask the LLM to score the current patch on a 0-100 rubric and
    declare PASS/FAIL with a concrete FIX_LIST.

    Returns a :class:`FinalVerdict`. A patch is considered ``passed``
    only when BOTH the explicit VERDICT is PASS AND the score is at
    least :data:`FINAL_CHECK_PASS_THRESHOLD`.

    Permissive on every endpoint error: network exceptions, empty
    responses, unparseable score/verdict, and FAIL-with-no-bullets all
    map to ``passed=True`` so a flaky judge can never block shipping.
    """
    if not patch.strip():
        return FinalVerdict(True, 100, "empty patch — judge skipped", [])
    try:
        response_text, _cost, _raw = chat_completion(
            messages=[
                {"role": "system", "content": FINAL_CHECK_SYSTEM},
                {
                    "role": "user",
                    "content": build_final_check_prompt(
                        issue_text, last_response, patch
                    ),
                },
            ],
            model=model,
            api_base=api_base,
            api_key=api_key,
            max_tokens=FINAL_CHECK_MAX_TOKENS,
        )
    except Exception as exc:
        return FinalVerdict(
            True,
            100,
            f"judge unreachable: {type(exc).__name__}: {str(exc)[:120]}",
            [],
        )

    return _parse_final_verdict(response_text)


# -----------------------------
# Main agent — v2.2 single-attempt
# -----------------------------
#
# v2.2: ``MultiShotSolver`` removed entirely. The previous v2.1 / v1.x line
# wrapped ``_solve_attempt`` in a 580-second multi-shot driver that
# revert-and-retried on a low-signal first attempt, with three time/threshold
# knobs (``_MULTISHOT_LOW_SIGNAL_THRESHOLD`` / ``_MULTISHOT_TOTAL_BUDGET``
# / ``_MULTISHOT_MIN_ATTEMPT_RESERVE``) and four helpers
# (``_multishot_count_substantive`` / ``_multishot_capture_head`` /
# ``_multishot_revert`` / ``_multishot_apply_patch``) — all gone.
#
# Why: the multi-shot wrapper was throwing away substantial work — full
# conversation history, refinement counters, accumulated repo state — on
# every retry, and added no measurable quality lift on the validator
# workload. Cost was non-deterministic (1 attempt or 2 depending on
# heuristics) and worst-case latency roughly doubled.
#
# v2.2 keeps the v43 patch-preserve safety net inline in ``solve()`` itself,
# so any uncaught exception from ``_solve_attempt`` still returns whatever
# patch is on disk rather than propagating to the validator. The result
# dict shape stays ``{patch, logs, steps, cost, success}`` (no
# ``multishot_*`` keys).


# MINER-EDITABLE: validator entry point. Single-attempt: ``solve(...)``
# runs ``_solve_attempt`` exactly once, with a try/except safety net that
# salvages the on-disk patch if anything raises.
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
    """Main portable interface for validators.

    v2.2: single-attempt solve. The previous multi-shot wrapper has been
    removed; this function now runs ``_solve_attempt`` exactly once. The
    v43 patch-preserve safety net is preserved inline: if anything in
    ``_solve_attempt`` raises (timeout, network, OOM, anything), we
    capture whatever is on disk at the time and return it as the patch.
    The validator scores empty patches at zero — any non-empty diff
    beats empty.
    """
    repo_obj: Optional[Path] = None
    try:
        repo_obj = _repo_path(repo_path)
    except Exception:
        pass

    try:
        return _solve_attempt(
            repo_path=repo_path,
            issue=issue,
            model=model,
            api_base=api_base,
            api_key=api_key,
            max_steps=max_steps,
            command_timeout=command_timeout,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        # v43 safety net: any uncaught exception → return on-disk patch.
        # (Don't catch BaseException — let SystemExit/KeyboardInterrupt
        # propagate so the validator can clean-kill the process.)
        salvaged = ""
        try:
            if repo_obj is not None:
                salvaged = get_patch(repo_obj)
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
    final_check_turns_used = 0  # v2.4: replaces 8 separate counters; v3.2: renamed
    consecutive_model_errors = 0
    has_patch_ever = False  # v2.1: latched once any patch lands; silences edit-nudge schedule

    # v2.1: wall-clock guard removed. The inner step loop now ends only on
    # success conditions, ``max_steps`` exhaustion, persistent
    # ``MODEL_ERROR``s, or ``MAX_NO_COMMAND_REPAIRS``. See module-level
    # comment near ``HTTP_MAX_RETRIES`` for the rationale.

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
        """v2.4 / v3.2 single-shot final-review gate.

        Replaces the 7-check ladder (hail-mary / polish / syntax / test /
        coverage / criteria / self-check) with one LLM-judge call. v3.2
        upgrades the binary PASS/FAIL judge into a 0-100 SCORE +
        VERDICT + REASON + FIX_LIST contract: the FAIL retry turn now
        carries concrete fix bullets so the agent can act on something
        more specific than a one-line prose reason. Returns True when
        the loop should continue (judge said FAIL and we queued a fix
        turn); False means the caller can declare success.

        Permissive on every judge error — see ``judge_final`` — so a
        flaky judge can never block shipping. Total budget is one judge
        call per solve (``MAX_FINAL_CHECK_TURNS``); after that the gate
        is silent and the model's ``<final>`` is honoured immediately.
        """
        nonlocal final_check_turns_used
        patch = get_patch(repo)

        # Empty patch: nothing to judge. The post-loop fallthrough below
        # converts this into ``success = False`` so the validator sees an
        # empty patch (which scores 0) rather than a synthetic guess.
        # (The hail-mary refinement turn this used to fire is gone — the
        # v2.4 design choice was that pure-LLM judging includes "is the
        # patch a real edit at all?", not a separate keyword-driven force-
        # an-edit prompt.)
        if not patch.strip():
            return False

        if final_check_turns_used >= MAX_FINAL_CHECK_TURNS:
            return False
        final_check_turns_used += 1

        verdict = judge_final(
            issue_text=issue,
            last_response=assistant_text,
            patch=patch,
            model=model_name,
            api_base=api_base,
            api_key=api_key,
        )
        if verdict.passed:
            logs.append(
                f"\nFINAL_CHECK_PASS (score {verdict.score}/100):\n"
                f"  {verdict.reason[:200]}"
            )
            return False

        truncated_reason = verdict.reason[:600]
        queue_refinement_turn(
            assistant_text,
            build_final_fix_prompt(
                verdict.score, verdict.reason, verdict.fix_list, issue
            ),
            f"FINAL_CHECK_FAILED (score {verdict.score}/100):\n  {truncated_reason}",
        )
        return True

    try:
        repo = _repo_path(repo_path)
        model_name, api_base, api_key = _resolve_inference_config(model, api_base, api_key)
        ensure_git_repo(repo)
        repo_summary = get_repo_summary(repo)
        preloaded = build_preloaded_context(repo, issue)

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_initial_user_prompt(issue, repo_summary, preloaded)},
        ]

        for step in range(1, max_steps + 1):
            logs.append(f"\n\n===== STEP {step} =====\n")

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
                    if retry_attempt < MAX_STEP_RETRIES:
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
                if consecutive_model_errors >= 3:
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
            # v2.3: do NOT pre-append the assistant message here. The
            # ``<final>`` + refinement path below uses
            # ``queue_refinement_turn`` which also appends the assistant —
            # pre-appending here produced a duplicate-assistant turn that
            # malformed the conversation for OpenAI-compatible APIs and
            # silently inflated token cost. The assistant is now appended
            # only on the "continue with observation" path at the bottom.
            observations: List[str] = []
            command_batch = commands[:MAX_COMMANDS_PER_RESPONSE]

            # v2.3: per-command auto-stop heuristics removed.
            #   * ``_looks_like_successful_test_output`` / ``_looks_like_patch_review_command``
            #     gates were keyword-overlap heuristics that called
            #     ``get_patch(repo)`` after every command — slow on mid-size
            #     repos and noisy ("traceback" in a passing log marked it failed).
            #   * The ``step >= 4`` / ``step >= 8`` magic numbers leaked policy
            #     into the loop.
            # ``<final>`` is now the only stop signal (plus the natural
            # post-loop fallthroughs below for empty patches / max_steps).
            for command_index, command in enumerate(command_batch, 1):
                result = run_command(command, repo, timeout=command_timeout)
                observation = format_observation(result)
                observations.append(f"OBSERVATION {command_index}/{len(command_batch)}:\n{observation}")
                logs.append(f"\nOBSERVATION {command_index}/{len(command_batch)}:\n" + observation)

            if len(commands) > len(command_batch):
                observations.append(
                    f"NOTE: Only the first {len(command_batch)} command blocks were executed. "
                    "Continue with one command at a time if more work remains."
                )

            # v2.3: ``<final>`` is honoured on its own merit. The model's
            # claim of completion is trusted; the post-loop fallthrough
            # below declares success only if a patch actually exists.
            if final is not None:
                if maybe_queue_refinement(response_text):
                    # ``queue_refinement_turn`` already appended assistant +
                    # corrective user message; skip the observation append.
                    continue
                logs.append("\nFINAL_SUMMARY:\n" + final)
                success = True

            if observations and not success:
                # v2.3: observation tail nudges removed. The "Patch now
                # exists. Next steps..." and "If you have enough context..."
                # blocks each called ``get_patch(repo)`` per step and added
                # noise to the conversation. The system prompt already
                # tells the model what to do after a patch exists.
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content": "\n\n".join(observations)})

            if success:
                break

            # v2.1: latch has_patch_ever once any patch lands so the
            # edit-pressure schedule goes silent for the rest of the run.
            if not has_patch_ever and get_patch(repo).strip():
                has_patch_ever = True

            nudge = _should_nudge_for_edit(step, has_patch_ever)
            if nudge is not None:
                messages.append({"role": "user", "content": nudge})

        patch = get_patch(repo)
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

        return AgentResult(
            patch=patch,
            logs=_safe_join_logs(logs),
            steps=0,
            cost=total_cost,
            success=False,
        ).to_dict()


# v2.3: ``_looks_like_successful_test_output`` /
# ``_looks_like_verification_command`` / ``_looks_like_patch_review_command``
# / ``_extract_observation_exit_code`` / ``_extract_observation_section``
# all removed. They existed only to power the per-command auto-stop
# heuristics in ``_solve_attempt`` which are now gone — ``<final>`` is
# the only stop signal.


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
