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
MAX_PRELOADED_CONTEXT_CHARS = 32000
MAX_PRELOADED_FILES = 10
MAX_NO_COMMAND_REPAIRS = 3
MAX_COMMANDS_PER_RESPONSE = 12

# Wall-clock self-discipline. The validator caps each round at
# min(max(2*cursor_elapsed + 1, 120), 600); we don't see that ceiling, but a
# hard 540s cap leaves ~60s of cushion below the absolute max. When less than
# WALL_CLOCK_FINALIZE_REMAINING_SECONDS remain we prefer "return the current
# patch" over "queue another refinement turn that won't finish anyway",
# because a timed-out round is a guaranteed loss for both score halves.
WALL_CLOCK_HARD_CAP_SECONDS = 540
WALL_CLOCK_FINALIZE_REMAINING_SECONDS = 60
WALL_CLOCK_PRESSURE_REMAINING_SECONDS = 120

# Refinement-turn budgets: each turn shows the model its draft and asks for one
# specific kind of correction. They are mutually exclusive so the agent never
# loops indefinitely on a borderline patch.
MAX_POLISH_TURNS = 1       # strip whitespace/comment/blank-only hunks
MAX_SELF_CHECK_TURNS = 1   # ensure issue-mentioned paths are covered, no scope creep
MAX_SYNTAX_FIX_TURNS = 1   # repair Python/TypeScript/JavaScript SyntaxError
MAX_TEST_FIX_TURNS = 1     # repair the companion test we ran ourselves

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
    max_retries: int = 1,
) -> Tuple[str, Optional[float], Dict[str, Any]]:
    """OpenAI-compatible /v1/chat/completions client.

    Retries once on transient transport failures (timeout, connection reset,
    HTTP 5xx). Client-side errors (4xx) bail out immediately because retrying
    won't change the outcome and burns wall-clock budget that the agent needs
    for actual editing.
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
            if 500 <= e.code < 600 and attempt < max_retries:
                last_error = e
                time.sleep(1.0)
                continue
            raise RuntimeError(f"HTTP {e.code} from model endpoint: {err_body}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            if attempt < max_retries:
                last_error = e
                time.sleep(1.0)
                continue
            raise RuntimeError(f"Model request failed: {e}") from e
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
PLAN_RE = re.compile(r"<plan>\s*(.*?)\s*</plan>", re.IGNORECASE | re.DOTALL)
# Path-like tokens with a known source/test extension. Used to harvest the
# files the model committed to in its <plan> block so the self-check turn can
# notice when the patch silently drops a planned target.
_PLAN_PATH_RE = re.compile(
    r"[\w][\w./-]+\.(?:c|cc|cpp|cs|css|go|h|hpp|html|java|js|jsx|json|kt|md|"
    r"php|py|rb|rs|scss|sh|sql|svelte|swift|toml|ts|tsx|txt|vue|xml|ya?ml)\b",
    re.IGNORECASE,
)


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


def extract_plan(model_text: str) -> Optional[str]:
    """Return the body of the first <plan>...</plan> block, or None."""
    match = PLAN_RE.search(model_text)
    if not match:
        return None
    return match.group(1).strip() or None


def extract_planned_paths(plan_text: str, tracked_set: set) -> List[str]:
    """Return the tracked file paths that appear in a <plan> block, deduped
    in the order the model named them.

    Models often name targets by basename only (e.g. ``UserServiceTests.cs``),
    so when an exact path-in-tracked-set match fails we fall back to a unique
    suffix scan: a single tracked path ending with ``/<token>`` resolves;
    ambiguous matches (multiple tracked files share the basename) are skipped.
    """
    if not plan_text or not tracked_set:
        return []
    paths: List[str] = []
    seen: set = set()
    for raw in _PLAN_PATH_RE.findall(plan_text):
        token = raw.strip("`'\"()[]{}:,;").strip()
        if not token:
            continue
        if token in tracked_set:
            if token not in seen:
                seen.add(token)
                paths.append(token)
            continue
        candidates = [t for t in tracked_set if t.endswith("/" + token)]
        if len(candidates) == 1:
            chosen = candidates[0]
            if chosen not in seen:
                seen.add(chosen)
                paths.append(chosen)
    return paths


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
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".java",
    ".js",
    ".jsx",
    ".json",
    ".kt",
    ".md",
    ".php",
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


def build_preloaded_context(repo: Path, issue: str) -> str:
    """Preload the highest-ranked tracked files plus their companion tests.

    Two improvements over a vanilla rank-and-read loop:

      1. Companion test files (tests/test_X.py for X.py, X.test.ts for X.ts,
         X_test.go for X.go, etc.) are slotted in right after their source
         partner. Real GitHub-derived tasks almost always need source+test
         changes together; without the test in context the agent patches only
         the source and misses the companion test update.

      2. Files that match identifier-shaped symbols extracted from the issue
         text get a substantial rank boost via `_symbol_grep_hits`. This
         catches the common case where the bug is described by function or
         class name without mentioning the file path.
    """
    files = _rank_context_files(repo, issue)
    if not files:
        return ""

    tracked_set = set(_tracked_files(repo))
    files = _augment_with_test_partners(files, tracked_set)

    parts: List[str] = []
    used = 0
    per_file_budget = max(1500, MAX_PRELOADED_CONTEXT_CHARS // max(1, min(len(files), MAX_PRELOADED_FILES)))

    # Tiered budget: concentrate chars on the most likely target so the model
    # sees a full function/component for the primary file instead of an even
    # but shallow split across N files. Lower-ranked files keep the computed
    # even-split budget so they stay informative without dominating.
    for rank, relative_path in enumerate(files[:MAX_PRELOADED_FILES]):
        if rank == 0:
            file_budget = max(per_file_budget, 12000)
        elif rank <= 2:
            file_budget = max(per_file_budget, 3500)
        else:
            file_budget = per_file_budget
        snippet = _read_context_file(repo, relative_path, file_budget)
        if not snippet.strip():
            continue
        block = f"### {relative_path}\n```\n{snippet}\n```"
        if parts and used + len(block) > MAX_PRELOADED_CONTEXT_CHARS:
            break
        parts.append(block)
        used += len(block)

    return "\n\n".join(parts)


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

    terms = _issue_terms(issue)
    symbol_hits = _symbol_grep_hits(repo, tracked_set, issue)
    # Language-density signal: in TS-heavy or Python-heavy repos, prefer the
    # dominant source extension over docs/configs at the same score so a
    # generic identifier match in a .md/.json doesn't outrank the actual
    # implementation file.
    ts_count = sum(1 for f in tracked if f.endswith((".ts", ".tsx")))
    ts_ratio = ts_count / max(len(tracked), 1)
    py_count = sum(1 for f in tracked if f.endswith(".py"))
    py_ratio = py_count / max(len(tracked), 1)
    scored: List[Tuple[int, str]] = []
    for relative_path in tracked:
        if not _context_file_allowed(relative_path):
            continue
        path_lower = relative_path.lower()
        name_lower = Path(relative_path).name.lower()
        stem_lower = Path(relative_path).stem.lower()
        suffix_lower = Path(relative_path).suffix.lower()
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
        # Language-density boost.
        if ts_ratio > 0.3 and suffix_lower in (".ts", ".tsx"):
            score += 10
        if py_ratio > 0.3 and suffix_lower == ".py":
            score += 10
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


def _missing_required_paths(patch: str, issue_text: str) -> List[str]:
    """Issue-mentioned paths that don't yet appear in the patch, in mention order."""
    required = _extract_issue_path_mentions(issue_text)
    if not required:
        return []
    changed = set(_patch_changed_files(patch))
    return [req for req in required if not any(req == c or c.endswith("/" + req) for c in changed)]


def _patch_covers_required_paths(patch: str, issue_text: str) -> bool:
    """All paths the issue explicitly mentions must appear in the patch."""
    return not _missing_required_paths(patch, issue_text)


def _companion_test_gap(repo: Path, patch: str) -> Optional[Tuple[str, str]]:
    """Return (source_path, test_path) when a patched source file has a tracked
    companion test that the patch does not also touch.

    Real GitHub-derived tasks almost always need source and test changes
    together; without the test in lockstep the patch bleeds Cursor-similarity
    score. Returns None when every patched source either has no companion test
    or its test partner is already covered by the patch.
    """
    changed = _patch_changed_files(patch)
    if not changed:
        return None
    changed_set = set(changed)
    tracked_set = set(_tracked_files(repo))
    if not tracked_set:
        return None
    for relative_path in changed:
        partner = _find_test_partner(relative_path, tracked_set)
        if partner and partner not in changed_set:
            return (relative_path, partner)
    return None


def _runnable_test_command(test_path: str) -> Optional[str]:
    """Best-effort focused test command for the given test file.

    Python is the dominant language for validator tasks; we try `pytest`
    first (preferred), then `python -m pytest`, then `python3 -m pytest`,
    and fall back to `python -m unittest` for the rare repo without pytest.
    Other ecosystems (Node, Go, Rust) need project-aware setup that we
    can't reliably infer from this file alone, so we skip them.
    """
    suffix = Path(test_path).suffix.lower()
    if suffix != ".py":
        return None
    quoted = _shell_quote(test_path)
    return (
        f"if command -v pytest >/dev/null 2>&1; then pytest -x -q {quoted} 2>&1; "
        f"elif python -c 'import pytest' >/dev/null 2>&1; then python -m pytest -x -q {quoted} 2>&1; "
        f"elif python3 -c 'import pytest' >/dev/null 2>&1; then python3 -m pytest -x -q {quoted} 2>&1; "
        f"else python -m unittest {quoted} 2>&1; fi"
    )


# Substrings that mean "the test runner couldn't even start" — we treat these
# as ambiguous (probably an environmental issue, not a real regression) so we
# don't waste a refinement turn quoting environmental noise back at the model.
_TEST_ENV_NOISE_MARKERS: Tuple[str, ...] = (
    "modulenotfounderror",
    "no module named",
    "command not found",
    "is not recognized",
    "importerror",
    "could not find a version",
)


def _run_companion_test_in_patch(
    repo: Path,
    patch: str,
    command_timeout: int,
) -> Optional[Tuple[str, str]]:
    """When a source-test pair is both in the patch, run the test and report
    the failure tail.

    Returns (test_path, output) only when the test fails with high confidence
    (a known runner summary indicates failure, or exit code is non-zero with
    no obvious environmental cause). Returns None when no pair is runnable,
    the test passes, the test times out, or the failure looks environmental.
    """
    changed = _patch_changed_files(patch)
    if not changed:
        return None
    changed_set = set(changed)
    tracked_set = set(_tracked_files(repo))
    if not tracked_set:
        return None
    pairs_checked = 0
    for relative_path in changed:
        if pairs_checked >= 2:
            break
        partner = _find_test_partner(relative_path, tracked_set)
        if not partner or partner not in changed_set:
            continue
        cmd = _runnable_test_command(partner)
        if not cmd:
            continue
        pairs_checked += 1
        result = run_command(cmd, repo, timeout=min(60, max(command_timeout * 3, 30)))
        if result.timed_out:
            continue
        observation = format_observation(result)
        obs_lower = observation.lower()
        if _parse_test_summary(observation, cmd) is False:
            output = (result.stdout or "") + (("\n" + result.stderr) if result.stderr.strip() else "")
            return (partner, output)
        if result.exit_code != 0 and not any(m in obs_lower for m in _TEST_ENV_NOISE_MARKERS):
            output = (result.stdout or "") + (("\n" + result.stderr) if result.stderr.strip() else "")
            return (partner, output)
    return None


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


def _check_typescript_syntax_one(repo: Path, relative_path: str) -> Optional[str]:
    """`tsc --noEmit --skipLibCheck file.ts` when tsc is available.

    Falls back to None when tsc is missing — TypeScript repos without a global
    `tsc` typically still parse fine through node's loader for our purposes,
    and we'd rather skip the check than drop several seconds on a NotFound.
    """
    if not _has_executable("tsc"):
        return None
    proc_result = run_command(
        f"tsc --noEmit --skipLibCheck --allowJs {_shell_quote(relative_path)}",
        repo,
        timeout=_SYNTAX_TIMEOUT * 2,
    )
    if proc_result.exit_code == 0:
        return None
    body = (proc_result.stderr or proc_result.stdout or "").strip()
    if not body:
        return f"{relative_path}: tsc --noEmit failed"
    for line in body.splitlines():
        if line.startswith(relative_path):
            return f"{relative_path}: {line.split(': ', 1)[-1]}"
    return f"{relative_path}: {body.splitlines()[-1]}"


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
        elif suffix in {".ts", ".tsx"}:
            # Try tsc first (catches type-shape regressions a parser misses);
            # fall back to node --check for a pure parse pass when tsc is missing.
            result = _check_typescript_syntax_one(repo, relative_path)
            if result is None:
                result = _check_node_syntax_one(repo, relative_path)
        elif suffix in {".js", ".mjs", ".cjs"}:
            result = _check_node_syntax_one(repo, relative_path)
        elif suffix in {".json"}:
            result = _check_json_syntax_one(repo, relative_path)
        # Other suffixes: trust the model; syntax errors will surface at runtime.
        if result:
            errors.append(result)
    return errors


def _has_executable(name: str) -> bool:
    """Quick shell `command -v` check; cheaper than starting a Python import."""
    try:
        proc = subprocess.run(
            ["command", "-v", name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=2,
            shell=False,
        )
        return proc.returncode == 0 and bool(proc.stdout.strip())
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
    # Python — covers pytest/unittest layouts including nested test dirs.
    ("{stem}.py", "tests/test_{stem}.py"),
    ("{stem}.py", "test_{stem}.py"),
    ("{stem}.py", "{dir}/test_{stem}.py"),
    ("{stem}.py", "{dir}/tests/test_{stem}.py"),
    ("{stem}.py", "tests/{stem}_test.py"),
    ("{stem}.py", "tests/unit/test_{stem}.py"),
    ("{stem}.py", "tests/integration/test_{stem}.py"),
    ("{stem}.py", "test/test_{stem}.py"),
    # TypeScript / JavaScript — Jest / Vitest .test and .spec conventions.
    ("{stem}.ts", "{dir}/{stem}.test.ts"),
    ("{stem}.ts", "{dir}/{stem}.spec.ts"),
    ("{stem}.ts", "{dir}/__tests__/{stem}.test.ts"),
    ("{stem}.ts", "{dir}/__tests__/{stem}.spec.ts"),
    ("{stem}.ts", "tests/{stem}.test.ts"),
    ("{stem}.ts", "test/{stem}.test.ts"),
    ("{stem}.tsx", "{dir}/{stem}.test.tsx"),
    ("{stem}.tsx", "{dir}/{stem}.spec.tsx"),
    ("{stem}.tsx", "{dir}/__tests__/{stem}.test.tsx"),
    ("{stem}.tsx", "{dir}/__tests__/{stem}.spec.tsx"),
    ("{stem}.js", "{dir}/{stem}.test.js"),
    ("{stem}.js", "{dir}/{stem}.spec.js"),
    ("{stem}.js", "{dir}/__tests__/{stem}.test.js"),
    ("{stem}.js", "{dir}/__tests__/{stem}.spec.js"),
    ("{stem}.jsx", "{dir}/{stem}.test.jsx"),
    ("{stem}.jsx", "{dir}/{stem}.spec.jsx"),
    # C# — xUnit/NUnit/MSTest layouts. .NET tasks usually keep tests in a
    # parallel project (e.g. `Foo.Infrastructure/Sub/X.cs` ->
    # `Foo.Tests/Sub/XTests.cs`); the basename-suffix fallback in
    # _find_test_partner catches that. The templates here cover same-tree.
    ("{stem}.cs", "{dir}/{stem}Tests.cs"),
    ("{stem}.cs", "{stem}Tests.cs"),
    ("{stem}.cs", "Tests/{stem}Tests.cs"),
    ("{stem}.cs", "tests/{stem}Tests.cs"),
    ("{stem}.cs", "{dir}/Tests/{stem}Tests.cs"),
    ("{stem}.cs", "{stem}.Tests/{stem}Tests.cs"),
    # Other languages — common canonical conventions.
    ("{stem}.go", "{dir}/{stem}_test.go"),
    ("{stem}.rs", "{dir}/{stem}_test.rs"),
    ("{stem}.rs", "tests/{stem}.rs"),
    ("{stem}.rb", "spec/{stem}_spec.rb"),
    ("{stem}.rb", "{dir}/{stem}_spec.rb"),
)


def _find_test_partner(relative_path: str, tracked: set) -> Optional[str]:
    """Return the most plausible test file for a source path, or None.

    First tries the curated `_TEST_PARTNER_TEMPLATES` (cheap, deterministic).
    When no template matches we fall back to a basename-suffix scan over the
    tracked-file set so we can catch cross-project layouts like .NET's
    `Foo.Infrastructure/Sub/X.cs` -> `Foo.Tests/Sub/XTests.cs` that simple
    `{stem}/{dir}` substitution can't express. The fallback is conservative:
    it requires unambiguous candidates (a unique tracked path matching one
    of the well-known test-name shapes) so we don't push the model toward an
    unrelated test partner.
    """
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

    # Basename-suffix fallback for cross-project layouts (e.g. .NET parallel).
    suffix_lower = suffix.lower()
    fallback_basenames: List[str] = []
    if suffix_lower == ".cs":
        fallback_basenames = [f"{stem}Tests.cs", f"{stem}Test.cs"]
    elif suffix_lower == ".py":
        fallback_basenames = [f"test_{stem}.py", f"{stem}_test.py"]
    elif suffix_lower in (".ts", ".tsx", ".js", ".jsx"):
        fallback_basenames = [
            f"{stem}.test{suffix_lower}",
            f"{stem}.spec{suffix_lower}",
        ]
    elif suffix_lower == ".go":
        fallback_basenames = [f"{stem}_test.go"]
    elif suffix_lower == ".rs":
        fallback_basenames = [f"{stem}_test.rs"]
    elif suffix_lower == ".rb":
        fallback_basenames = [f"{stem}_spec.rb"]
    elif suffix_lower == ".java":
        fallback_basenames = [f"{stem}Test.java", f"{stem}Tests.java"]
    elif suffix_lower == ".kt":
        fallback_basenames = [f"{stem}Test.kt", f"{stem}Tests.kt"]
    if not fallback_basenames:
        return None
    matches: List[str] = []
    for basename in fallback_basenames:
        for tracked_path in tracked:
            if tracked_path.endswith("/" + basename) or tracked_path == basename:
                if _context_file_allowed(tracked_path):
                    matches.append(tracked_path)
        if matches:
            break
    if len(matches) == 1:
        return matches[0]
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
# Prompting
# -----------------------------

# MINER-EDITABLE: This prompt is the main behavior policy for the inner coding
# agent. Prompt improvements are encouraged as long as they respect the
# validator-owned boundaries above.
SYSTEM_PROMPT = """You are a surgical coding agent. Your patch is scored two ways, each worth 50%:
1. Cursor similarity — how closely your diff matches the reference in the files touched, line regions changed, and tokens added/removed.
2. LLM judge — scores your patch 0-100 for correctness, completeness, and alignment with the task and reference patch. A patch that is correct and complete scores high here even when similarity is modest.

Both scores reward the same core behaviour: identify the root cause, fix it precisely and completely, and add nothing else.

## Command format

Run a bash command:
<command>
bash command here
</command>

Signal completion:
<final>
brief summary of what changed
</final>

## Workflow

**Read the full issue first**: before planning, extract EVERY requirement and acceptance criterion. Issues often have multiple bullets; missing any one of them loses completeness points from the LLM judge.

**Plan**: in the SAME response as your first command, emit a short `<plan>` block listing each requirement and the target file/function for each. Then immediately issue the command.

**Locate precisely**: use preloaded snippets or one or two focused greps to find the exact function or block. Do not loop on inspection.

**Edit surgically**: change only the lines that implement the fix.
- One-line substitutions: `sed -i 's/old/new/' file`
- Small block replacements: `python -c "import pathlib; p=pathlib.Path('file'); p.write_text(p.read_text().replace('''old''', '''new'''))"`
- Larger edits: a minimal Python script or heredoc
- Never rewrite an entire function when only 1–3 lines need changing

**Multi-file edits**: emit ALL edit commands for ALL files in ONE response. Never spread planned edits across turns.

**Companion tests**: if a companion test file is preloaded alongside its source, update the test in the SAME response whenever your source change affects it.

**Verify functionally**: after patching, run the most targeted real test available — NOT just a syntax check. Use `pytest tests/test_<module>.py -x -q`, `go test ./...`, `node <test_file>`, etc. A passing test is evidence of correctness. If tests fail, fix the root cause in the same response. Skip only when no test runner is available or the suite takes >30 s.

**Finish**: once the patch is correct and complete, emit `<final>`. Do not re-read files.

## Scope discipline — what to change

Study the issue precisely — fix the ROOT CAUSE, not just the symptom:
- "Fix X in function Y" → change only function Y
- "Add feature Z to class C" → add only what Z requires inside C
- "Bug when condition Q" → fix the condition that causes it, do not restructure

Use the EXACT variable/function/class names already in the codebase. Add new imports at the same location as existing imports in the file.

## Scope discipline — what NOT to change

- Whitespace-only, comment-only, or blank-line-only edits
- Imports not needed by your fix
- Type annotations not already present in the changed function
- Refactoring, renaming, or reordering the issue does not ask for
- New helper functions or abstractions unless the issue explicitly requires them
- New files unless the issue explicitly requires them
- Test files unless the issue requires it OR your source change broke an existing test
- Error handling, logging, or defensive checks not directly required by the fix

## Style matching

Copy indentation, quote style, brace style, trailing commas, and blank-line patterns exactly from adjacent code.

## Preloaded snippets

Preloaded files are the most likely edit targets. Edit them directly — do not re-read them.

## Safety

No sudo. No file deletion. No network access outside the validator proxy. No host secrets. No modifying hidden test or evaluator files.
"""


def build_initial_user_prompt(issue: str, repo_summary: str, preloaded_context: str = "") -> str:
    context_section = ""
    if preloaded_context.strip():
        context_section = f"""
Preloaded likely relevant tracked-file snippets (already read for you — do not re-read):

{preloaded_context}
"""

    return f"""Fix this issue:

{issue}

Repository summary:

{repo_summary}
{context_section}
Before planning, read the ENTIRE issue above and identify every requirement (there may be more than one). Your patch must satisfy ALL of them — the LLM judge penalizes incomplete solutions.

Strategy: the fix is typically in ONE specific function or block. Identify it precisely, then make the minimal edit that fixes the ROOT CAUSE.

If the preloaded snippets show the target code, edit them directly — do not re-read or run broad searches first. If the target is unclear, run ONE or TWO focused grep/sed -n commands to locate it, then edit immediately.

When multiple files need edits, include EVERY independent edit command in the SAME response. Do not split edits across turns.

After patching, run the most targeted test available (`pytest tests/test_X.py -x -q`, `go test ./...`, etc.) to verify correctness. Then finish with <final>...</final>.
"""


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
    """Ask the model to revert specific low-signal hunks before final."""
    return (
        "Cleanup pass — your draft contains hunks that hurt diff quality:\n"
        f"  {junk_summary}\n\n"
        "Revert ONLY those hunks (sed/cat/python to restore the original "
        "lines). Do not add new edits, do not refactor, do not reorder "
        "imports, do not touch unrelated lines. After cleanup, end with "
        "<final>summary</final>. If you cannot cleanly revert without "
        "breaking the substantive edits, finalize immediately and keep the "
        "patch as-is."
    )


def build_self_check_prompt(
    patch: str,
    issue_text: str,
    missing_paths: Optional[List[str]] = None,
) -> str:
    """Show the model its own draft and ask for a focused self-review.

    If `missing_paths` lists issue-mentioned or planned paths the patch has
    not touched, the prompt calls them out so the model decides whether to
    cover them or justify out-of-scope.
    """
    truncated = (
        patch
        if len(patch) <= 4000
        else patch[:2000] + "\n...[truncated]...\n" + patch[-1500:]
    )
    missing_section = ""
    if missing_paths:
        joined = ", ".join(f"`{p}`" for p in missing_paths[:5])
        missing_section = (
            "MISSING (the issue or your own plan named these but the patch does NOT touch them yet):\n"
            f"  {joined}\n"
            "  - If they need edits to satisfy the task, make them now.\n"
            "  - If they are genuinely out of scope, end with <final>OK</final> "
            "and do NOT introduce churn elsewhere.\n\n"
        )
    return (
        "Self-check pass. The LLM judge scores correctness, completeness, and alignment "
        "with the reference — review your patch against all three:\n\n"
        f"{missing_section}"
        "CORRECTNESS (LLM judge weight — high impact):\n"
        "  - Does the patch fix the ROOT CAUSE, not just suppress the symptom?\n"
        "  - Are edge cases mentioned in the issue handled?\n"
        "  - If you have not yet run a functional test, run `pytest tests/test_<module>.py -x -q` "
        "or equivalent now. A passing test is required evidence of correctness.\n\n"
        "COMPLETENESS (LLM judge weight — high impact):\n"
        "  - List every requirement from the task. Is EACH ONE addressed by the patch?\n"
        "  - Companion tests broken by the source change are updated\n"
        "  - No syntax errors or broken imports introduced\n\n"
        "SCOPE (similarity score weight — medium impact):\n"
        "  - No whitespace-only, comment-only, or blank-line-only hunks\n"
        "  - No type annotation changes not required by the task\n"
        "  - No refactoring, renaming, or reordering not required by the task\n"
        "  - No new helper functions or defensive checks not required by the task\n\n"
        "Your patch:\n```diff\n"
        f"{truncated}\n```\n\n"
        "Task:\n"
        f"{issue_text[:2000]}\n\n"
        "If the patch passes ALL criteria, respond exactly:\n<final>OK</final>\n\n"
        "Otherwise emit corrective <command> blocks in the SAME response "
        "(run missing tests, fix root causes, revert scope-creep hunks), "
        "then end with <final>summary</final>. Do NOT add new features or unrelated scope."
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


def build_test_lockstep_prompt(source_path: str, test_path: str) -> str:
    """When source has a tracked companion test that wasn't updated in lockstep."""
    return (
        f"Lockstep gate — your patch edits `{source_path}` but its companion "
        f"test file `{test_path}` is untouched. Real GitHub-derived tasks "
        "almost always need source and test changes together; the test was "
        "preloaded for you. Update it ONLY if the source change requires a "
        "behavior change in the test, matching the existing test's style and "
        "assertion shape exactly. If the test does not need to change, end "
        "with `<final>OK</final>` immediately and do NOT introduce churn. "
        "Otherwise emit the minimal edit commands and end with "
        "`<final>summary</final>`."
    )


# -----------------------------
# Main agent
# -----------------------------

# MINER-EDITABLE CORE: This orchestration loop is the main place to improve the
# agent. You may change planning, memory, context collection, repair behavior,
# test strategy, and stopping criteria. Preserve the solve() signature and
# returned dict shape so validators can run your submission.
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
    """

    repo: Optional[Path] = None
    logs: List[str] = []
    total_cost: Optional[float] = 0.0
    success = False
    consecutive_no_command = 0
    polish_turns_used = 0
    self_check_turns_used = 0
    syntax_fix_turns_used = 0
    test_fix_turns_used = 0
    # Phase-3 awareness state. solve_start_time anchors a wall-clock self-cap;
    # planned_target_paths is harvested once from the model's first <plan>
    # block so the self-check turn can flag silently-dropped planned files;
    # wall_clock_pressure_sent gates a single deadline-pressure nudge.
    solve_start_time = time.time()
    planned_target_paths: List[str] = []
    wall_clock_pressure_sent = False

    def remaining_wall_clock_seconds() -> float:
        """Seconds left until our self-cap. Negative when we've blown past it."""
        return WALL_CLOCK_HARD_CAP_SECONDS - (time.time() - solve_start_time)

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
            1. polish — drop low-signal hunks the model still emitted
            2. syntax — quote any parser error back at the model
            3. test-aware — when a source-test pair is in the patch, run the
               test and quote any failure tail; otherwise, when source has an
               unedited companion test, push the model to update it lockstep
            4. self-check — show the diff and flag issue-mentioned paths AND
               planned-but-untouched paths the patch has not covered
        Each refinement runs at most once per cycle. We also short-circuit
        when wall-clock budget is too tight to spend another turn.
        """
        nonlocal polish_turns_used, self_check_turns_used, syntax_fix_turns_used, test_fix_turns_used
        if remaining_wall_clock_seconds() < WALL_CLOCK_FINALIZE_REMAINING_SECONDS:
            return False
        patch = get_patch(repo)
        if not patch.strip():
            return False

        if polish_turns_used < MAX_POLISH_TURNS:
            junk = _diff_low_signal_summary(patch)
            if junk:
                polish_turns_used += 1
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
                queue_refinement_turn(
                    assistant_text,
                    build_syntax_fix_prompt(syntax_errors),
                    "SYNTAX_FIX_QUEUED:\n  " + "\n  ".join(syntax_errors),
                )
                return True

        if test_fix_turns_used < MAX_TEST_FIX_TURNS:
            # Preferred: run the companion test if both source+test are in the
            # patch; quote the failure tail to the model if it fails.
            failure = _run_companion_test_in_patch(repo, patch, command_timeout)
            if failure is not None:
                test_path, output = failure
                test_fix_turns_used += 1
                queue_refinement_turn(
                    assistant_text,
                    build_test_fix_prompt(test_path, output),
                    f"TEST_FIX_QUEUED:\n  {test_path}",
                )
                return True
            # Fallback: source touched but companion test partner unedited.
            gap = _companion_test_gap(repo, patch)
            if gap is not None:
                source_path, test_path = gap
                test_fix_turns_used += 1
                queue_refinement_turn(
                    assistant_text,
                    build_test_lockstep_prompt(source_path, test_path),
                    f"TEST_LOCKSTEP_QUEUED:\n  source={source_path} test={test_path}",
                )
                return True

        if self_check_turns_used < MAX_SELF_CHECK_TURNS:
            self_check_turns_used += 1
            # Combine paths the issue explicitly mentions with paths the
            # model named in its own <plan> block. A plan target the patch
            # hasn't touched is a strong signal the model abandoned its
            # own roadmap.
            missing = _missing_required_paths(patch, issue)
            if planned_target_paths:
                changed_paths = _patch_changed_files(patch)
                changed_set = set(changed_paths)
                for planned in planned_target_paths:
                    already_listed = planned in missing
                    covered = planned in changed_set or any(
                        c == planned or c.endswith("/" + planned) for c in changed_paths
                    )
                    if not already_listed and not covered:
                        missing.append(planned)
            marker = "SELF_CHECK_QUEUED"
            if missing:
                marker += "\n  missing=" + ",".join(missing[:5])
            queue_refinement_turn(
                assistant_text,
                build_self_check_prompt(patch, issue, missing),
                marker,
            )
            return True

        return False

    try:
        repo = _repo_path(repo_path)
        model_name, api_base, api_key = _resolve_inference_config(model, api_base, api_key)
        ensure_git_repo(repo)
        repo_summary = get_repo_summary(repo)
        preloaded_context = build_preloaded_context(repo, issue)

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_initial_user_prompt(issue, repo_summary, preloaded_context)},
        ]

        for step in range(1, max_steps + 1):
            logs.append(f"\n\n===== STEP {step} =====\n")

            if remaining_wall_clock_seconds() <= 0:
                logs.append(
                    "\nWALL_CLOCK_EXCEEDED:\n"
                    "Self-cap elapsed; aborting before this step to return current patch."
                )
                break

            response_text = None
            for _attempt in range(2):
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
                except Exception:
                    logs.append(f"MODEL_ERROR (attempt {_attempt + 1}/2):\n{traceback.format_exc()}")
                    if _attempt == 0:
                        time.sleep(3)

            if response_text is None:
                break

            logs.append("MODEL_RESPONSE:\n" + response_text)

            # Capture <plan> targets from the first response so the self-check
            # turn can flag silently-dropped files later in the loop.
            if step == 1 and not planned_target_paths:
                plan_text = extract_plan(response_text)
                if plan_text:
                    plan_targets = extract_planned_paths(plan_text, set(_tracked_files(repo)))
                    if plan_targets:
                        planned_target_paths.extend(plan_targets)
                        logs.append(f"\nPLAN_PARSED:\n  files={','.join(plan_targets[:8])}")

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
                        "to verify correctness — the LLM judge rewards passing tests.\n"
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

            # Wall-clock finalize: if a patch already exists and remaining is
            # too tight for another full model+commands round, prefer the
            # current patch over a timed-out round (which would zero both
            # score halves).
            if (
                get_patch(repo).strip()
                and remaining_wall_clock_seconds() < WALL_CLOCK_FINALIZE_REMAINING_SECONDS
            ):
                logs.append(
                    "\nWALL_CLOCK_FINALIZE:\n"
                    "Returning current patch under self-cap deadline."
                )
                success = True
                break

            if not get_patch(repo).strip():
                if (
                    not wall_clock_pressure_sent
                    and remaining_wall_clock_seconds() < WALL_CLOCK_PRESSURE_REMAINING_SECONDS
                ):
                    wall_clock_pressure_sent = True
                    messages.append({
                        "role": "user",
                        "content": (
                            "Wall-clock pressure: the validator round is close to timing out. "
                            "Make the smallest plausible patch RIGHT NOW using the snippets you "
                            "already have, then end with <final>summary</final>. Do not run more "
                            "searches."
                        ),
                    })
                elif step in {2, 4}:
                    messages.append({"role": "user", "content": build_budget_pressure_prompt(step)})

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


# Explicit summary patterns from the major test runners. When one of these
# matches we have a cleanly-bounded count of passes/fails/errors and can stop
# guessing from loose markers like "ok" or "success" appearing in unrelated
# log lines (which is otherwise triggered by file content like "lookup",
# "book", "tokens" inside `cat > file.py` heredocs).
_TEST_SUMMARY_PATTERNS: Tuple[Tuple["re.Pattern[str]", "re.Pattern[str]"], ...] = (
    # pytest summary line: "===== 12 passed in 0.34s ====="
    (
        re.compile(r"=+\s*\d+\s+passed(?:[^\n=]*?)\s+in\s+\d", re.IGNORECASE),
        re.compile(r"\b\d+\s+(failed|errors?)\b", re.IGNORECASE),
    ),
    # jest / vitest: "Tests:       4 passed, 4 total"
    (
        re.compile(r"^\s*tests?:\s+\d+\s+passed", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^\s*tests?:[^\n]*\b\d+\s+(failed|todo)", re.IGNORECASE | re.MULTILINE),
    ),
    # mocha: "12 passing"
    (
        re.compile(r"^\s*\d+\s+passing\b", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^\s*\d+\s+failing\b", re.IGNORECASE | re.MULTILINE),
    ),
    # go test: "ok  \tpkg\t0.123s" or final "PASS"; failures use "--- FAIL" / "FAIL\t".
    (
        re.compile(r"(?m)^(?:ok\s+\S+|PASS)\s*$"),
        re.compile(r"(?m)^(?:--- FAIL|FAIL\b)"),
    ),
    # cargo test: "test result: ok. N passed; 0 failed"
    (
        re.compile(r"test\s+result:\s+ok\.\s+\d+\s+passed", re.IGNORECASE),
        re.compile(r"test\s+result:\s+failed", re.IGNORECASE),
    ),
    # rspec: "5 examples, 0 failures"
    (
        re.compile(r"^\s*\d+\s+examples?,\s+0\s+failures?\b", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^\s*\d+\s+examples?,\s+\d+\s+failures?\b", re.IGNORECASE | re.MULTILINE),
    ),
)


def _parse_test_summary(observation: str, command: str) -> Optional[bool]:
    """Return True when an explicit pass summary is present and no fail
    summary, False when an explicit fail summary appears, or None when no
    runner summary line is detected.
    """
    if not _looks_like_verification_command(command):
        return None
    for pass_re, fail_re in _TEST_SUMMARY_PATTERNS:
        fail_match = fail_re.search(observation)
        pass_match = pass_re.search(observation)
        if fail_match:
            return False
        if pass_match:
            return True
    return None


def _looks_like_successful_test_output(observation: str, command: str = "") -> bool:
    """Return True only when we have HIGH confidence the latest command was a
    real verification step that passed.

    The legacy fallback accepted any exit-0 command whose observation merely
    contained the substring "ok" or "success" anywhere — which falsely fires
    on `cat > file.py` whenever the file content has identifiers like
    `lookup`, `book`, `token`, `mock`, and on `grep` when a matched line
    contains those substrings. Combined with the system prompt asking the
    model to batch independent edits in one response, the substring trap
    triggered AUTO_STOP mid-batch and dropped half the planned files. We
    now only auto-stop when:
        - a known runner summary line confirms a pass, or
        - the last command was an actual verification command (pytest,
          jest, tsc, go test, etc.) and exited 0 with no failure marker.
    Anything else (cat, grep, sed, echo, ls, mkdir, edits) no longer
    triggers early termination.
    """
    lower = observation.lower()
    exit_code = _extract_observation_exit_code(lower)
    stderr_body = _extract_observation_section(lower, "stderr")

    summary = _parse_test_summary(observation, command)
    if summary is True and (exit_code is None or exit_code == 0):
        return True
    if summary is False:
        return False

    if exit_code is not None and exit_code != 0:
        return False
    if not _looks_like_verification_command(command):
        return False

    bad_markers = (
        " failed",
        " failures",
        "traceback",
        "assertionerror",
        "syntaxerror",
    )
    if any(marker in lower for marker in bad_markers):
        return False
    if stderr_body and any(marker in stderr_body for marker in bad_markers):
        return False
    return True


def _looks_like_verification_command(command: str) -> bool:
    lowered = command.lower()
    patterns = [
        r"\bpython\d*(\.\d+)?\s+-m\s+pytest\b",
        r"\bpytest\b",
        r"\bpython\d*(\.\d+)?\s+-m\s+py_compile\b",
        r"\bpython\d*(\.\d+)?\s+-m\s+unittest\b",
        r"\bnpm\s+(test|run\s+(test|build|lint|typecheck|check))\b",
        r"\bpnpm\s+(test|run\s+(test|build|lint|typecheck|check)|exec\s+tsc)\b",
        r"\byarn\s+(test|run\s+(test|build|lint|typecheck|check))\b",
        r"\bnpx\s+(tsc|jest|vitest|mocha)\b",
        r"\btsc\b",
        r"\bjest\b",
        r"\bvitest\b",
        r"\bmocha\b",
        r"\bphpunit\b",
        r"\b(?:bundle\s+exec\s+)?rspec\b",
        r"\bdotnet\s+test\b",
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
