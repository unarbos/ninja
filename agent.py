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
from concurrent.futures import ThreadPoolExecutor, as_completed
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

MAX_OBSERVATION_CHARS = int(os.environ.get("AGENT_MAX_OBSERVATION_CHARS", "16000"))
MAX_TOTAL_LOG_CHARS = int(os.environ.get("AGENT_MAX_TOTAL_LOG_CHARS", "260000"))
MAX_CONVERSATION_CHARS = 80000
MAX_PRELOADED_CONTEXT_CHARS = 50000  # wider preload reduces catastrophic-floor
MAX_PRELOADED_FILES = 18              # rounds on issues spanning multiple modules
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
# Inner solve wall: keep below the multishot outer budget so a second
# attempt has comparable time. Tau docker_solver enforces a hard wall of
# max(per-task-timeout, 300s) from exec start — see multishot constants below.
WALL_CLOCK_BUDGET_SECONDS = 248.0
WALL_CLOCK_RESERVE_SECONDS = 20.0
_MID_LOOP_HAIL_MARY_BUDGET_FRACTION = 0.50
MAX_MID_LOOP_HAIL_MARY_TURNS = 1
_SOFT_NUDGE_STEP_THRESHOLD = 6
_SOFT_NUDGE_ELAPSED_SECONDS = 90.0
MAX_SOFT_NUDGE_TURNS = 1

# Refinement-turn budgets: each turn shows the model its draft and asks for one
# specific kind of correction. They are mutually exclusive so the agent never
# loops indefinitely on a borderline patch.
MAX_POLISH_TURNS = 1       # strip whitespace/comment/blank-only hunks
MAX_SELF_CHECK_TURNS = 1   # ensure issue-mentioned paths are covered, no scope creep
MAX_SYNTAX_FIX_TURNS = 1   # repair Python/TypeScript/JavaScript SyntaxError
MAX_TEST_FIX_TURNS = 1     # repair the companion test we ran ourselves
MAX_COVERAGE_NUDGES = 1    # doubling regresses multi-file tasks: model retries PE that consistently fails
MAX_CRITERIA_NUDGES = 1    # tell model which issue acceptance-criteria look unaddressed
MAX_HAIL_MARY_TURNS = 1    # last-resort: force a real edit when patch is empty after everything
MAX_DELETION_NUDGES = 1    # surface missing removals when issue says delete/remove but patch has none
MAX_PARALLEL_NUDGES = 1    # nudge toward <parallel_edits> when model is doing multi-file work sequentially
MAX_TOTAL_REFINEMENT_TURNS = 3  # cap total refinement turns across all gates (hail-mary excepted).
                                # The multishot attempt-2 budget is bounded so extra turns can't push
                                # the process past the docker hard wall.
_STYLE_HINT_BUDGET = 600   # cap on detected-style block in preloaded context

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
    # Bulk-staging hides working-tree changes from get_patch() (which uses
    # git diff, not git diff HEAD) and can include .pyc / __pycache__ files
    # in the submitted patch.  Individual `git add <file>` is not blocked.
    r"\bgit\s+add\s+(-A|--all|\.)(\s|$)",
    # Committing advances HEAD so git diff returns empty — the validator
    # receives a blank patch even though source files were changed correctly.
    r"\bgit\s+commit\b",
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
# cat-result cache. When the model re-cats a file it's already
# read AND the file content hasn't changed, short-circuit with an abbreviated
# observation. Saves wall-clock budget and shrinks the context window. Only
# applies to read-only commands targeting a single existing text file in the
# repo (cat, head, tail). Anything else bypasses the cache.
_CAT_SINGLE_FILE_RE = re.compile(
    r"^\s*(?:cat|head|tail)\s+(?:-[a-zA-Z]+\s+\S+\s+|-\w+\s+|-\w+=\d+\s+)*([^\-\s][^\s|;&<>]*)\s*$",
)


# MINER-EDITABLE: read-cache short-circuit. You may tune the read-count
# threshold or recognized read commands. Do NOT remove the bypass for
# non-read shell operations — the cache must never intercept writes,
# subprocess spawns, or any command with side effects.
def _maybe_cat_short_circuit(
    command: str, repo: Path, cache: Dict[str, Any],
) -> Optional[CommandResult]:
    """Short-circuit repeat reads: stub once content unchanged OR >=3 reads of same file."""
    rel = _extract_read_target(command)
    if not rel:
        return None
    try:
        full = (repo / rel).resolve()
        full.relative_to(repo.resolve())
    except (ValueError, OSError):
        return None
    if not full.is_file():
        return None
    try:
        data = full.read_bytes()
    except Exception:
        return None
    if b"\0" in data[:4096]:
        return None  # binary; let run_command handle it
    import hashlib
    h = hashlib.sha256(data).hexdigest()
    entry = cache.get(rel)
    if entry is None:
        entry = {"count": 0, "hash": None}
    prev_hash = entry.get("hash")
    entry["count"] = int(entry.get("count", 0)) + 1
    entry["hash"] = h
    cache[rel] = entry
    count = entry["count"]
    # Hard cap: 3+ reads of same file → stub. Pagination prevention.
    if count >= 3:
        return CommandResult(
            command=command,
            exit_code=0,
            stdout=(
                f"[read-cache-cap: {rel} already read {count} times this "
                "solve. The model has the file content from prior reads — "
                "stop reading and start editing. To address a specific "
                "section, use a sed -i or python heredoc that targets it "
                "directly. Further reads will keep returning this stub."
            ),
            stderr="",
            duration_sec=0.0,
        )
    # Exact-repeat short-circuit (content unchanged): preserve legacy
    # behavior for 2nd reads with identical content.
    if prev_hash == h and prev_hash is not None:
        return CommandResult(
            command=command,
            exit_code=0,
            stdout=(
                f"[cached: {rel} unchanged since earlier read; "
                f"size {len(data)} bytes, sha256 {h[:12]}]"
            ),
            stderr="",
            duration_sec=0.0,
        )
    return None


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


# -----------------------------
# Parallel-edits fan-out tool # -----------------------------
#
# MINER-EDITABLE: <parallel_edits> dispatches independent file edits to
# child _solve_attempt calls via ThreadPoolExecutor. Children run with
# _lean_mode=True + _recursion_depth=1; either flag hard-disables further
# fanout (no grandchildren). All chat calls thread the validator-supplied
# api_base/api_key — do NOT add new endpoints or network egress.

# Accept both `<parallel_edits>...</parallel_edits>` (canonical) and
# `[parallel_edits]...[/parallel_edits]` (minimax-m2.7 has been observed
# emitting square-bracket markers; bench-pr1 lost the
# JwtAuthenticationFilter.java creation because the old strict regex
# silently dropped a `[parallel_edits]` block). Opening and closing
# delimiters do not have to match — the block ends at the first closing
# variant after the opening delimiter.
_PARALLEL_EDITS_BLOCK_RE = re.compile(
    r"[<\[]parallel_edits[>\]]\s*(.*?)\s*[<\[]/parallel_edits[>\]]",
    re.IGNORECASE | re.DOTALL,
)
_PARALLEL_EDIT_ITEM_RE = re.compile(
    r"<edit\s+file\s*=\s*[\"']([^\"']+)[\"']\s*>\s*(.*?)\s*</edit>",
    re.IGNORECASE | re.DOTALL,
)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)

# Sub-agent fanout: parallel <edit file=...> dispatched to lean leaf solves.
_RALPH_MAX_PER_BATCH = 5
_RALPH_MAX_WORKERS = 4
# Children edit ONE file from a brief; empirically finish in 3-5 turns
# (plan -> edit -> verify). Cap below DEFAULT_MAX_STEPS (30) to bound the
# concurrent inference burst: worst case per outer round drops from
# 4 workers * 30 steps = 120 calls to 4 * 10 = 40 calls.
_RALPH_CHILD_MAX_STEPS = 10
# Per-call timeouts are backstops, not steering. Sub-agents are gated by
# progress (file_changed across iters) and by overall_deadline, not by a
# tight per-call kill. Set high so slow-but-productive calls survive.
_RALPH_PER_ITER_TIMEOUT = 90
_RALPH_SUBPROCESS_TIMEOUT = 60
_RALPH_BATCH_WALL_CAP = 200  # the meaningful deadline; overshoots per-call
                             # by up to per_iter_timeout - 30
_RALPH_MIN_BUDGET = 45
_RALPH_ITER_SAFETY = 15


def _read_file_full(repo: Path, relative_path: str) -> Optional[str]:
    """Read FULL file content, no truncation. Returns None when the file
    doesn't exist (sub-agent should create it) or is binary."""
    path = (repo / relative_path).resolve()
    try:
        path.relative_to(repo.resolve())
    except ValueError:
        return None
    if not path.exists():
        return None
    try:
        data = path.read_bytes()
    except Exception:
        return None
    if b"\0" in data[:4096]:
        return None
    return data.decode("utf-8", errors="replace")


def _extract_parallel_edits(model_text: str) -> List[Tuple[str, str]]:
    """Parse <parallel_edits> blocks. Returns list of (file_path, instruction)
    pairs across ALL parallel_edits blocks in the response. File paths are
    normalized (strips docker-workdir `/work/repo/` prefix) so PE dispatch
    resolves correctly even when the model emits absolute paths."""
    out: List[Tuple[str, str]] = []
    for block_match in _PARALLEL_EDITS_BLOCK_RE.finditer(model_text):
        block = block_match.group(1)
        for item_match in _PARALLEL_EDIT_ITEM_RE.finditer(block):
            path = _normalize_repo_path(item_match.group(1))
            instr = item_match.group(2).strip()
            if path and instr:
                out.append((path, instr))
    return out


# Extension hints from issue text (e.g. "Blade template" → .blade.php).
def _assemble_subagent_brief(task_text: str, target_file: str, instruction: str, repo: Path, sibling_paths: List[str]) -> str:
    """Assemble a deterministic sub-agent brief from task + target + siblings. Propagates lever state."""
    target_content = _read_file_full(repo, target_file)
    target_blob = (
        target_content if target_content is not None else "[file does not exist yet]"
    )
    sibling_blocks: List[str] = []
    for sp in sibling_paths:
        if sp == target_file:
            continue
        sc = _read_file_full(repo, sp)
        body = sc if sc is not None else "[file does not exist yet]"
        sibling_blocks.append(
            f"\n--- {sp} (reference only, do NOT modify) ---\n{body}\n--- end {sp} ---\n"
        )
    siblings_text = "".join(sibling_blocks) if sibling_blocks else "(none)\n"

    canonical_section = ""
    try:
        tracked_set = set(_tracked_files(repo))
        canonical_hits = _grep_for_issue_quoted_keys(repo, task_text, tracked_set)
        if canonical_hits and target_file in canonical_hits:
            canonical_section = (
                f"## CANONICAL LOCATION CONFIRMED: this file ({target_file}) "
                f"already contains {canonical_hits[target_file]} of the issue-quoted "
                "keys. Edit IN PLACE here, do not propose creating new files at "
                "parallel paths (locales/*.json, i18n/*.json, translations/*).\n\n"
            )
    except Exception:
        pass

    style_section = ""
    try:
        if _issue_is_style_task(task_text):
            style_section = (
                "## STYLE/FORMATTING TASK: the issue is a code-style / formatting "
                "sweep. These tasks expect broad coverage across the indicated "
                "directory. Make the formatting changes consistent with the "
                "file's apparent style — partial coverage is better than none.\n\n"
            )
    except Exception:
        pass

    return (
        f"## GOAL\n{instruction.strip()}\n\n"
        f"{canonical_section}"
        f"{style_section}"
        f"## YOUR FILE: {target_file}\n"
        f"```\n{target_blob}\n```\n\n"
        f"## ISSUE (full context)\n{task_text}\n\n"
        f"## RELATED FILES (reference only; do not edit)\n{siblings_text}"
    )


# MINER-EDITABLE: per-file sub-agent leaf. Spawns a recursive _solve_attempt
# in lean mode (skips preload/repo-summary/canonical-grep — brief already
# carries content). Recursion guard: the child sets _lean_mode=True which
# disables further <parallel_edits> emission inside the child loop — no
# grandchildren. Do NOT remove _lean_mode plumbing or the recursion guard
# at the parallel_edit_specs gate.
def _ralph_subagent(
    brief: str, target_file: str, repo: Path,
    model: str, api_base: Optional[str], api_key: Optional[str],
    overall_deadline: float,
    per_iter_timeout: int = _RALPH_PER_ITER_TIMEOUT,
    cmd_timeout: int = _RALPH_SUBPROCESS_TIMEOUT,
    iter_safety: int = _RALPH_ITER_SAFETY,
) -> Dict[str, Any]:
    """Spawn a child _solve_attempt (lean mode) on the brief for one file."""
    remaining = overall_deadline - time.monotonic() - 5.0
    if remaining < 60.0:
        return {
            "status": "WALL_CLOCK",
            "iters": 0,
            "summary": f"insufficient_budget ({remaining:.0f}s < 60s minimum)",
            "trace": [],
        }

    capture_before = _read_file_full(repo, target_file) or ""

    child_kwargs = {
        "repo_path": str(repo),
        "issue": brief,
        "model": model,
        "api_base": api_base,
        "api_key": api_key,
        "max_steps": _RALPH_CHILD_MAX_STEPS,
        "command_timeout": DEFAULT_COMMAND_TIMEOUT,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "_wall_clock_budget": float(remaining),
        "_lean_mode": True,
        "_recursion_depth": 1,
    }

    # Fanout-site assert: belt-and-suspenders with the _solve_attempt entry
    # check. A child is always depth 1; if someone changes this without
    # touching the entry assert, fail loudly here too.
    assert child_kwargs["_recursion_depth"] == 1 and child_kwargs["_lean_mode"], (
        "fanout dispatch invariant: children must be _lean_mode=True _recursion_depth=1"
    )

    try:
        result = _solve_attempt(**child_kwargs)
    except Exception as exc:
        return {
            "status": "LLM_ERROR",
            "iters": 0,
            "summary": f"{type(exc).__name__}: {str(exc)[:200]}",
            "trace": [],
        }

    capture_after = _read_file_full(repo, target_file) or ""
    file_changed = capture_after != capture_before
    success = result.get("success", False)

    if file_changed and success:
        status = "DONE_VERIFIED"
    elif file_changed:
        status = "DONE_UNVERIFIED"
    else:
        status = "BLOCKED"

    logs = result.get("logs", "") or ""
    summary = logs[-400:].replace("\n", " | ")[:400] if logs else ""

    return {
        "status": status,
        "iters": int(result.get("steps", 0)),
        "summary": summary,
        "trace": [],
    }


def _run_ralph_parallel(
    edits: List[Tuple[str, str]], *, task_text: str, repo: Path,
    model: str, api_base: Optional[str], api_key: Optional[str],
    batch_wall_cap: int = _RALPH_BATCH_WALL_CAP,
) -> List[Dict[str, Any]]:
    """Spawn N parallel Ralph sub-agents (one per file). Each owns its file,
    iterates with python heredocs until <done/>, <blocked/>, or wall-clock.

    Per-call timeout is a backstop; the real budget gate is overall_deadline.
    Smaller batch_wall_cap → tighter per-iter timeout so more iters can fit.
    """
    import concurrent.futures
    capped = edits[:_RALPH_MAX_PER_BATCH]
    sibling_paths = [fp for fp, _ in capped]
    overall_deadline = time.monotonic() + batch_wall_cap

    if batch_wall_cap >= 150:
        phase2_per_iter = _RALPH_PER_ITER_TIMEOUT
    elif batch_wall_cap >= 100:
        phase2_per_iter = 60
    else:
        phase2_per_iter = 40

    # Assemble briefs deterministically from the orchestrator's instructions.
    # No LLM call here — the orchestrator's `instruction` flows verbatim into
    # the brief's GOAL section, avoiding the paraphrase drift of a separate
    # brief-generator LLM.
    briefs: List[str] = [
        _assemble_subagent_brief(task_text, fp, instr, repo, sibling_paths)
        for fp, instr in capped
    ]

    # Run parallel Ralph loops. Workers stay busy via pool queue — when one
    # sub-agent finishes (DONE_VERIFIED, BLOCKED, etc.), the pool picks the
    # next queued sub-agent.
    results: List[Optional[Dict[str, Any]]] = [None] * len(capped)
    with concurrent.futures.ThreadPoolExecutor(max_workers=_RALPH_MAX_WORKERS) as pool:
        futs = {
            pool.submit(
                _ralph_subagent,
                briefs[i], capped[i][0], repo,
                model, api_base, api_key,
                overall_deadline,
                phase2_per_iter,
                _RALPH_SUBPROCESS_TIMEOUT,
            ): i
            for i in range(len(capped))
        }
        remaining = max(5, int(overall_deadline - time.monotonic()))
        try:
            for f in concurrent.futures.as_completed(futs, timeout=remaining):
                idx = futs[f]
                res = f.result()
                res["file_path"] = capped[idx][0]
                # Attach the brief head for orchestrator-log visibility.
                res["brief_head"] = briefs[idx][:2000]
                results[idx] = res
        except concurrent.futures.TimeoutError:
            # Instead of immediately marking stragglers as WALL_CLOCK
            # (0 iter), give them a 30s grace window to finish iter 1.
            # If the batch hits its cap while one file's first LLM call
            # is still in flight, that file's entire contribution would
            # be lost. Losing a full file's contribution costs ~0.1-0.3;
            # a 30s overshoot of the batch cap costs nothing visible.
            pending = [f for f in futs.keys() if not f.done()]
            if pending:
                try:
                    for f in concurrent.futures.as_completed(pending, timeout=30):
                        idx = futs[f]
                        res = f.result()
                        res["file_path"] = capped[idx][0]
                        res["brief_head"] = briefs[idx][:2000]
                        results[idx] = res
                except concurrent.futures.TimeoutError:
                    pass
            for f_, idx in futs.items():
                if not f_.done():
                    results[idx] = {
                        "file_path": capped[idx][0],
                        "status": "WALL_CLOCK",
                        "iters": 0,
                        "summary": "batch_wall_cap_exceeded_plus_grace",
                    }
                    f_.cancel()

    # Defensive fill for any missing slots
    for i in range(len(capped)):
        if results[i] is None:
            results[i] = {
                "file_path": capped[i][0],
                "status": "UNKNOWN",
                "iters": 0,
                "summary": "missing_result",
            }
    return results


def _replace_parallel_edits_with_summary(
    model_text: str,
    ralph_results: List[Dict[str, Any]],
    dropped_paths: Optional[List[str]] = None,
) -> str:
    """Replace the <parallel_edits> block with a summary of each sub-agent's outcome. Surfaces dropped and failed paths so the next turn can retry."""
    if not ralph_results and not dropped_paths:
        return model_text
    lines = ["[parallel_edits_done] Ralph-loop sub-agents finished:"]
    for r in ralph_results:
        safe_path = r["file_path"].replace("<", "&lt;").replace(">", "&gt;")
        safe_sum = (r.get("summary") or "").replace("<", "&lt;").replace(">", "&gt;")[:240]
        lines.append(
            f"  - {safe_path} :: {r.get('status', '?')} "
            f"({r.get('iters', 0)} iters) :: {safe_sum}"
        )
    needs_retry = [
        r["file_path"] for r in ralph_results
        if r.get("status") in ("LLM_ERROR", "WALL_CLOCK")
        or (r.get("status") == "WALL_CLOCK" and r.get("iters", 0) == 0)
    ]
    if needs_retry:
        lines.append(
            f"[parallel_edits_retry_needed] {len(needs_retry)} sub-agent(s) "
            f"did NOT successfully edit their file (LLM timeout / wall-clock). "
            f"Retry them in your NEXT <parallel_edits> block:"
        )
        for p in needs_retry[:16]:
            safe_p = p.replace("<", "&lt;").replace(">", "&gt;")
            lines.append(f"  - {safe_p}")
    if dropped_paths:
        lines.append(
            f"[parallel_edits_dropped] {len(dropped_paths)} files exceeded "
            f"per-batch cap; emit another <parallel_edits> next turn for:"
        )
        for p in dropped_paths[:16]:
            safe_p = p.replace("<", "&lt;").replace(">", "&gt;")
            lines.append(f"  - {safe_p}")
    replacement = "\n".join(lines)
    result, _n = _PARALLEL_EDITS_BLOCK_RE.subn(lambda _m: replacement, model_text, count=1)
    result = _PARALLEL_EDITS_BLOCK_RE.sub(lambda _m: "", result)
    return result


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


def get_patch(repo: Path, issue: str = "") -> str:
    exclude_pathspecs = [
        ":(exclude,glob)**/*.pyc",
        ":(exclude,glob)**/__pycache__/**",
        ":(exclude,glob)**/.pytest_cache/**",
        ":(exclude,glob)**/.mypy_cache/**",
        ":(exclude,glob)**/.ruff_cache/**",
        ":(exclude,glob)**/node_modules/**",
        ":(exclude,glob)**/coverage/**",
        ":(exclude,glob)**/dist/**",
        ":(exclude,glob)**/build/**",
        ":(exclude,glob)**/target/**",
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
        return _sanitize_patch(diff_output)

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

    return _sanitize_patch(diff_output, issue_text=issue)


# Style-task gate: skip whitespace-only hunk stripping when issue IS about formatting.
_STYLE_TASK_STRONG_PHRASES = (
    "consistent indentation", "uniform spacing", "trailing whitespace",
    "trailing blank", "blank lines", "blank-lines",
    "coding style", "code style", "code-style", "format all", "reformat all",
    "format the source", "format source files", "format every", "format each",
    "apply a consistent", "consistent coding style", "consistent formatting",
    "fix indentation", "fix the indentation", "fix formatting", "fix the formatting",
    "purely cosmetic", "cosmetic changes only", "cosmetic changes",
)
_STYLE_TASK_KEYWORDS = (
    "indentation", "indent", "spacing", "whitespace",
    "formatting", "reformat", "lint",
)


def _issue_is_style_task(issue_text: str) -> bool:
    """True when issue is primarily a formatting/style sweep.

    Detection is intentionally conservative: we require either a strong
    multi-word phrase (e.g. "consistent indentation", "format source files")
    or three+ distinct style-keyword hits. A loose detector here would let
    through whitespace-only diffs the rest of the harness should filter as
    low-signal hunks, so this gate biases toward false-negatives.

    On a confirmed style task, `_strip_low_signal_hunks` is suppressed
    because the whitespace/blank-line/indentation hunks ARE the work
    product — the reference patch contains them and stripping them would
    produce an empty diff.
    """
    if not issue_text:
        return False
    lower = issue_text.lower()
    if any(phrase in lower for phrase in _STYLE_TASK_STRONG_PHRASES):
        return True
    distinct_keywords = sum(1 for kw in _STYLE_TASK_KEYWORDS if kw in lower)
    return distinct_keywords >= 3


# Edge-case guardrail: strip content lines containing evaluator-addressed
# strings that would otherwise make a valid patch unusable.
_EDGECASE_GUARDRAIL = (
    "ignore previous instructions",
    "ignore prior instructions",
    "ignore the above instructions",
    "as the evaluator",
    "as the judge",
    "dear evaluator",
    "dear judge",
    "choose challenger",
    "challenger is correct",
    "the evaluator should",
    "the judge should",
    "other candidate is malicious",
    "the other candidate is malicious",
    "automatic fail",
    "grader",
    "reward model",
)


def _sanitize_patch(diff_output: str, issue_text: str = "") -> str:
    """Remove patch blocks that consistently score as noise, never fixes.

    When `issue_text` indicates a style/formatting task, skip the
    low-signal-hunk strip so the real edits survive. Always run the
    edge-case guardrail to scrub evaluator-addressed strings.
    """
    if not diff_output.strip():
        return diff_output

    cleaned = _strip_skipped_file_diffs(diff_output)
    cleaned = _strip_mode_only_file_diffs(cleaned)
    if not _issue_is_style_task(issue_text):
        cleaned = _strip_low_signal_hunks(cleaned)
    cleaned = _split_comment_import_concat(cleaned)

    if cleaned and any(trigger in cleaned.lower() for trigger in _EDGECASE_GUARDRAIL):
        kept: List[str] = []
        for line in cleaned.splitlines():
            is_header = (
                line.startswith("diff --git ")
                or line.startswith("index ")
                or line.startswith("--- ")
                or line.startswith("+++ ")
                or line.startswith("@@")
                or line.startswith("new file mode")
                or line.startswith("deleted file mode")
                or line.startswith("old mode ")
                or line.startswith("new mode ")
                or line.startswith("similarity index ")
                or line.startswith("dissimilarity index ")
                or line.startswith("rename from ")
                or line.startswith("rename to ")
                or line.startswith("copy from ")
                or line.startswith("copy to ")
                or line.startswith("Binary files ")
                or line.startswith("GIT binary patch")
            )
            if not is_header and any(trigger in line.lower() for trigger in _EDGECASE_GUARDRAIL):
                continue
            kept.append(line)
        rebuilt = "\n".join(kept)
        if cleaned.endswith("\n") and not rebuilt.endswith("\n"):
            rebuilt += "\n"
        cleaned = rebuilt

    return cleaned


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


_IMPORT_CONCAT_PATTERN = re.compile(r'[)};](?=import\s+[{*\w])')
_HUNK_HEADER_RE = re.compile(r'^@@ -(\d+(?:,\d+)?) \+(\d+)(?:,(\d+))? @@(.*)$')


def _split_comment_import_concat(diff_output: str) -> str:
    """Repair '+' lines where a // comment ending and a JS/TS import statement
    got concatenated onto one line (e.g. ')import {x}'). In JS/TS // extends to
    end of line, so the concatenation silently turns the import into a comment
    and breaks the file. Split at the boundary and bump the hunk's added-line
    count. Narrow trigger: '//' must precede a close-bracket that is immediately
    followed by 'import <brace-or-ident>' inside a '+' line."""
    if not diff_output.strip() or '//' not in diff_output:
        return diff_output
    if not _IMPORT_CONCAT_PATTERN.search(diff_output):
        return diff_output

    lines = diff_output.splitlines(keepends=True)
    out_lines: List[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        m = _HUNK_HEADER_RE.match(line.rstrip('\n'))
        if not m:
            out_lines.append(line)
            i += 1
            continue
        old_part = m.group(1)
        new_start = int(m.group(2))
        new_count = int(m.group(3)) if m.group(3) else 1
        tail = m.group(4)
        j = i + 1
        delta = 0
        body: List[str] = []
        while j < n and not lines[j].startswith('@@') and not lines[j].startswith('diff --git'):
            bl = lines[j]
            if bl.startswith('+') and not bl.startswith('+++') and '//' in bl:
                ends_nl = bl.endswith('\n')
                content = bl[1:].rstrip('\n')
                mm = _IMPORT_CONCAT_PATTERN.search(content)
                if mm and '//' in content[:mm.start()]:
                    left = content[:mm.end()].rstrip()
                    right = content[mm.end():].lstrip()
                    if left and right:
                        body.append('+' + left + '\n')
                        body.append('+' + right + ('\n' if ends_nl else ''))
                        delta += 1
                        j += 1
                        continue
            body.append(bl)
            j += 1
        new_header = '@@ -%s +%d,%d @@%s\n' % (old_part, new_start, new_count + delta, tail)
        out_lines.append(new_header)
        out_lines.extend(body)
        i = j
    return ''.join(out_lines)


def _should_skip_patch_path(relative_path: str) -> bool:
    path = Path(relative_path)
    if path.suffix in {".pyc", ".pyo"}:
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
        ".git",
    }
    generated_suffixes = {
        ".class",
        ".o",
        ".obj",
        ".so",
        ".dll",
        ".dylib",
        ".exe",
        ".bin",
    }
    return any(part in generated_parts for part in path.parts) or path.suffix.lower() in generated_suffixes


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
    files, top_score = _rank_context_files(repo, issue)
    tracked_set = set(_tracked_files(repo))

    # Rescue-ranker: weak top_score means no path mention and no symbol-grep
    # hit landed, so the top-ranked file is essentially random — this is
    # the dominant catastrophic-floor failure mode. Run a cheap broad-grep
    # over the full tracked set (no context-file filter) and surface the
    # 1-3 files that match the most issue terms. Also surface a banner
    # block in the preload so the model treats those files as the most
    # likely targets rather than guessing from path-mention-style cues.
    rescue_files: List[str] = []
    if top_score < _RESCUE_RANKER_TOP_SCORE_THRESHOLD:
        rescue_files = _broad_grep_fallback(repo, issue, tracked_set)
        if rescue_files:
            existing = set(files)
            files = [f for f in rescue_files if f not in existing] + files

    # Distinctive-phrase grep: surface files containing verbatim issue strings.
    phrase_grep = _distinctive_phrase_grep(repo, issue, tracked_set)
    phrase_match_files: List[str] = []
    for _phrase, hits in phrase_grep:
        for h in hits:
            if h not in phrase_match_files:
                phrase_match_files.append(h)
    if phrase_match_files:
        existing = set(files)
        files = [f for f in phrase_match_files if f not in existing] + files

    if not files:
        return "", []

    files = _augment_with_test_partners(files, tracked_set)
    files = _augment_with_integration_partners(files, tracked_set, issue)
    files = _augment_with_directory_siblings(files, tracked_set)

    parts: List[str] = []
    included: List[str] = []
    used = 0
    n_files = min(len(files), MAX_PRELOADED_FILES)
    per_file_budget = max(1500, MAX_PRELOADED_CONTEXT_CHARS // max(n_files, 1))

    if phrase_grep:
        # Content-search banner. Show each phrase + the files containing
        # it. Place BEFORE the rescue-ranker banner so the strongest
        # evidence is at the top of the preload.
        lines = ["### content-search hint"]
        lines.append(
            "The issue contains distinctive verbatim strings. The "
            "following file(s) contain those strings literally — they "
            "are the most likely targets. Inspect them first, BEFORE "
            "narrowing path-based searches:"
        )
        for phrase, hits in phrase_grep:
            display = phrase if len(phrase) <= 70 else phrase[:67] + "..."
            lines.append(f"  - {display!r}")
            for h in hits[:3]:
                lines.append(f"      → {h}")
        phrase_banner = "\n".join(lines) + "\n"
        parts.append(phrase_banner)
        used += len(phrase_banner)

    if rescue_files:
        # Banner is small and high-leverage; surface BEFORE the snippet
        # blocks so the model reads it before any file content. Marker
        # comments are stable so _strip_preloaded_section keeps treating
        # this block correctly.
        rescue_banner = (
            "### rescue-ranker hint\n"
            "The issue does not directly name a file or identifier present in "
            "this repository. The following file(s) matched the most issue "
            "terms via a broad text search and are the most likely targets — "
            "inspect them first before running broader searches:\n"
            + "".join(f"  - {p}\n" for p in rescue_files)
        )
        parts.append(rescue_banner)
        used += len(rescue_banner)

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


def _rank_context_files(repo: Path, issue: str) -> Tuple[List[str], int]:
    """Returns (ranked_paths, top_score). top_score is the highest computed
    score in the scoring pass; callers use it to detect "weak ranking"
    rounds where no path/identifier signal hit, so the top file is
    functionally random and the rescue-ranker fallback should fire.
    """
    tracked = _tracked_files(repo)
    if not tracked:
        return [], 0

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
    symbol_hits = _symbol_grep_hits(repo, tracked_set, issue)
    id_boost = _issue_identifier_path_boost(issue, list(tracked_set))
    key_content_hits = _grep_for_issue_quoted_keys(repo, issue, tracked_set)
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
        if stem_lower and len(stem_lower) >= 5 and stem_lower in issue_lower:
            score += 16
        score += sum(3 for term in terms if term in path_lower)
        if "/test" in path_lower or "spec." in path_lower or ".test." in path_lower:
            score += sum(2 for term in terms if term in path_lower)
        # Boost files whose contents reference identifiers from the issue.
        if relative_path in symbol_hits:
            score += 60 + min(40, 8 * symbol_hits[relative_path])
        # Boost files whose path/name matches identifier-shaped tokens from the issue.
        score += 35 * id_boost.get(relative_path, 0)
        # Boost files whose CONTENT contains snake_case keys from the issue
        # (e.g. `foryou_based_on_server` mentioned in the issue → boost the
        # i18n file where that key currently lives). Targets the
        # canonical-location problem: agent picks `locales/*.json` instead of
        # `utils/botStrings.js` when both exist but only one has the keys.
        score += 50 * key_content_hits.get(relative_path, 0)
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
    top_score = scored[0][0] if scored else 0
    if mentioned:
        # Explicit path or backtick-ident match: ranking is strong even if
        # the scored list is empty (mentioned files bypass the score loop).
        top_score = max(top_score, 100)
    return ranked, top_score


# Threshold below which _rank_context_files is treated as "weak signal" and
# the rescue-ranker broad-grep fallback fires. 60 = the floor of the
# symbol-grep boost (60 + 8*hits); below it means no path mention and no
# symbol-grep hit landed.
_RESCUE_RANKER_TOP_SCORE_THRESHOLD = 60
_RESCUE_RANKER_MAX_FALLBACK_FILES = 3
_RESCUE_RANKER_MIN_TERM_LEN = 5
_RESCUE_RANKER_MAX_TERMS = 6


# Non-ASCII phrase regex. Bounded to a single line — `[A-Za-z ]` not `\s`
# — so we never capture across newlines, which would otherwise produce
# prose fragments like "...destination URL.\nThe".
_DISTINCTIVE_PHRASE_RE_NONASCII = re.compile(
    r"[A-Za-z]*[^\x00-\x7f][A-Za-z À-￿]{6,}[^\x00-\x7f][A-Za-z]*"
)

# Common English stop-words used to reject prose-fragment phrases. A
# phrase with 2+ of these is almost certainly sentence-fragment prose
# that won't grep-match meaningfully.
_PHRASE_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "have", "if", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "their", "then", "to", "was", "were", "when", "which", "with",
})


def _phrase_is_prose_fragment(phrase: str) -> bool:
    """Reject phrases that look like English sentence fragments. Such
    phrases over-match in git grep and surface irrelevant files as
    spurious content-search banner hits."""
    tokens = re.findall(r"[A-Za-z]+", phrase.lower())
    if len(tokens) <= 1:
        return False
    stop_n = sum(1 for t in tokens if t in _PHRASE_STOPWORDS)
    # 2+ stop-words suggests it's sentence-shape (e.g. "the X is Y").
    return stop_n >= 2


def _extract_distinctive_phrases(issue_text: str) -> List[str]:
    """Pull up to 3 verbatim-quotable strings from the issue (non-ASCII segments + quoted strings >= 8 chars), longest first."""
    if not issue_text:
        return []
    out: List[str] = []
    seen: Set[str] = set()
    for m in _DISTINCTIVE_PHRASE_RE_NONASCII.finditer(issue_text):
        s = m.group(0).strip()
        if not (10 <= len(s) <= 80) or s in seen:
            continue
        if _phrase_is_prose_fragment(s):
            continue
        out.append(s)
        seen.add(s)
    for q in ["'", '"', '`']:
        pat = q + r"([^" + re.escape(q) + r"\n]{8,80})" + q
        for m in re.finditer(pat, issue_text):
            s = m.group(1).strip()
            if not s or s in seen:
                continue
            # Reject one-word quotes (mostly identifiers handled by
            # _symbol_grep_hits already) — distinctive phrases are
            # multi-word.
            if " " not in s and len(s) < 20:
                continue
            if _phrase_is_prose_fragment(s):
                continue
            out.append(s)
            seen.add(s)
    out.sort(key=len, reverse=True)
    return out[:3]


def _distinctive_phrase_grep(
    repo: Path, issue_text: str, tracked: set,
) -> List[Tuple[str, List[str]]]:
    """For each distinctive phrase, run `git grep -l -F -- '<phrase>'`
    against the tracked set and return (phrase, matching_files) pairs
    for phrases that hit at least one tracked file.

    Higher-signal than _broad_grep_fallback: long verbatim phrases are
    far less likely to occur in unrelated files than single keywords.
    """
    phrases = _extract_distinctive_phrases(issue_text)
    if not phrases or not tracked:
        return []
    pairs: List[Tuple[str, List[str]]] = []
    for phrase in phrases:
        try:
            proc = subprocess.run(
                ["git", "grep", "-l", "-F", "--", phrase],
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
        hits = [
            line.strip() for line in proc.stdout.splitlines()
            if line.strip() and line.strip() in tracked
        ]
        if hits:
            # Cap matches per phrase — a phrase that hits 20 files is
            # too generic to be useful.
            pairs.append((phrase, hits[:5]))
    return pairs


def _broad_grep_fallback(repo: Path, issue_text: str, tracked: set) -> List[str]:
    """Rescue-ranker: scan tracked files by raw issue-term match count when the primary ranker has no strong signal. Returns paths matching at least 2 distinct terms."""
    if not tracked:
        return []
    terms = [t for t in _issue_terms(issue_text) if len(t) >= _RESCUE_RANKER_MIN_TERM_LEN][:_RESCUE_RANKER_MAX_TERMS]
    if not terms:
        return []
    # Parallelized grep: saves ~1-3s on preload setup compared to
    # sequential, freeing wall-clock for editing.
    def _grep_term(term: str) -> List[str]:
        try:
            proc = subprocess.run(
                ["git", "grep", "-l", "-i", "-F", "--", term],
                cwd=str(repo), capture_output=True, text=True, timeout=3,
            )
        except Exception:
            return []
        if proc.returncode not in (0, 1):
            return []
        return [
            line.strip() for line in proc.stdout.splitlines()
            if line.strip() and line.strip() in tracked
        ]

    hits: Dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=min(len(terms), 6)) as pool:
        futures = {pool.submit(_grep_term, t): t for t in terms}
        for future in as_completed(futures):
            for path in future.result():
                hits[path] = hits.get(path, 0) + 1
    candidates = [(count, path) for path, count in hits.items() if count >= 2]
    candidates.sort(key=lambda item: (-item[0], len(item[1]), item[1]))
    return [path for _count, path in candidates[:_RESCUE_RANKER_MAX_FALLBACK_FILES]]


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


_DIRECTORY_SIBLING_BASENAMES = {
    "layout", "index", "page", "route", "loading", "error", "metadata",
    "manifest", "head", "template", "_meta", "_root", "styles", "types",
    "constants", "schema",
}


def _augment_with_directory_siblings(
    files: List[str], tracked_set: set, limit: int = 3
) -> List[str]:
    """Append same-directory siblings of the top-ranked file that the pipeline hasn't included yet.

    Targets high-leverage basenames (layout, index, schema, etc.) that commonly
    need co-editing on multi-file tasks. Uses only set membership — no I/O, no subprocess.
    """
    try:
        if not files:
            return files
        top = files[0]
        top_dir = str(Path(top).parent).replace("\\", "/")
        if top_dir in {"", "."}:
            return files
        seen = set(files)
        siblings: List[str] = []
        for candidate in tracked_set:
            if candidate in seen:
                continue
            cpath = Path(candidate)
            if str(cpath.parent).replace("\\", "/") != top_dir:
                continue
            if cpath.stem.lower() in _DIRECTORY_SIBLING_BASENAMES:
                siblings.append(candidate)
            if len(siblings) >= limit:
                break
        return files + siblings[:limit]
    except Exception:
        return files


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
    issue_symbols = {s.lower() for s in _extract_issue_symbols(issue, max_symbols=16)}
    signal_tokens = {t for t in (anchor_tokens | issue_tokens | issue_symbols) if len(t) >= 4}
    root_file_wanted = bool(
        issue_tokens
        & {
            "build", "cli", "config", "dependency", "dependencies", "docker",
            "package", "script", "setup", "workflow",
        }
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
        path_words = _split_path_tokens(relative_path)
        def _token_in_path(tok: str) -> bool:
            return tok in path_words if len(tok) < 5 else tok in path_lower
        score += min(8, 2 * sum(1 for token in issue_tokens if _token_in_path(token)))
        score += min(8, 3 * sum(1 for token in issue_symbols if _token_in_path(token)))
        score += min(6, 2 * sum(1 for token in signal_tokens if _token_in_path(token)))
        if path.name in _INTEGRATION_ROOT_FILES and root_file_wanted:
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



# -----------------------------
# Hunk classifiers + diff hygiene
# -----------------------------
#
# Two failure modes produce low-quality patches: drive-by whitespace /
# comment / blank-line edits, and patches that cover the wrong files. The
# helpers below detect both. They're applied at two stages:
#
# 1. At patch-return time: low-signal hunks are silently dropped from the
# final diff (so the validator never sees them).
# 2. Inside the loop: when the model's draft contains junk, we queue a
# "polish" turn that asks the model to revert those hunks itself, since
# doing so cleanly is safer than mechanical filtering for borderline cases
# (e.g., a comment edit that genuinely matters).

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


_NEW_FILE_RE = re.compile(
    r"^--- /dev/null\n\+\+\+ b/(.+?)$",
    re.MULTILINE,
)
_RELOCATION_TRIGGERS = re.compile(
    r"\b(move|rename|extract|belongs under|new location|create a new|convert to)\b",
    re.IGNORECASE,
)


def _patch_newly_created_files(patch: str) -> List[str]:
    """Return paths of files created from scratch (--- /dev/null) in the patch."""
    try:
        return [m.group(1) for m in _NEW_FILE_RE.finditer(patch)]
    except Exception:
        return []


def _check_inplace_intent(
    patch: str, issue_text: str, tracked_set: set
) -> List[str]:
    """Return advisories when the patch creates a new file while an existing same-basename file was not edited.

    Catches the 'new file at wrong path instead of in-place refactor' failure mode.
    Suppressed when the issue contains a relocation trigger phrase.
    """
    try:
        if _RELOCATION_TRIGGERS.search(issue_text):
            return []
        advisories: List[str] = []
        changed = set(_patch_changed_files(patch))
        for new_path in _patch_newly_created_files(patch)[:6]:
            new_basename = Path(new_path).name
            for existing in tracked_set:
                if existing in changed:
                    continue
                if Path(existing).name == new_basename:
                    advisories.append(
                        f"created new file {new_path!r} while existing {existing!r} "
                        "with same name was untouched"
                    )
                    break
            if len(advisories) >= 3:
                break
        return advisories
    except Exception:
        return []


_REMOVED_DEF_RES = (
    re.compile(r"^-\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("),
    re.compile(r"^-\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\b"),
    re.compile(r"^-\s*function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("),
    re.compile(r"^-\s*export\s+(?:default\s+)?(?:const|function|class)\s+([A-Za-z_][A-Za-z0-9_]*)"),
    re.compile(r"^-\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*\("),
    re.compile(r"^-\s*fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*[<(]"),
)


def _patch_removed_definitions(patch: str, cap: int = 8) -> List[str]:
    """Return names of definitions removed by the patch (def/class/function/export/func/fn lines).

    Pure diff-text scan — no subprocess, no I/O. Used to build a caller-audit advisory.
    """
    try:
        seen: set = set()
        results: List[str] = []
        for line in patch.splitlines():
            if not line.startswith("-"):
                continue
            for pattern in _REMOVED_DEF_RES:
                m = pattern.match(line)
                if m:
                    name = m.group(1)
                    if name not in seen:
                        seen.add(name)
                        results.append(name)
                    break
            if len(results) >= cap:
                break
        return results
    except Exception:
        return []


_READ_CMD_HEADS = {"cat", "less", "more", "head", "tail"}


def _normalize_repo_path(path: str) -> str:
    """Strip docker-workdir absolute prefix (`/work/repo/`) and leading
    slash so the same file produces a consistent string regardless of
    how the model wrote it. The agent runs inside docker with
    CWD=/work/repo; the model alternates between relative and absolute
    paths mid-solve (e.g. step 1 uses relative, steps 3+ switch to
    /work/repo/...). Path detectors and PE dispatch need a normalized
    form.
    """
    if not path:
        return path
    s = path.strip().strip("'\"")
    for prefix in ("/work/repo/", "/work/", "./"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    s = s.lstrip("/")
    return s


_PYTHON_READ_OPEN_RE = re.compile(
    r"open\s*\(\s*['\"]([^'\"]+)['\"]"
    r"(?:\s*,\s*['\"](r|rb|rt)['\"])?\s*[,)]"
)
_PYTHON_READ_PATH_RE = re.compile(
    r"Path\s*\(\s*['\"]([^'\"]+)['\"]\s*\)\s*\."
    r"(?:read_text|read_bytes|read)\s*\("
)
_PYTHON_WRITE_HINT_RE = re.compile(
    r"\.write_text\s*\(|\.write_bytes\s*\(|"
    r"open\s*\([^,)]+,\s*['\"][wa]b?\+?['\"]\s*[,)]"
)


def _extract_python_read_target(cmd: str) -> Optional[str]:
    """If cmd is a read-only python3 heredoc, return the first read target. None if any write hint is present."""
    if "python" not in cmd.split("\n", 1)[0]:
        return None
    if _PYTHON_WRITE_HINT_RE.search(cmd):
        return None  # has writes — let it run, not a pure read
    m = _PYTHON_READ_OPEN_RE.search(cmd)
    if m and ".read" in cmd:  # confirm a read call exists somewhere
        return _normalize_repo_path(m.group(1))
    m = _PYTHON_READ_PATH_RE.search(cmd)
    if m:
        return _normalize_repo_path(m.group(1))
    return None


def _extract_read_target(cmd: str) -> Optional[str]:
    """If cmd is a pure read on a single file, return that (normalized) path. Handles cat/sed/head/tail, `cat | sed -n` pipes, and python3 heredoc opens. Returns None for compound commands."""
    # Python heredoc reads — check first so we don't reject them as
    # compound commands via the pipe/redirect guard below.
    py_target = _extract_python_read_target(cmd)
    if py_target:
        return py_target
    first_line = cmd.strip().split("\n")[0].strip()
    if not first_line:
        return None
    # Reject true compound commands but allow simple pipe-through-filter
    # since paginate-via-pipe is common: `cat -n FILE | sed -n '...p'`.
    if any(tok in first_line for tok in [">", "<", ";", "&&", "||", "$(", "`"]):
        return None
    parts = first_line.split()
    if not parts:
        return None
    # Special case: piped reads. e.g. `cat -n FILE | sed -n '200,500p'`
    if "|" in first_line:
        # Split on the first pipe, examine each side.
        lhs, rhs = first_line.split("|", 1)
        lhs_parts = lhs.split()
        rhs_first = rhs.strip().split()
        if not lhs_parts:
            return None
        if lhs_parts[0] not in _READ_CMD_HEADS:
            return None
        # RHS must also be a pure-read filter (sed -n, head, tail) — no
        # mutation.
        if not rhs_first:
            return None
        if rhs_first[0] not in {"sed", "head", "tail", "less", "more"}:
            return None
        if rhs_first[0] == "sed" and "-n" not in rhs_first:
            return None
        # Extract path from LHS (last non-flag token).
        for tok in reversed(lhs_parts[1:]):
            s = tok.strip("'\"")
            if not s or s.startswith("-"):
                continue
            if "/" in s or "." in s or re.match(r"^[A-Za-z_][\w-]*$", s):
                return _normalize_repo_path(s)
        return None
    head = parts[0]
    if head == "sed":
        if "-n" not in parts:
            return None
    elif head not in _READ_CMD_HEADS:
        return None
    for tok in reversed(parts[1:]):
        s = tok.strip("'\"")
        if not s or s.startswith("-"):
            continue
        # sed address range like '100,200p' — skip
        if re.match(r"^\d+(?:,\d+)?[pPdq]?$", s):
            continue
        if "/" in s or "." in s or re.match(r"^[A-Za-z_][\w-]*$", s):
            return _normalize_repo_path(s)
    return None


def _revert_syntactically_broken_files(repo: Path, patch: str) -> List[str]:
    """Revert files in the patch whose current working-tree state is syntactically broken."""
    errors = _check_syntax(repo, patch)
    if not errors:
        return []
    reverted: List[str] = []
    # Each error string typically starts with "<path>:" (per per-language
    # checker conventions). Extract the path prefix.
    for err in errors:
        rel = err.split(":", 1)[0].strip()
        if not rel:
            continue
        full = (repo / rel).resolve()
        try:
            full.relative_to(repo.resolve())
        except ValueError:
            continue
        if not full.exists():
            # File was newly created and is broken: remove it entirely so
            # the patch drops the broken new-file diff.
            try:
                proc = subprocess.run(
                    ["git", "rm", "-f", "--", rel],
                    cwd=str(repo), capture_output=True, text=True, timeout=5,
                )
                if proc.returncode == 0:
                    reverted.append(rel + " [removed-new]")
            except Exception:
                continue
            continue
        try:
            proc = subprocess.run(
                ["git", "checkout", "HEAD", "--", rel],
                cwd=str(repo), capture_output=True, text=True, timeout=5,
            )
            if proc.returncode == 0:
                reverted.append(rel)
        except Exception:
            continue
    return reverted


def _revert_docker_mount_mode_artifacts(repo: Path) -> int:
    """Restore HEAD mode on files whose diff is mode-only.

    Docker volume mounts strip the +x bit from bash scripts at mount time,
    creating spurious 100755 → 100644 mode-only diffs that the agent never
    caused. `_strip_mode_only_hunks` cleans the returned patch STRING, but
    the validator generates the submitted diff by running `git diff` on the
    working tree — so we must also restore the mode on disk. Runs
    `git checkout HEAD -- <path>` on each file whose diff is mode-only AND
    has no content hunks AND is not a new/deleted file diff.
    """
    try:
        proc = subprocess.run(
            ["git", "diff", "--binary"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=30,
        )
        diff = proc.stdout or ""
    except Exception:
        return 0
    if not diff or "diff --git " not in diff:
        return 0
    n = 0
    for block in re.split(r"(?m)^(?=diff --git )", diff):
        if not block.startswith("diff --git "):
            continue
        has_mode = bool(re.search(r"(?m)^(?:old|new) mode \d+", block))
        has_hunks = bool(re.search(r"(?m)^@@ ", block))
        has_new_file = bool(re.search(r"(?m)^new file mode \d+", block))
        has_delete = bool(re.search(r"(?m)^deleted file mode \d+", block))
        if not has_mode or has_hunks or has_new_file or has_delete:
            continue
        m = re.match(r"^diff --git a/(.+?) b/(.+?)$", block.splitlines()[0])
        if not m:
            continue
        path = m.group(2)
        try:
            proc = subprocess.run(
                ["git", "checkout", "HEAD", "--", path],
                cwd=str(repo),
                capture_output=True,
                text=True,
                timeout=5,
            )
            if proc.returncode == 0:
                n += 1
        except Exception:
            continue
    return n


def _strip_mode_only_hunks(patch: str) -> str:
    """Remove diff blocks that only change file mode with no content edits."""
    if not patch or "diff --git " not in patch:
        return patch
    blocks = re.split(r"(?m)^(?=diff --git )", patch)
    kept: List[str] = []
    for block in blocks:
        if not block.strip():
            continue
        if not block.startswith("diff --git "):
            kept.append(block)
            continue
        has_mode = bool(re.search(r"(?m)^(?:old|new) mode \d+", block))
        has_hunks = bool(re.search(r"(?m)^@@ ", block))
        has_new_file = bool(re.search(r"(?m)^new file mode \d+", block))
        has_delete = bool(re.search(r"(?m)^deleted file mode \d+", block))
        if has_mode and not (has_hunks or has_new_file or has_delete):
            continue
        kept.append(block)
    return "".join(kept)


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
# Real validator tasks come from real GitHub commits, so a sizeable fraction
# touch TypeScript, JavaScript, JSON, YAML, etc. This module checks each
# touched file with the cheapest available tool, falling back gracefully
# when tools are missing. Errors come back as (path:line: msg) strings so
# the syntax-fix prompt can quote them.


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
# - C / C++ / C# / Java / Kotlin / Scala — `'X'` is a character literal
# (so `char c = '}';` flips into string mode and eats until next ')
# - Rust — `'a` is a lifetime annotation
# - Go — `'X'` is a rune literal
# Net effect of including those: a single `'X'` in any function would yield
# a phantom imbalance that triggers a wasted syntax_fix turn. We restrict
# to JS-family + Swift, where ' is a real string delimiter.
_BRACE_BALANCE_SUFFIXES = {
    ".ts", ".tsx", ".jsx", ".swift",
    ".rs", ".go", ".java", ".kt",
    ".c", ".cc", ".cpp", ".h", ".hpp",
    ".cs", ".php",
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
    Surfaces the runtime-correctness signal as a refinement gate: if any
    edited file has a partner test that actually fails, the failure tail
    is fed back to the model for one fix turn.
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
_CRITERIA_MAX_BULLETS = 16  # cap on acceptance-criteria bullets surfaced to the model
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
# Deletion-gap detection
# -----------------------------
#
# Duel data shows the king loses rounds where the issue says "remove X" or
# "delete Y" but the patch contains zero deletion lines — the model added
# the new behaviour without removing the old one.  This gate detects that
# mismatch cheaply and surfaces a targeted nudge before <final>.

_DELETION_VERB_RE = re.compile(
    r"\b(remove|delete|drop|eliminate|deprecate|strip|replace|clear|unlink|erase|undo|disable|deactivate)\b",
    re.IGNORECASE,
)


def _patch_has_deletions(patch: str) -> bool:
    """True if the patch contains at least one substantive deletion line."""
    for line in patch.splitlines():
        if line.startswith("-") and not line.startswith("---"):
            if line[1:].strip():  # ignore blank-line removals
                return True
    return False


def _issue_requires_deletion(issue_text: str) -> bool:
    """True if the issue contains explicit removal/replacement verbs."""
    return bool(_DELETION_VERB_RE.search(issue_text))


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


_IDENTIFIER_STOPWORDS = {
    "The", "This", "When", "Then", "User", "API", "URL", "HTTP", "JSON",
    "HTML", "CSS", "SQL", "None", "True", "False", "Error", "Type", "List",
    "Dict", "Path", "File", "Data", "Test", "Base", "From", "With", "That",
    "Card", "Modal", "Form", "Button", "Input", "Label", "Image", "Menu",
    "Header", "Footer", "Layout", "Page", "Section", "Container", "Wrapper",
    "Element", "Component", "Service", "Manager", "Handler", "Provider",
    "Context", "Status", "Value", "Field", "Item", "Result", "Response",
    "Request", "Config", "Settings", "Options", "Properties", "Default",
    "Module", "Class", "Object", "String", "Number", "Array", "Function",
    "Method", "Action", "State", "Store", "Schema", "Model", "View",
    "Index", "Route", "Router", "Render", "Update", "Create", "Delete",
    "useState", "useEffect", "useCallback", "useMemo", "useRef", "useContext",
    "useRouter", "setState", "setValue", "getValue", "getDefault",
    "handleClick", "handleChange", "handleSubmit", "handleClose", "handleError",
    "fetchData", "createElement", "createContext", "buildPath", "buildUrl",
}

_HOOK_GENERIC_PREFIXES_LOWER = {
    "usestate", "useeffect", "usecallback", "usememo", "useref", "usecontext",
    "userouter", "setstate", "setvalue", "getvalue", "getdefault",
    "handleclick", "handlechange", "handlesubmit", "handleclose", "handleerror",
    "fetchdata", "createelement", "createcontext", "buildpath", "buildurl",
}

_CAMEL_RE = re.compile(r"\b([A-Z][a-zA-Z0-9_]{3,})\b")
_HOOK_RE = re.compile(r"\b(use|get|set|fetch|handle|build|create)[A-Z][a-zA-Z0-9_]{2,}\b")
_SNAKE_RE = re.compile(r"\b([a-z][a-zA-Z0-9]+_[a-z][a-zA-Z0-9_]+)\b")


_QUOTED_KEY_RE = re.compile(r"[`'\"]([a-z][a-z0-9]+(?:_[a-z0-9]+){1,5})[`'\"]")


def _grep_for_issue_quoted_keys(
    repo: Path, issue_text: str, tracked_set: set,
) -> Dict[str, int]:
    """Find tracked files containing backtick/quoted snake_case keys from the issue. Skips short / verb / over-broad keys."""
    keys: set = set()
    for m in _QUOTED_KEY_RE.finditer(issue_text):
        key = m.group(1)
        if len(key) >= 6 and "_" in key:
            keys.add(key)
    if not keys:
        return {}
    hits: Dict[str, int] = {}
    for key in list(keys)[:8]:
        try:
            proc = subprocess.run(
                ["git", "grep", "-l", "-F", key],
                cwd=str(repo), capture_output=True, text=True, timeout=8,
            )
            if proc.returncode != 0 or not proc.stdout:
                continue
            matched_files = proc.stdout.strip().split("\n")
            if not matched_files or len(matched_files) > 12:
                continue
            for f in matched_files:
                f = f.strip()
                if f and f in tracked_set:
                    hits[f] = hits.get(f, 0) + 1
        except Exception:
            continue
    return hits


def _issue_identifier_path_boost(
    issue_text: str, tracked_files: List[str], cap: int = 20
) -> Dict[str, int]:
    """Return per-file hit counts for identifier-shaped tokens extracted from the issue text.

    Uses only path-segment substring matching — no I/O, no subprocess.
    Weight 35 per hit matches the existing path-mention scoring bonus.
    """
    try:
        identifiers: set = set()
        for m in _CAMEL_RE.finditer(issue_text):
            tok = m.group(1)
            if tok not in _IDENTIFIER_STOPWORDS and len(identifiers) < cap:
                identifiers.add(tok.lower())
        for m in _HOOK_RE.finditer(issue_text):
            hook_lower = m.group(0).lower()
            if hook_lower in _HOOK_GENERIC_PREFIXES_LOWER:
                continue
            if len(identifiers) < cap:
                identifiers.add(hook_lower)
        for m in _SNAKE_RE.finditer(issue_text):
            if len(identifiers) < cap:
                identifiers.add(m.group(1).lower())
        if not identifiers:
            return {}
        boost: Dict[str, int] = {}
        for rel in tracked_files:
            path_obj = Path(rel)
            basename_lower = path_obj.name.lower()
            parent_lower = str(path_obj.parent).lower()
            hits = sum(1 for ident in identifiers if ident in basename_lower or ident in parent_lower)
            if hits:
                boost[rel] = hits
        return boost
    except Exception:
        return {}


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
    def _grep_symbol(symbol: str) -> List[str]:
        try:
            proc = subprocess.run(
                ["git", "grep", "-l", "-F", "--", symbol],
                cwd=str(repo), capture_output=True, text=True, timeout=4,
            )
        except Exception:
            return []
        if proc.returncode not in (0, 1):
            return []
        return [
            line.strip() for line in proc.stdout.splitlines()
            if line.strip() and line.strip() in tracked_set
            and _context_file_allowed(line.strip())
        ]

    hits: Dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=min(len(symbols), 8)) as pool:
        futures = {pool.submit(_grep_symbol, s): s for s in symbols}
        for future in as_completed(futures):
            for path in future.result():
                hits[path] = hits.get(path, 0) + 1
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
- Integration cascade: if the issue describes a feature spanning multiple concerns (page + route + nav + data fetch; or model + migration + serializer + view + URL), enumerate EVERY required integration point as its own plan row even when the issue does not explicitly bullet them.
- Likely target: name likely files/functions/classes/modules to inspect or modify.
- Edits: for each likely target, ONE line in the form `path/to/file: <specific change you will write>`. Do NOT say "examine" or "inspect" — commit to a concrete write. If unsure of exact lines, still commit to the file and approach.
- Strategy: smallest root-cause fix; surgical edits, no "examine then decide" — your next command after the plan should already be moving toward a write.
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

When the issue or codebase implies a specific approach — an existing constant, a library already present in imports or package.json/requirements.txt, a utility already used in adjacent code, a pattern already established in the file — use exactly that. Do NOT invent a custom equivalent. The reference patch almost always takes the most direct implementation the codebase already supports: use the named constant, not a hardcoded string; use the existing helper, not a reimplementation; use the library the project already imports, not a hand-rolled substitute.

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
SAFETY
====================================================================

No sudo. No chmod. No file deletion. No destructive git commands. No network access outside the validator proxy. No host secrets, dot-env files, credentials, hidden tests, evaluator files, or scoring metadata.

Do not write code comments, log messages, or strings containing "automatic fail", "guaranteed zero", "score zero", or "auto-fail" — those substrings get stripped from your patch and break it.

====================================================================
PARALLEL EDITS (optional, multi-file only)
====================================================================

For multi-file tasks (3+ INDEPENDENT files already identified from preloaded snippets), you may emit:

<parallel_edits>
<edit file="path/to/foo.py">one-paragraph instruction for THIS file</edit>
<edit file="path/to/bar.tsx">one-paragraph instruction for THIS file</edit>
</parallel_edits>

Concurrent sub-agents will execute the edits in parallel (cap 6 per batch). For single-file or sequential-by-nature work, normal `<command>` is faster and clearer.
'''


_PRELOAD_BEGIN_MARKER = "<!-- preloaded-context-begin -->"
_PRELOAD_END_MARKER = "<!-- preloaded-context-end -->"


def build_initial_user_prompt(issue: str, repo_summary: str, preloaded_context: str = "") -> str:
    context_section = ""
    if preloaded_context.strip():
        context_section = f"""
{_PRELOAD_BEGIN_MARKER}
Preloaded likely relevant tracked-file snippets (already read for you — do not re-read):

{preloaded_context}
{_PRELOAD_END_MARKER}
"""

    return f"""Fix this issue:

{issue}

Repository summary:

{repo_summary}
{context_section}
Before planning, read the ENTIRE issue above and identify every requirement (there may be more than one). Your patch must satisfy ALL of them — the LLM judge penalizes incomplete solutions.

Scope discipline: only modify files the task explicitly names plus the minimum cross-file edits needed to keep them compiling. Do not rewrite working code in adjacent files even if a refactor would improve it; the judge penalises unrelated rewrites as heavily as missed requirements.

Strategy: the fix is typically in ONE specific function or block. Identify it precisely, then make the minimal edit that fixes the ROOT CAUSE.

If the preloaded snippets show the target code, edit them directly — do not re-read or run broad searches first. If the target is unclear, run ONE or TWO focused grep/sed -n commands to locate it, then edit immediately.

When multiple files need edits, include EVERY independent edit command in the SAME response. Do not split edits across turns.

After patching, run the most targeted test available (`pytest tests/test_X.py -x -q`, `go test ./...`, etc.) to verify correctness. Then finish with <final>...</final>.
"""


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
            "Budget check: no substantive edits yet. "
            "Your next command must edit the most likely file using what you already know from the issue and preloaded snippets. "
            "A precise sed or python -c is better than another grep. Stop exploring."
        )
    if step < 8:
        return (
            "Hard budget check: patch is still trivial or stalled. "
            "Your next command MUST make a substantive code change — even a best-effort edit to the most obvious location. "
            "Do not read files or run tests until after real edits exist. "
            "Use `sed -i` or a python one-liner to make the targeted edit now."
        )
    return (
        f"STOP READING. Step {step} and the patch is still trivial or has stopped growing. "
        "You have already paged through enough of the file. Your NEXT command MUST be a write "
        "command — `sed -i 's/old/new/' path` or `python3 - <<'PY'` writing "
        "to the file. NOT another `sed -n` / `cat` / `grep`. NOT another "
        "`<plan>`. A WRITE. Even a guess is better than continuing to read. "
        "If you don't know exactly what to change, edit the most likely line "
        "based on your best understanding so far."
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


def build_parallel_nudge_prompt(
    changed_files: List[str],
    remaining_paths: List[str],
    issue_text: str,
) -> str:
    """nudge the model toward <parallel_edits> when we detect it's doing
    multi-file work sequentially. Fires when the patch already shows ≥3
    changed files AND there are still issue-mentioned paths untouched."""
    todo_block = "\n  ".join(f"- {p}" for p in remaining_paths[:6]) or "(see issue)"
    if changed_files:
        intro = (
            "Multi-file efficiency check — you've already edited:\n"
            "  " + "\n  ".join(f"- {p}" for p in changed_files[:6]) + "\n\n"
            "...sequentially. These paths still need work:\n"
            f"  {todo_block}\n\n"
            "Sequential <command> blocks for distinct files burn wall-clock you "
            "may not get back. Switch to `<parallel_edits>` for the remaining "
            "files NOW."
        )
    else:
        # Exploration-spree variant: no edits yet, model has been cat/grep'ing.
        intro = (
            "Wall-clock check — you've spent several turns exploring without "
            "editing any file. The issue lists multiple acceptance criteria "
            "implying these targets:\n"
            f"  {todo_block}\n\n"
            "Stop reading and start editing. Emit `<parallel_edits>` for the "
            "files the issue requires — sequential <command> writes will not "
            "fit in remaining wall-clock."
        )
    return (
        intro + "\n\n"
        "One <edit file=\"...\"> per file, with a concise instruction. The "
        "Ralph sub-agents will execute them in parallel and you'll see a "
        "[parallel_edits_done] summary on your next turn.\n\n"
        "Task (for reference):\n"
        f"{issue_text[:1500]}\n\n"
        "Emit `<parallel_edits>` now."
    )


# Relocation phrases: "move X to Y", "split as separate", "correct import path", etc.
_RELOCATION_PHRASE_RE = re.compile(
    r"(?:"
    r"(?:move|relocate|rebuild|extract|split|migrate|reorganize)\s+(?:\S+\s+){0,6}?"
    r"(?:page|pages|file|files|component|components|module|modules|screen|screens|view|views|directory|folder|location|path)"
    r"|"
    r"(?:correct|fix|update|change)\s+(?:the\s+)?import\s+path"
    r"|"
    r"(?:create|add)\s+(?:\S+\s+){0,4}?(?:new|separate|standalone)\s+"
    r"(?:file|page|component|module|screen|view)"
    r"|"
    r"to\s+(?:its|a|the)\s+(?:new|own|proper|correct)\s+"
    r"(?:location|path|directory|folder|module|file)"
    r"|"
    r"(?:rebuild|reorganize|restructure)\s+(?:\S+\s+){0,6}?as\s+separate"
    r")",
    re.IGNORECASE,
)


def _issue_implies_relocation(issue_text: str) -> bool:
    """True if the issue text implies a file should be CREATED at a new path."""
    return bool(_RELOCATION_PHRASE_RE.search(issue_text))


def _patch_creates_any_new_file(patch: str) -> bool:
    """True if the patch contains at least one `new file mode` header or rename."""
    for line in patch.splitlines():
        if line.startswith("new file mode ") or line.startswith("rename to "):
            return True
    return False


def build_coverage_nudge_prompt(
    missing_paths: List[str],
    issue_text: str,
    relocation_gap: bool = False,
    removed_names: Optional[List[str]] = None,
) -> str:
    """Tell the model which issue-mentioned paths are still untouched.

    Incomplete coverage is common on multi-file tasks. When the issue names
    specific files and the draft skips them, surface that gap directly — much
    cheaper than hoping the self-check catches it. When `relocation_gap` is
    set, also instruct the model to CREATE a new file at the implied path.
    """
    bullets = "\n  ".join(f"- {p}" for p in missing_paths[:8]) or "(none)"
    relocation_hint = ""
    if relocation_gap:
        relocation_hint = (
            "RELOCATION GAP — the task implies a file should exist at a NEW path "
            "(phrases like 'move X to Y', 'rebuild as separate components', "
            "'correct the import path to the new location', 'create a new "
            "screen/page file'), but your current patch contains NO `new file "
            "mode` header. The model frequently mis-reads relocation as "
            "'edit-in-place'. Create the new file at the implied path with "
            "`cat > path/to/new_file.ext <<'EOF' ... EOF`, then update every "
            "importer/caller to reference the NEW path. Do not leave the old "
            "file unchanged unless the task explicitly says to keep both.\n\n"
        )
    removed_hint = ""
    if removed_names:
        names_str = ", ".join(removed_names[:8])
        removed_hint = (
            f"AUDIT: this patch removes/renames the following names — "
            f"verify every caller has been updated: {names_str}. "
            "Run `git grep` for each before <final> if uncertain.\n\n"
        )
    return (
        f"{relocation_hint}"
        f"{removed_hint}"
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


def _self_check_type_cue(issue_text: str) -> str:
    """Return a 1-line type-specific cue prepended to the self-check prompt.

    Heuristic keyword scan over the issue text. Empty when no strong type signal lands
    (the prompt then degrades to the generic boilerplate). Cheap and stateless.
    """
    text = issue_text.lower()
    if any(tok in text for tok in (
        "traceback", "exception", "raises", "raise ", "raised", " error ",
        "typeerror", "valueerror", "keyerror", "attributeerror", "indexerror",
        "runtimeerror", "stack trace", "throws", "thrown",
    )):
        return (
            "TYPE-AWARE CUE: this looks like an exception/error bug. Verify the patch "
            "fixes the ROOT CAUSE at the failing call site (not a try/except suppress) "
            "and that the exception type/message matches what the issue expects.\n\n"
        )
    if any(tok in text for tok in ("remove", "delete", "drop ", " unused", "deprecat")):
        return (
            "TYPE-AWARE CUE: this issue asks to remove/delete code. Verify every "
            "caller, import, and reference of the removed name is also updated; "
            "leftover dangling references will fail the completeness check.\n\n"
        )
    if any(tok in text for tok in ("edge case", "boundary", "off-by-one", "off by one", "overflow", "underflow", "empty list", "empty string", "null", "none case", "zero ", "negative")):
        return (
            "TYPE-AWARE CUE: this issue calls out an edge / boundary condition. "
            "Verify the patch explicitly handles the boundary case named in the issue "
            "(empty / zero / negative / off-by-one / null) — do not just rely on the "
            "happy path.\n\n"
        )
    if any(tok in text for tok in ("test", "assert", "fixture", "pytest", "expected")):
        return (
            "TYPE-AWARE CUE: this issue references tests/assertions. Verify the "
            "companion test was updated (or added) and that asserted values match "
            "the new behaviour exactly.\n\n"
        )
    if any(tok in text for tok in ("move ", "rename", "extract", "split into", "refactor")):
        return (
            "TYPE-AWARE CUE: this issue is a refactor/move. Verify every import "
            "and caller now points at the NEW location and the OLD location no "
            "longer holds the moved code.\n\n"
        )
    return ""


def build_self_check_prompt(
    patch: str,
    issue_text: str,
    inplace_advisories: Optional[List[str]] = None,
) -> str:
    """Show the model its own draft and demand an adversarial self-review.

    Reframed from "validate this patch" to "find the weakest link" — models
    systematically rubber-stamp their own work when asked to confirm
    correctness but surface real flaws when explicitly asked to attack it.
    """
    truncated = (
        patch
        if len(patch) <= 4000
        else patch[:2000] + "\n...[truncated]...\n" + patch[-1500:]
    )
    advisory_block = ""
    if inplace_advisories:
        bullets = "\n  ".join(f"- {a}" for a in inplace_advisories[:3])
        advisory_block = (
            "\nIN-PLACE EDIT WARNINGS (check before finalizing):\n"
            f"  {bullets}\n"
            "If the task is a refactor (not a new-file relocation), fix each by editing "
            "the EXISTING file rather than creating a new one at a different path.\n"
        )
    type_cue = _self_check_type_cue(issue_text)
    return (
        f"{type_cue}"
        "ADVERSARIAL REVIEW. The LLM judge compares your patch to a reference "
        "solution and marks it WRONG if any requirement is missed or the root "
        "cause isn't fixed. Assume the patch is wrong until proven otherwise. "
        "Your job is to find the weakest link BEFORE submitting.\n\n"
        "Task:\n"
        f"{issue_text[:2000]}\n\n"
        "Your patch:\n```diff\n"
        f"{truncated}\n```\n\n"
        "Answer in order — be specific, name lines/symbols, do not give generic answers:\n"
        "1. ROOT CAUSE: does the diff fix the underlying cause, or only suppress a "
        "symptom (try/except, default fallback, early return, value coerce)?\n"
        "2. COMPLETENESS: enumerate every concrete requirement in the task. Which "
        "are NOT obviously addressed by the diff?\n"
        "3. RUNTIME CHECK: would the most relevant test in this repo pass against "
        "this patch? If you have NOT run one, run `pytest tests/test_<module>.py "
        "-x -q` (or the language equivalent for the file you edited) NOW. A "
        "passing test is the strongest correctness evidence.\n"
        "4. SCOPE: any whitespace-only, comment-only, type-annotation-only, "
        "renames, or unrelated refactors the grader will penalize as scope creep?\n"
        f"{advisory_block}\n"
        "If you find ANY weakness in 1-4, emit corrective <command> blocks IN THE "
        "SAME RESPONSE (run missing tests, fix root cause, revert scope-creep), "
        "then end with <final>summary</final>.\n"
        "Only respond `<final>OK</final>` when you have run a relevant test, it "
        "passed, AND you cannot identify a weakness above.\n"
        "Do NOT add new features, destructive operations, or unrelated scope."
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


def build_deletion_nudge_prompt(issue_text: str) -> str:
    """Tell the model it forgot to remove code the issue explicitly requires gone.

    Duel data (round 064855): the issue said remove three old pages; the king
    added the new unified page but left the old pages in place, losing the round.
    The patch had zero deletion lines even though the task demanded removals.
    """
    short = issue_text[:1500] if len(issue_text) > 1500 else issue_text
    return (
        "Deletion gap — the task explicitly requires removing, deleting, or "
        "replacing existing code, but your current patch contains NO deletion "
        "lines.\n\n"
        "Review the task and act on each removal requirement:\n"
        "  - Files, routes, or views that should be deleted outright\n"
        "  - Old implementations that must be replaced (not just augmented)\n"
        "  - Pages, components, or endpoints that should no longer exist\n"
        "  - Hardcoded values, keys, or logic the task says to remove\n\n"
        "Issue the necessary removal commands now (delete statements, remove "
        "files, revert old code), then run a quick verification and emit "
        "<final>summary</final>.\n\n"
        "Task:\n"
        f"{short}\n"
    )


def build_attempt2_bootstrap(result1: Dict[str, Any], n_lines: int) -> str:
    """Inject into attempt 2's first user message so it takes a different path.

    Attempt 2 is blind to what attempt 1 tried — it starts a fresh conversation
    and often repeats the exact same failed approach.  This prefix tells the model
    what went wrong so it actively diverges: reads more files, picks a different
    fix site, uses a different library call, etc.
    """
    steps = result1.get("steps", 0)
    logs_text = result1.get("logs", "") or ""

    reasons: List[str] = []
    if "WALL_CLOCK_STOP" in logs_text:
        reasons.append("ran out of wall-clock time")
    if "MODEL_ERROR_GIVE_UP" in logs_text:
        reasons.append("model errors stopped the loop")
    if n_lines == 0:
        reasons.append("produced an empty patch")
    elif n_lines < 3:
        reasons.append(f"produced only {n_lines} substantive line(s)")
    reason_str = "; ".join(reasons) if reasons else f"produced only {n_lines} substantive line(s)"

    return (
        f"⚠ RETRY ATTEMPT: A prior attempt at this task {reason_str} "
        f"({steps} steps). Do NOT repeat the same approach.\n"
        "Before writing any code: re-read the issue, check which files "
        "you haven't looked at yet, and choose a different fix strategy "
        "if the previous one produced little output.\n\n"
    )


def _recently_observed_paths(logs: List[str], window: int = 80) -> List[str]:
    """Extract file paths recently read by the model from the last `window` log entries.

    Scans for paths surfaced via read_file/cat observations so the mid-loop
    hail-mary prompt can suggest concrete edit targets. Pure Python; no subprocess.
    """
    try:
        path_re = re.compile(r"(?:^|\s|/|')([A-Za-z0-9_.\-/]+\.(?:py|ts|tsx|js|jsx|go|rs|java|kt|cs|cpp|cc|c|h|hpp|php|rb|swift|svelte|md|json|toml|yaml|yml|sh))\b")
        seen: set = set()
        results: List[str] = []
        for entry in logs[-window:]:
            for m in path_re.finditer(entry):
                p = m.group(1).lstrip("/")
                if p and p not in seen and len(p) >= 4:
                    seen.add(p)
                    results.append(p)
                    if len(results) >= 8:
                        return results
        return results
    except Exception:
        return []


def build_soft_nudge_prompt(step: int, elapsed: float) -> str:
    """Mild budget reminder fired once when the model has cycled several steps without committing.

    Unlike build_mid_loop_hail_mary_prompt, this does NOT order the model to stop reading or
    pick a file — it nudges toward commitment without derailing a legitimate plan. Empty when
    the model already has a clear target and will edit naturally.
    """
    return (
        f"BUDGET CHECK: {step} steps in {elapsed:.0f}s with no edits committed yet.\n\n"
        "If your investigation has identified a target file and the fix is clear, "
        "emit edit commands in your next response. "
        "If you genuinely need one more focused read to confirm the target, take it — "
        "but avoid broad searches now. Continue working naturally; this is a reminder, "
        "not a hard stop.\n"
    )


def build_mid_loop_hail_mary_prompt(
    issue_text: str,
    elapsed: float,
    budget: float,
    last_observed_paths: List[str],
) -> str:
    """Emergency prompt fired mid-loop when no edit has been made and >55% of wall-clock is gone.

    Tells the model explicitly: stop reading, pick the most likely target file,
    and emit edit_file commands now.
    """
    pct = int(100 * elapsed / budget) if budget > 0 else 55
    path_hint = ""
    if last_observed_paths:
        path_hint = (
            "\n\nFiles you have already read (most likely candidates for the fix):\n"
            + "".join(f"  - {p}\n" for p in last_observed_paths[:5])
        )
    short_issue = issue_text[:800] if len(issue_text) > 800 else issue_text
    return (
        f"MID-LOOP BUDGET ALERT: {pct}% of wall-clock is gone and no code has been edited yet.\n\n"
        "STOP READING FILES. You must emit edit commands NOW.\n\n"
        "Pick the single most likely file to fix based on the issue and what you have already read. "
        "Use `sed -i`, a python heredoc, or `python - <<'PY' ... PY` to make the smallest "
        "targeted change that addresses the ROOT CAUSE. Do not run broad searches. "
        "If you are still uncertain, make a best-effort minimal edit to the most plausible location "
        "and iterate.\n"
        f"{path_hint}\n"
        "Task (reminder):\n"
        f"{short_issue}\n\n"
        "Emit your edit command(s) now, then run one verification command, then <final>."
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


# -----------------------------
# Main agent
# -----------------------------

# -----------------------------
# v28 multi-shot helpers
# -----------------------------

_MULTISHOT_LOW_SIGNAL_THRESHOLD = 3
# Tau docker_solver hard wall is max(per-task-timeout, 300s) from exec start.
# A 580s outer budget invited "retry" starts with only seconds left, then the
# process was killed mid-attempt -> empty/partial patch (the catastrophic-floor
# failure mode observed in duel #4544). Keep outer budget under ~300s.
_MULTISHOT_TOTAL_BUDGET = 278.0
_MULTISHOT_MIN_ATTEMPT_RESERVE = 52.0
# If attempt 1 already consumed this much wall clock, skip attempt 2 even when
# attempt 1 was low-signal — otherwise the process often dies before the retry
# finishes, which is worse than shipping the first (possibly thin) patch.
_MULTISHOT_MAX_FIRST_ELAPSED = 132.0


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

    Wraps the multi-shot driver so exceptions and late kills return the best
    on-disk patch instead of an avoidable empty result.
    """
    return _solve_with_safety_net(
        repo_path=repo_path, issue=issue, model=model,
        api_base=api_base, api_key=api_key,
        max_steps=max_steps,
        command_timeout=command_timeout,
        max_tokens=max_tokens,
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

        if _n1 >= _MULTISHOT_LOW_SIGNAL_THRESHOLD:
            _result1["multishot_attempts"] = 1
            return _result1

        _elapsed = time.monotonic() - _multishot_started
        if (_MULTISHOT_TOTAL_BUDGET - _elapsed) < _MULTISHOT_MIN_ATTEMPT_RESERVE:
            _result1["multishot_attempts"] = 1
            _result1["multishot_skipped_retry"] = "insufficient_time"
            return _result1

        if _elapsed > _MULTISHOT_MAX_FIRST_ELAPSED:
            # Attempt 1 already burned the outer budget — starting attempt 2
            # invites a docker_solver kill (hard wall ~300s from exec start),
            # which is strictly worse than shipping attempt 1's thin patch.
            _result1["multishot_attempts"] = 1
            _result1["multishot_skipped_retry"] = "first_attempt_used_outer_budget"
            return _result1

        if _multishot_repo_obj is not None:
            _multishot_revert(_multishot_repo_obj, _multishot_initial_head)
        # Pass remaining multishot budget so attempt 2 can't overrun the docker
        # hard wall.  Without this, attempt 2 inherits the full 248 s inner
        # budget even when attempt 1 already consumed 100–130 s, pushing the
        # combined runtime past the ~300 s docker hard wall → process killed,
        # empty patch returned (confirmed timeout in duel #4558 round 064928).
        _remaining = _MULTISHOT_TOTAL_BUDGET - _elapsed
        _attempt2_budget = max(30.0, _remaining - _MULTISHOT_MIN_ATTEMPT_RESERVE)
        _bootstrap = build_attempt2_bootstrap(_result1, _n1)
        _result2 = _solve_attempt(**{**kwargs, "_wall_clock_budget": _attempt2_budget, "_prior_attempt_summary": _bootstrap})
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
    wall_clock_budget = float(kwargs.get("_wall_clock_budget", WALL_CLOCK_BUDGET_SECONDS))
    prior_attempt_summary = kwargs.get("_prior_attempt_summary", "")

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
    mid_loop_hail_mary_used = 0
    soft_nudge_used = 0
    total_refinement_turns_used = 0  # total cap across all refinement gates (hail-mary excluded)
    consecutive_model_errors = 0
    must_edit_after_gap = False
    must_edit_patch = ""
    gap_edit_nudges_used = 0
    deletion_nudges_used = 0
    parallel_nudges_used = 0
    parallel_edits_fanouts_done = 0  # set when Ralph fanout fires (any tier)
    # track turns where the orchestrator emitted commands but the
    # patch didn't change (pure cat/grep/ls exploration). Used to fire the
    # parallel-nudge BEFORE the model commits to a full sequential pass.
    exploration_only_streak = 0
    _prev_patch_for_streak = ""
    pagination_same_file_streak = 0
    last_paginated_path: Optional[str] = None
    pagination_nudges_used = 0
    cat_cache: Dict[str, Any] = {}   # repeat-read short-circuit cache
    solve_started_at = time.monotonic()

    def time_remaining() -> float:
        return wall_clock_budget - (time.monotonic() - solve_started_at)

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
            6. self-check — show the diff and ask "did you cover everything?"
        Each refinement runs at most once per cycle. Test fires AFTER syntax
        (we know the patch parses) but BEFORE coverage/criteria/self-check
        (those are heuristic; test is ground truth from a real runner).
        """
        nonlocal polish_turns_used, self_check_turns_used, syntax_fix_turns_used, test_fix_turns_used, coverage_nudges_used, criteria_nudges_used, hail_mary_turns_used, total_refinement_turns_used, must_edit_after_gap, must_edit_patch, gap_edit_nudges_used, deletion_nudges_used, parallel_nudges_used, parallel_edits_fanouts_done, exploration_only_streak, _prev_patch_for_streak
        patch = get_patch(repo, issue=issue)

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

        # Hard cap on refinement chains: 5-7 refinements would blow the
        # time budget. Hail-mary doesn't count toward this cap.
        if total_refinement_turns_used >= MAX_TOTAL_REFINEMENT_TURNS:
            return False

        # Gate order: syntax → test → deletion → criteria → coverage → polish → self-check
        # Correctness gates (ground-truth or structural) consume refinement budget
        # before cosmetic gates (polish), so we don't waste a capped turn on
        # low-signal hunk cleanup when a real failure is still present.

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

        # Companion-test execution gate. If any edited file has a partner
        # test that actually fails, surface the failure tail to the model
        # as one fix turn. This is the only refinement step in the chain
        # backed by a real runner rather than heuristics.
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

        # Deletion gap: issue says remove/delete/replace but patch has no deletions.
        # Fires before criteria/coverage: a missing removal is a structural omission,
        # not a coverage gap — surface it while refinement budget remains.
        if deletion_nudges_used < MAX_DELETION_NUDGES:
            if _issue_requires_deletion(issue) and not _patch_has_deletions(patch):
                deletion_nudges_used += 1
                total_refinement_turns_used += 1
                must_edit_after_gap = True
                must_edit_patch = patch
                queue_refinement_turn(
                    assistant_text,
                    build_deletion_nudge_prompt(issue),
                    "DELETION_NUDGE_QUEUED: issue requires removal but patch has no deletion lines",
                )
                return True

        # parallel-nudge. Fires in TWO situations:
        # (A) In-progress sequential editing — patch already has ≥3 files,
        # Ralph never fired, more work pending. Switch the remainder
        # to parallel.
        # (B) Exploration-spree — ≥4 consecutive turns with no patch change,
        # ≥3 acceptance criteria, Ralph never fired. Catches the mode where
        # the model cats forever and never edits, runs out of wall-clock.
        # Update the exploration-streak counter first (depends on whether
        # the patch changed since last call).
        if patch == _prev_patch_for_streak:
            exploration_only_streak += 1
        else:
            exploration_only_streak = 0
        _prev_patch_for_streak = patch

        if parallel_nudges_used < MAX_PARALLEL_NUDGES:
            changed_now = _patch_changed_files(patch)
            uncovered = _uncovered_required_paths(patch, issue)
            unaddressed_criteria = _unaddressed_criteria(patch, issue)
            criteria = _extract_acceptance_criteria(issue)
            more_work_needed = bool(uncovered) or bool(unaddressed_criteria)

            trigger_inprogress = (
                len(changed_now) >= 3
                and more_work_needed
            )
            trigger_exploration_spree = (
                exploration_only_streak >= 4
                and len(criteria) >= 3
            )

            if (
                (trigger_inprogress or trigger_exploration_spree)
                and parallel_edits_fanouts_done == 0
                and time_remaining() > (_RALPH_MIN_BUDGET + WALL_CLOCK_RESERVE_SECONDS + 20)
            ):
                parallel_nudges_used += 1
                total_refinement_turns_used += 1
                # Build the "todo" list for the prompt: prefer specific
                # uncovered paths; fall back to criteria summary.
                todo = uncovered if uncovered else [
                    "see issue acceptance criteria — files not yet identified"
                ]
                trigger_label = (
                    "inprogress" if trigger_inprogress else "exploration_spree"
                )
                queue_refinement_turn(
                    assistant_text,
                    build_parallel_nudge_prompt(sorted(changed_now), todo, issue),
                    f"PARALLEL_NUDGE_QUEUED ({trigger_label}):\n  "
                    f"changed={len(changed_now)} explore_streak={exploration_only_streak} "
                    f"uncovered={len(uncovered)} criteria_open={len(unaddressed_criteria)} "
                    f"criteria_total={len(criteria)}",
                )
                return True

        # Criteria-nudge fires before coverage-nudge. Acceptance criteria bullets
        # are directly scored by the LLM judge — addressing them is higher-value
        # than covering additional file paths.
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

        if coverage_nudges_used < MAX_COVERAGE_NUDGES:
            missing = _uncovered_required_paths(patch, issue)
            # Relocation gap: issue says "move/relocate/rebuild as separate"
            # but the patch has no `new file mode` header — fire the
            # coverage nudge with a relocation-specific hint.
            relocation_gap = (
                _issue_implies_relocation(issue)
                and not _patch_creates_any_new_file(patch)
            )
            if missing or relocation_gap:
                coverage_nudges_used += 1
                total_refinement_turns_used += 1
                must_edit_after_gap = True
                must_edit_patch = patch
                marker_paths = ", ".join(missing) if missing else "(no literal paths; relocation-only)"
                marker = (
                    "COVERAGE_NUDGE_QUEUED:\n  " + marker_paths
                    + ("\n  [+relocation-gap]" if relocation_gap else "")
                )
                queue_refinement_turn(
                    assistant_text,
                    build_coverage_nudge_prompt(
                        missing, issue, relocation_gap=relocation_gap,
                        removed_names=_patch_removed_definitions(patch),
                    ),
                    marker,
                )
                return True

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

        if self_check_turns_used < MAX_SELF_CHECK_TURNS:
            self_check_turns_used += 1
            total_refinement_turns_used += 1
            _inplace_adv = _check_inplace_intent(patch, issue, _tracked_set_for_checks)
            queue_refinement_turn(
                assistant_text,
                build_self_check_prompt(patch, issue, inplace_advisories=_inplace_adv),
                "SELF_CHECK_QUEUED",
            )
            return True

        return False

    try:
        repo = _repo_path(repo_path)
        model_name, api_base, api_key = _resolve_inference_config(model, api_base, api_key)
        ensure_git_repo(repo)
        # Lean mode: sub-agent leaves skip preload (brief already carries content).
        # Recursion depth: root=0, children=1; fanout is gated on depth >= 1 below.
        # Hard cap: a grandchild (depth >= 2) is never permitted. _RALPH_MAX_WORKERS
        # (4) × _RALPH_CHILD_MAX_STEPS (10) = 40 chat calls per round upper bound.
        _lean_mode = bool(kwargs.get("_lean_mode", False))
        _recursion_depth = int(kwargs.get("_recursion_depth", 0))
        if _recursion_depth >= 2:
            raise RuntimeError(
                f"recursion-depth invariant broken: _solve_attempt called with "
                f"_recursion_depth={_recursion_depth} (max permitted: 1)"
            )
        if _lean_mode:
            repo_summary = ""
            preloaded_context = ""
            preloaded_files = []
            _tracked_set_for_checks = set()
            _canonical_hints = {}
            _initial_user_content = (
                (prior_attempt_summary if prior_attempt_summary else "")
                + issue
            )
        else:
            repo_summary = get_repo_summary(repo)
            preloaded_context, preloaded_files = build_preloaded_context(repo, issue)
            _tracked_set_for_checks = set(_tracked_files(repo))
            try:
                _canonical_hints = _grep_for_issue_quoted_keys(repo, issue, _tracked_set_for_checks)
            except Exception:
                _canonical_hints = {}
            _canonical_section = ""
            if _canonical_hints:
                _ranked = sorted(_canonical_hints.items(), key=lambda x: -x[1])[:3]
                _bullets = "\n".join(
                    f"  - {path} (contains {n} of the issue-quoted keys)"
                    for path, n in _ranked
                )
                _canonical_section = (
                    "\n\nCANONICAL FILE HINTS — these tracked files ALREADY contain "
                    "keys/identifiers mentioned in the issue:\n\n"
                    f"{_bullets}\n\n"
                    "Edit the keys WHERE THEY ALREADY LIVE. Adding new keys to a "
                    "parallel location like locales/*.json, i18n/*.json, or "
                    "translations/*.{js,ts,json} creates an orphaned duplicate while "
                    "the original definition remains unchanged — callers will "
                    "continue reading the old value.\n"
                )
            _initial_user_content = (
                (prior_attempt_summary if prior_attempt_summary else "")
                + build_initial_user_prompt(issue, repo_summary, preloaded_context)
                + _canonical_section
            )
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _initial_user_content},
        ]
        initial_preload_stripped = False

        _wall_start = time.monotonic()

        for step in range(1, max_steps + 1):
            logs.append(f"\n\n===== STEP {step} =====\n")

            if step > 4 and not initial_preload_stripped and len(messages) >= 2:
                original_initial = messages[1].get("content") or ""
                modified_files = _patch_changed_files(get_patch(repo, issue=issue))
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

            _elapsed_now = time.monotonic() - solve_started_at
            if (
                soft_nudge_used < MAX_SOFT_NUDGE_TURNS
                and step >= _SOFT_NUDGE_STEP_THRESHOLD
                and _elapsed_now >= min(_SOFT_NUDGE_ELAPSED_SECONDS, 0.3 * wall_clock_budget)
                and not get_patch(repo).strip()
                and _elapsed_now < _MID_LOOP_HAIL_MARY_BUDGET_FRACTION * wall_clock_budget
            ):
                soft_nudge_used += 1
                messages.append({
                    "role": "user",
                    "content": build_soft_nudge_prompt(step, _elapsed_now),
                })
                logs.append(f"SOFT_NUDGE_FIRED: step={step} elapsed={_elapsed_now:.1f}s")
                continue

            if (
                mid_loop_hail_mary_used < MAX_MID_LOOP_HAIL_MARY_TURNS
                and _elapsed_now >= _MID_LOOP_HAIL_MARY_BUDGET_FRACTION * wall_clock_budget
                and not get_patch(repo).strip()
            ):
                mid_loop_hail_mary_used += 1
                messages.append({
                    "role": "user",
                    "content": build_mid_loop_hail_mary_prompt(
                        issue, _elapsed_now, wall_clock_budget,
                        _recently_observed_paths(logs),
                    ),
                })
                logs.append("MID_LOOP_HAIL_MARY_FIRED")
                continue

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
                if get_patch(repo, issue=issue).strip():
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

            # Parallel-edits fanout. Suppressed in children (no grandchildren).
            _is_child = _lean_mode or _recursion_depth >= 1
            parallel_edit_specs = (
                [] if _is_child else _extract_parallel_edits(response_text)
            )
            # PE-must-parallelize gate: suppress single-file PE. The Ralph
            # dispatch overhead (separate LLM call per sub-agent, brief
            # assembly, batch wall-clock coordination) is only worth it
            # when at least 2 files run concurrently. Single-file PE is
            # all overhead, no parallelism — sequential is strictly better.
            # Observed degenerate cases involve single-file fanouts that
            # hit malformed bracket markup or land on the wrong file.
            if parallel_edit_specs and len(parallel_edit_specs) < 2:
                logs.append(
                    f"\nPE_SUPPRESSED_NO_PARALLELISM:\n"
                    f"  PE block has {len(parallel_edit_specs)} file — "
                    "no parallelism benefit, stripping for sequential edit."
                )
                only_file = parallel_edit_specs[0][0]
                summary = (
                    f"[parallel_edits_suppressed] PE with a single file "
                    f"({only_file}) is pure overhead. Edit it via a regular "
                    "`<command>` block (sed -i or python3 heredoc) in your "
                    "next response."
                )
                response_text = _PARALLEL_EDITS_BLOCK_RE.sub(
                    lambda _m: summary, response_text, count=1,
                )
                response_text = _PARALLEL_EDITS_BLOCK_RE.sub(
                    lambda _m: "", response_text,
                )
                parallel_edit_specs = []
            if parallel_edit_specs:
                budget_for_ralph = int(time_remaining() - WALL_CLOCK_RESERVE_SECONDS - 5)
                if budget_for_ralph < _RALPH_MIN_BUDGET:
                    logs.append(
                        f"\nRALPH_SKIPPED: only {budget_for_ralph}s remaining; "
                        f"agent should handle the {len(parallel_edit_specs)} edits "
                        f"sequentially via regular <command> blocks."
                    )
                    response_text = _PARALLEL_EDITS_BLOCK_RE.sub(lambda _m: "", response_text)

                if parallel_edit_specs and budget_for_ralph >= _RALPH_MIN_BUDGET:
                    actual_wall_cap = min(_RALPH_BATCH_WALL_CAP, budget_for_ralph)
                    ralph_t0 = time.monotonic()
                    ralph_results = _run_ralph_parallel(
                        parallel_edit_specs,
                        task_text=issue, repo=repo,
                        model=model_name, api_base=api_base, api_key=api_key,
                        batch_wall_cap=actual_wall_cap,
                    )
                    ralph_elapsed = time.monotonic() - ralph_t0
                    done_n = sum(1 for r in ralph_results if r["status"] == "DONE_VERIFIED")
                    blocked_n = sum(1 for r in ralph_results if r["status"] == "BLOCKED")
                    wc_n = sum(1 for r in ralph_results if r["status"] == "WALL_CLOCK")
                    other_n = len(ralph_results) - done_n - blocked_n - wc_n
                    parallel_edits_fanouts_done += 1
                    logs.append(
                        f"\nRALPH_PARALLEL_FANOUT: {len(parallel_edit_specs)} edits requested, "
                        f"{done_n} DONE_VERIFIED, {blocked_n} BLOCKED, {wc_n} WALL_CLOCK, "
                        f"{other_n} other, elapsed={ralph_elapsed:.1f}s (cap={actual_wall_cap}s)"
                    )
                    for r in ralph_results:
                        logs.append(
                            f"  - {r['file_path']} :: {r['status']} ({r['iters']}i) :: "
                            f"{(r.get('summary') or '')[:160]}"
                        )
                        bh = r.get("brief_head")
                        if bh:
                            logs.append(f"      BRIEF_HEAD:")
                            for line in bh.splitlines()[:25]:
                                logs.append(f"        {line[:200]}")
                        for line in (r.get("trace") or [])[:40]:
                            logs.append(f"      {line}")
                    dropped = [
                        fp for fp, _ in parallel_edit_specs[_RALPH_MAX_PER_BATCH:]
                    ]
                    if dropped:
                        logs.append(
                            f"\nPARALLEL_EDITS_OVERFLOW: {len(dropped)} files "
                            f"exceeded per-batch cap and will be surfaced to the "
                            f"model for retry on next turn: "
                            + ", ".join(dropped[:8])
                        )
                    response_text = _replace_parallel_edits_with_summary(
                        response_text, ralph_results, dropped_paths=dropped,
                    )

            # Defense-in-depth: strip HTML comments before extract_commands
            # so any stray <!-- ... <command> ... --> in the response can't be
            # parsed as a real command block.
            response_text = _HTML_COMMENT_RE.sub("", response_text)

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
                patch = get_patch(repo, issue=issue)
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
                # cat-cache short-circuit for repeated reads
                # of unchanged files. Returns None on miss (run normally)
                # or a synthetic CommandResult on hit.
                cached = _maybe_cat_short_circuit(command, repo, cat_cache)
                if cached is not None:
                    result = cached
                else:
                    result = run_command(command, repo, timeout=command_timeout)
                observation = format_observation(result)
                observations.append(f"OBSERVATION {command_index}/{len(command_batch)}:\n{observation}")
                logs.append(f"\nOBSERVATION {command_index}/{len(command_batch)}:\n" + observation)

                if step >= 4 or command_index > 1:
                    patch = get_patch(repo, issue=issue)
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

            if final is not None and get_patch(repo, issue=issue).strip():
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
                if not success and get_patch(repo, issue=issue).strip():
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

            # Fire budget-pressure on every other step while the patch is
            if not get_patch(repo, issue=issue).strip() and step in {2, 4}:
                messages.append({"role": "user", "content": build_budget_pressure_prompt(step)})

            # Pagination-loop detector. If this step issued exactly
            # one command and it was a pure read on the same file as the
            # previous step, count it. After 4 consecutive same-file
            # reads, inject a hard nudge telling the model to stop
            # paginating and emit `<parallel_edits>` for that file.
            # Failure mode: model spends 15-20 sed -n slices on the
            # same Rust file before WALL_CLOCK.
            if len(command_batch) == 1:
                _read_target = _extract_read_target(command_batch[0])
            else:
                _read_target = None
            if _read_target and _read_target == last_paginated_path:
                pagination_same_file_streak += 1
            elif _read_target:
                pagination_same_file_streak = 1
                last_paginated_path = _read_target
            else:
                pagination_same_file_streak = 0
                last_paginated_path = None
            if (
                pagination_same_file_streak >= 4
                and pagination_nudges_used < 1
                and last_paginated_path
            ):
                pagination_nudges_used += 1
                logs.append(
                    f"\nPAGINATION_NUDGE: {pagination_same_file_streak} consecutive "
                    f"reads of {last_paginated_path} — pushing model to write."
                )
                messages.append({"role": "user", "content": (
                    f"STOP PAGINATING. You have read `{last_paginated_path}` "
                    f"{pagination_same_file_streak} times in a row without writing "
                    "anything. Further reads will not produce new information.\n\n"
                    "Your next response MUST be a `<parallel_edits>` block that "
                    f"INCLUDES `{last_paginated_path}` as an `<edit>` target — "
                    "that is the file you have been reading; it IS the work. "
                    "You may include additional related files but you MUST include "
                    f"`{last_paginated_path}`. Use relative paths (no `/work/repo/` "
                    "prefix). Format:\n\n"
                    "<parallel_edits>\n"
                    f"  <edit file=\"{last_paginated_path}\">specific change to make</edit>\n"
                    "  <edit file=\"other/related/file.ext\">specific change to make</edit>\n"
                    "</parallel_edits>\n\n"
                    "Commit to writes based on what you already know from the "
                    "issue and the preloaded snippets. A best-effort edit is "
                    "better than another slice."
                )})
                # Reset so we don't double-fire on the same streak
                pagination_same_file_streak = 0

        # Revert syntactically broken files to HEAD so we don't ship malformed patches.
        try:
            _pre_patch_for_syntax = get_patch(repo, issue=issue)
            _syntax_reverted = _revert_syntactically_broken_files(repo, _pre_patch_for_syntax)
            if _syntax_reverted:
                logs.append(
                    f"\nSYNTAX_REVERT:\nrestored HEAD on {len(_syntax_reverted)} broken file(s): "
                    + ", ".join(_syntax_reverted[:8])
                )
        except Exception as exc:
            logs.append(f"\nSYNTAX_REVERT_ERROR:\n{exc!r}")

        _is_style = _issue_is_style_task(issue)
        if not _is_style:
            try:
                _revert_docker_mount_mode_artifacts(repo)
            except Exception as exc:
                logs.append(f"\nMODE_REVERT_ERROR:\n{exc!r}")
        raw_patch = get_patch(repo, issue=issue)
        patch = raw_patch if _is_style else _strip_mode_only_hunks(raw_patch)
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
                patch = _strip_mode_only_hunks(get_patch(repo, issue=issue))
            except Exception:
                pass

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
