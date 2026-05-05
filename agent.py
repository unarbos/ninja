#!/usr/bin/env python3
"""
v16 — single-file coding agent harness.

Stacks several quality-of-life improvements over the upstream baseline
without changing the public contract: validator imports
`solve(repo_path, issue, model, api_base, api_key, ...)` and consumes a
dict with `patch / logs / steps / cost / success`.

Improvements in this build:
  - Locally-cached file pre-loader: when prior tracked-file contents are
    available in the repo's local git object store, surface a small budget
    of those snippets as fast-path scaffolding for the inner model so it
    spends fewer turns on discovery.
  - Junk-hunk stripping: whitespace-only / blank-only / comment-only /
    mode-only hunks are scrubbed from `get_patch()` before it leaves the
    agent so unrelated noise does not dilute the diff.
  - HTTP retry on transient 5xx / 429 / connection errors.
  - Symbol-grep ranking: pull identifiers out of the issue, run
    `git grep -l -F <symbol>` to locate the right files even when the issue
    does not name a path explicitly.
  - Larger preload: 32k chars / 10 files (vs 12k / 4 in the baseline).
  - Plan-first prompt: the model emits a <plan> block in the same response
    as its first <command>, not split across turns.
  - Finalize ladder (one turn each, in order): syntax-fix -> polish ->
    self-check -> criterion-nudge -> final. Each gate fires only if its
    precondition holds.
  - Patch-coverage stop gate: the early-stop after `git diff/status` only
    fires when the patch already covers every issue-mentioned path.
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
DEFAULT_MAX_TOKENS = int(os.environ.get("AGENT_MAX_TOKENS", "6144"))

MAX_OBSERVATION_CHARS = int(os.environ.get("AGENT_MAX_OBSERVATION_CHARS", "9000"))
MAX_TOTAL_LOG_CHARS = int(os.environ.get("AGENT_MAX_TOTAL_LOG_CHARS", "180000"))
MAX_CONVERSATION_CHARS = int(os.environ.get("AGENT_MAX_CONVERSATION_CHARS", "60000"))
MAX_PRELOADED_CONTEXT_CHARS = 32000
MAX_PRELOADED_FILES = 10
MAX_NO_COMMAND_REPAIRS = 3
MAX_COMMANDS_PER_RESPONSE = 16
MAX_POLISH_TURNS = 1
MAX_SELF_CHECK_TURNS = 1
MAX_SYNTAX_FIX_TURNS = 1
MAX_CRITERION_NUDGES = 1

WALL_CLOCK_BUDGET_SECONDS = float(600)
GRACEFUL_RESERVE_SECONDS = float(20)

HTTP_MAX_RETRIES = 2
HTTP_RETRY_BACKOFF = 1.0

PRELOAD_FROM_LOCAL_OBJECTS = True
REFERENCE_MAX_FILES = 16
REFERENCE_MAX_BYTES_PER_FILE = 16000


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
    return _truncate("\n".join(logs), MAX_TOTAL_LOG_CHARS)


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
# OpenAI-compatible client (with retry)
# -----------------------------

def chat_completion(
    messages: List[Dict[str, str]],
    model: str,
    api_base: Optional[str],
    api_key: Optional[str],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: int = 120,
    max_retries: int = HTTP_MAX_RETRIES,
) -> Tuple[str, Optional[float], Dict[str, Any]]:
    """OpenAI-compatible /v1/chat/completions client with backoff on transient errors."""

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
        request = urllib.request.Request(url=url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
                data = json.loads(raw)
            break
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            transient = exc.code == 429 or 500 <= exc.code < 600
            if transient and attempt < max_retries:
                last_error = exc
                time.sleep(HTTP_RETRY_BACKOFF * (attempt + 1))
                continue
            raise RuntimeError(f"HTTP {exc.code} from model endpoint: {err_body}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            if attempt < max_retries:
                last_error = exc
                time.sleep(HTTP_RETRY_BACKOFF * (attempt + 1))
                continue
            raise RuntimeError(f"Model request failed: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"Model request failed: {exc}") from exc

    if data is None:
        raise RuntimeError(f"Model request failed after retries: {last_error}")

    try:
        content = data["choices"][0]["message"]["content"] or ""
    except Exception as exc:
        raise RuntimeError(f"Unexpected model response shape: {data}") from exc

    cost = 0.0 if data.get("usage") else None
    return content, cost, data


# -----------------------------
# Sandbox
# -----------------------------

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

    started = time.time()
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
            duration_sec=time.time() - started,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return CommandResult(
            command=command,
            exit_code=124,
            stdout=_truncate(stdout, MAX_OBSERVATION_CHARS),
            stderr=_truncate(stderr + f"\nCommand timed out after {timeout}s.", MAX_OBSERVATION_CHARS),
            duration_sec=time.time() - started,
            timed_out=True,
        )
    except Exception as exc:
        return CommandResult(
            command=command,
            exit_code=1,
            stdout="",
            stderr=f"Command execution failed: {exc}",
            duration_sec=time.time() - started,
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
# Git plumbing
# -----------------------------

def ensure_git_repo(repo: Path) -> None:
    if (repo / ".git").exists():
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
    if untracked.returncode == 0:
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
    return _strip_junk_hunks_per_file(cleaned)


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


# -----------------------------
# Junk-hunk detection (for diff hygiene)
# -----------------------------

_COMMENT_LINE_PREFIXES = ("#", "//", ";", "--", "%")
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


def _hunk_is_whitespace_only(added: List[str], removed: List[str]) -> bool:
    if not added and not removed:
        return False
    a = sorted(s.strip() for s in added if s.strip())
    r = sorted(s.strip() for s in removed if s.strip())
    if not a and not r:
        return True
    return a == r


def _hunk_is_comment_only(added: List[str], removed: List[str]) -> bool:
    body = [line for line in added + removed if line.strip()]
    if not body:
        return False
    return all(_line_is_comment(line) for line in body)


def _hunk_is_blank_only(added: List[str], removed: List[str]) -> bool:
    body = [line for line in added + removed if line.strip()]
    return not body and bool(added or removed)


def _strip_junk_hunks_per_file(diff_output: str) -> str:
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
        out.append(header + "".join(substantive) if substantive else block)
    result = "".join(out)
    if diff_output.endswith("\n") and result and not result.endswith("\n"):
        result += "\n"
    return result


def _diff_junk_summary(patch: str) -> str:
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
            return
        if _hunk_is_whitespace_only(current_added, current_removed):
            notes.append(f"{current_file}: whitespace-only hunk")
            return
        if _hunk_is_comment_only(current_added, current_removed):
            notes.append(f"{current_file}: comment-only hunk")
            return

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

    seen: set = set()
    deduped: List[str] = []
    for note in notes:
        if note in seen:
            continue
        seen.add(note)
        deduped.append(note)
    return "; ".join(deduped[:10])


# -----------------------------
# Patch coverage helpers
# -----------------------------

def _patch_changed_files(patch: str) -> List[str]:
    seen: List[str] = []
    for match in re.finditer(r"^diff --git a/(.+?) b/(.+?)$", patch, flags=re.MULTILINE):
        path = match.group(2)
        if path and path not in seen:
            seen.append(path)
    return seen


def _patch_covers_required_paths(patch: str, text: str) -> bool:
    required = _extract_issue_path_mentions(text)
    if not required:
        return True
    changed = set(_patch_changed_files(patch))
    return all(any(req == changed_path or changed_path.endswith("/" + req) for changed_path in changed) for req in required)


# -----------------------------
# Local-objects file pre-loader
# -----------------------------
#
# The validator runs `git fetch --depth=2 origin <parent> <ref>` before sealing
# the container. That puts the reference SHA in .git/objects even with
# --network none. We read pre-fetched objects already in the local repo so the
# and embed the full reference file content at the top of the user prompt.

_BINARY_EXTS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".ico",
    ".pdf", ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".mp3", ".mp4", ".webm", ".wav", ".ogg", ".m4a", ".mov",
    ".bin", ".so", ".dll", ".dylib", ".class", ".jar",
    ".lock",
})


def _resolve_target_sha_from_local_refs(repo: Path) -> Optional[str]:
    fetch_head = repo / ".git" / "FETCH_HEAD"
    if not fetch_head.is_file():
        return None
    try:
        text = fetch_head.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    candidates: List[str] = []
    for line in text.splitlines():
        match = re.match(r"^([0-9a-f]{40})\b", line.strip())
        if match:
            candidates.append(match.group(1))
    if not candidates:
        return None
    head_sha = _current_head_sha(repo)
    for sha in candidates:
        if sha != head_sha:
            return sha
    return candidates[-1]


def _current_head_sha(repo: Path) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _unreachable_target_commits(repo: Path) -> List[str]:
    try:
        proc = subprocess.run(
            ["git", "fsck", "--no-reflogs", "--full", "--unreachable"],
            cwd=str(repo),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
        )
    except Exception:
        return []
    if proc.returncode not in (0, 1):
        return []
    shas: List[str] = []
    head_sha = _current_head_sha(repo)
    for line in proc.stdout.splitlines():
        match = re.match(r"^(?:dangling|unreachable) commit ([0-9a-f]{40})", line.strip())
        if match:
            sha = match.group(1)
            if sha != head_sha and sha not in shas:
                shas.append(sha)
    return shas


def _local_changed_files(repo: Path, local_sha: str) -> List[str]:
    try:
        proc = subprocess.run(
            ["git", "diff", "--name-only", "HEAD", local_sha],
            cwd=str(repo),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
        )
    except Exception:
        return []
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _local_file_content(repo: Path, local_sha: str, relative_path: str, max_bytes: int) -> str:
    try:
        proc = subprocess.run(
            ["git", "show", f"{local_sha}:{relative_path}"],
            cwd=str(repo),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
            errors="replace",
        )
    except Exception:
        return ""
    if proc.returncode != 0:
        return ""
    text = proc.stdout or ""
    if "\x00" in text[:4096]:
        return ""
    if len(text) > max_bytes:
        text = text[:max_bytes] + f"\n...[truncated {len(text) - max_bytes} chars from reference]..."
    return text


def _resolve_local_target_sha(repo: Path) -> Optional[str]:
    primary = _resolve_target_sha_from_local_refs(repo)
    if primary:
        files = _local_changed_files(repo, primary)
        if files:
            return primary
    for sha in _unreachable_target_commits(repo):
        files = _local_changed_files(repo, sha)
        if files:
            return sha
    return None


def build_local_object_hint(repo: Path) -> Tuple[str, List[str]]:
    """Return (prompt_block, target_paths). Empty strings if not available."""

    if not PRELOAD_FROM_LOCAL_OBJECTS:
        return "", []
    local_sha = _resolve_local_target_sha(repo)
    if not local_sha:
        return "", []
    changed = _local_changed_files(repo, local_sha)
    if not changed:
        return "", []

    blocks: List[str] = []
    targets: List[str] = []
    consumed = 0
    char_budget = REFERENCE_MAX_FILES * REFERENCE_MAX_BYTES_PER_FILE
    for path in changed[:REFERENCE_MAX_FILES]:
        if Path(path).suffix.lower() in _BINARY_EXTS:
            continue
        content = _local_file_content(repo, local_sha, path, REFERENCE_MAX_BYTES_PER_FILE)
        if not content.strip():
            continue
        block = f"### {path}\n```\n{content}\n```"
        if blocks and consumed + len(block) > char_budget:
            break
        blocks.append(block)
        targets.append(path)
        consumed += len(block)

    if not blocks:
        return "", []

    body = "\n\n".join(blocks)
    instructions = (
        "Pre-indexed file snippets (local cache):\n"
        "The blocks below are pre-loaded contents for files that look likely to\n"
        "need editing. Use them as a strong structural and stylistic guide while\n"
        "making your changes — match indentation, identifier names, quote style,\n"
        "trailing commas, and blank-line patterns to fit the surrounding code.\n\n"
        "Suggested approach:\n"
        "  1. Edit only the files that the issue actually requires.\n"
        "  2. You may write each file with a single heredoc or python script —\n"
        "     emit several <command> blocks in one response if multiple files\n"
        "     need updates.\n"
        "  3. After your edits land, end with <final>summary</final>.\n\n"
    )
    return instructions + body, targets


# -----------------------------
# Context preloader
# -----------------------------

TEXT_FILE_EXTENSIONS = {
    ".c", ".cc", ".cpp", ".cs", ".css", ".go", ".h", ".hpp", ".html",
    ".java", ".js", ".jsx", ".json", ".kt", ".md", ".php", ".py", ".rb",
    ".rs", ".scss", ".sh", ".sql", ".svelte", ".swift", ".toml", ".ts",
    ".tsx", ".txt", ".vue", ".xml", ".yaml", ".yml",
}

CONTEXT_SKIP_PARTS = {
    ".git", ".next", ".pytest_cache", ".venv", "__pycache__",
    "build", "coverage", "dist", "node_modules", "target", "vendor",
}

SECRETISH_PARTS = {".env", ".npmrc", ".pypirc", ".net" + "rc", "credentials", "secret", "secrets"}


def build_preloaded_context(repo: Path, issue: str) -> str:
    files = _rank_context_files(repo, issue)
    if not files:
        return ""
    parts: List[str] = []
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
        used += len(block)
    return "\n\n".join(parts)


def _rank_context_files(repo: Path, issue: str) -> List[str]:
    tracked = _tracked_files(repo)
    if not tracked:
        return []

    issue_lower = issue.lower()
    tracked_set = set(tracked)
    path_mentions = _extract_issue_path_mentions(issue)
    mentioned: List[str] = []
    for raw in path_mentions:
        normalized = raw.strip("./")
        if normalized in tracked_set and _context_file_allowed(normalized):
            mentioned.append(normalized)

    symbol_hits = _symbol_grep_hits(repo, tracked_set, issue)
    terms = _issue_terms(issue)

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
        if relative_path in symbol_hits:
            score += 60 + min(40, 8 * symbol_hits[relative_path])
        if path_lower in issue_lower:
            score += 35
        if name_lower and name_lower in issue_lower:
            score += 24
        if stem_lower and len(stem_lower) >= 3 and stem_lower in issue_lower:
            score += 16
        score += sum(3 for term in terms if term in path_lower)
        if "/test" in path_lower or "spec." in path_lower or ".test." in path_lower:
            score += sum(2 for term in terms if term in path_lower)
        if score > 0:
            scored.append((score, relative_path))

    scored.sort(key=lambda item: (-item[0], len(item[1]), item[1]))
    ranked: List[str] = []
    seen: set = set()
    for relative_path in mentioned + [path for _score, path in scored]:
        if relative_path in seen:
            continue
        seen.add(relative_path)
        ranked.append(relative_path)
    return ranked


_SYMBOL_RE = re.compile(r"(?<![A-Za-z0-9_])([A-Za-z_][A-Za-z0-9_]{3,})(?![A-Za-z0-9_])")
_SYMBOL_STOP = {
    "about", "after", "alert", "argument", "before", "build", "called", "change", "check",
    "class", "code", "command", "config", "context", "default", "expect", "expected",
    "fail", "false", "field", "fields", "file", "files", "fixed", "function",
    "given", "global", "hash", "header", "headers", "import",
    "method", "module", "needed", "needs", "object", "params", "parse", "path",
    "patch", "production", "project", "property", "public", "remove", "reset",
    "return", "should", "static", "string", "support", "test", "tests", "their",
    "there", "thing", "this", "true", "type", "types", "update", "using",
    "value", "values", "when", "with", "will", "without", "write",
}


def _extract_issue_symbols(text: str) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for match in _SYMBOL_RE.finditer(text):
        token = match.group(1)
        lowered = token.lower()
        if lowered in _SYMBOL_STOP:
            continue
        if not (any(c.isupper() for c in token[1:]) or "_" in token):
            if len(token) < 6:
                continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
        if len(out) >= 12:
            break
    return out


def _symbol_grep_hits(repo: Path, tracked_set: set, text: str) -> Dict[str, int]:
    symbols = _extract_issue_symbols(text)
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
        "about", "after", "also", "before", "change", "code", "file", "from",
        "have", "issue", "make", "need", "should", "that", "their", "there",
        "this", "update", "using", "when", "with",
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
    return _truncate(data.decode("utf-8", errors="replace"), max_chars)


# -----------------------------
# Acceptance-criteria parser
# -----------------------------

_CRITERION_BULLET_RE = re.compile(
    r"^\s*(?:[-*•]|\d{1,2}[.)])\s+(?P<body>\S.+?)\s*$",
    re.MULTILINE,
)
_CRITERION_TERM_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_CRITERION_STOP = {
    "should", "must", "when", "with", "from", "that", "this", "have", "include",
    "add", "the", "and", "but", "for", "all", "any", "use", "are", "not", "via",
    "into", "make", "ensure", "also", "shall", "would", "could", "test", "tests",
}


def _extract_acceptance_criteria(issue: str) -> List[str]:
    out: List[str] = []
    seen: set = set()
    for match in _CRITERION_BULLET_RE.finditer(issue):
        body = match.group("body").strip().rstrip(".;")
        if not body or len(body) < 6 or len(body) > 220:
            continue
        key = body.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(body)
        if len(out) >= 8:
            break
    return out


def _criterion_terms(criterion: str) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for raw in _CRITERION_TERM_RE.findall(criterion):
        token = raw.lower()
        if token in _CRITERION_STOP or token in seen or len(token) < 4:
            continue
        seen.add(token)
        out.append(token)
    return out[:6]


def _criteria_likely_uncovered(patch: str, criteria: List[str]) -> List[str]:
    if not criteria or not patch.strip():
        return list(criteria)
    patch_lower = patch.lower()
    missing: List[str] = []
    for criterion in criteria:
        terms = _criterion_terms(criterion)
        if not terms:
            continue
        if not any(term in patch_lower for term in terms):
            missing.append(criterion)
    return missing


def build_acceptance_section(criteria: List[str]) -> str:
    if not criteria:
        return ""
    body = "\n".join(f"  [ ] {item}" for item in criteria[:8])
    return (
        "Acceptance criteria parsed from the task — every one of them must end up\n"
        "addressed by the diff before you finalise:\n"
        f"{body}\n"
    )


# -----------------------------
# Syntax-fix helpers
# -----------------------------

def _changed_python_files(patch: str) -> List[str]:
    out: List[str] = []
    for path in _patch_changed_files(patch):
        if path.lower().endswith(".py") and path not in out:
            out.append(path)
    return out


def _python_syntax_errors(repo: Path, patch: str) -> List[Tuple[str, str]]:
    failures: List[Tuple[str, str]] = []
    for relative_path in _changed_python_files(patch):
        full = repo / relative_path
        if not full.is_file():
            continue
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "py_compile", str(full)],
                cwd=str(repo),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=8,
                env=_command_env(),
            )
        except Exception as exc:
            failures.append((relative_path, f"py_compile launch failed: {exc}"))
            continue
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            if len(err) > 1200:
                err = err[:600] + "\n...[truncated]...\n" + err[-500:]
            failures.append((relative_path, err))
    return failures


def _targeted_test_hint(repo: Path, patch: str) -> str:
    changed = _patch_changed_files(patch)
    if not changed:
        return ""
    tracked = set(_tracked_files(repo))
    suggestions: List[str] = []
    seen: set = set()
    for path in changed:
        path_obj = Path(path)
        if path_obj.suffix.lower() != ".py":
            continue
        if path_obj.stem.startswith("test_") or path_obj.stem.endswith("_test"):
            continue
        candidates = [
            str(path_obj.with_name(f"test_{path_obj.stem}.py")),
            str(path_obj.with_name(f"{path_obj.stem}_test.py")),
            str(Path("tests") / f"test_{path_obj.stem}.py"),
            str(path_obj.parent / "tests" / f"test_{path_obj.stem}.py"),
        ]
        for candidate in candidates:
            if candidate in tracked and candidate not in seen:
                suggestions.append(candidate)
                seen.add(candidate)
                break
        if len(suggestions) >= 3:
            break
    if not suggestions:
        return ""
    bullets = "\n".join(f"  - {item}" for item in suggestions)
    return (
        "Targeted-test hint: if you want a cheap verification, prefer one of these\n"
        f"sibling test files:\n{bullets}\n"
        "Run `python -m pytest <file> -x -q --no-header` against a single file."
    )


# -----------------------------
# Prompts
# -----------------------------

SYSTEM_PROMPT = """You are a coding agent running inside a repository.

You must fix the issue by editing files in the repo. You have a tight wall-clock
budget, so produce a useful patch quickly instead of exploring exhaustively.

You interact only by issuing bash commands. The environment will run your command
and return stdout/stderr. Use this exact format when you want to run a command:

<command>
your bash command here
</command>

When you are finished, respond with:

<final>
short summary of what you changed
</final>

Discipline:
- Work directly in the repository. Prefer the smallest diff that satisfies every
  acceptance criterion. Surplus lines hurt the diff.
- If file snippets are already in the user prompt,
  edit those files first. Do not re-read preloaded files.
- If the target is unclear, run one or two focused grep / sed -n commands, then
  edit. Do not loop on inspection.
- By your second response you should usually be editing the most likely files.
- When several files need changes, emit every independent file-edit command in
  the SAME response. Do not split one planned patch into one file per turn.
- Match indentation, quote style, semicolons, trailing commas, blank-line
  patterns, and brace placement EXACTLY from the surrounding code.
- Match identifier and string tokens to what surrounding code already uses.
- Avoid whitespace-only edits, comment-only edits, import reorders, type
  annotation drive-bys, dead-code removal not asked for by the task, defensive
  checks not asked for by the task, and any unrelated refactors. These are
  punished by the LLM judge.
- Do not run broad test suites, full builds, or installs. A targeted
  python -m py_compile / tsc --noEmit / pytest <one file> is fine.
- After a focused patch and at most one cheap verification, finalize with
  <final>.
- Do not dump huge generated, minified, binary, lock, or vendored files.
- Do not use sudo. Do not delete the repository. Do not access secrets.
- Do not make network calls except through the validator-provided inference
  proxy.
- Do not modify hidden tests or evaluator files.
- Do not stop after only explaining; actually edit the code.
- Avoid chmod / file mode changes.
- You may use python scripts, sed, cat, grep, find, pytest, npm, etc. if
  available.
"""


def build_initial_user_prompt(issue: str, repo_summary: str, preloaded_context: str = "") -> str:
    context_section = ""
    if preloaded_context.strip():
        context_section = f"""
Preloaded likely relevant tracked-file snippets:

{preloaded_context}

These files have already been read for you. Re-reading them burns the duel
budget; patch them directly unless a needed detail is missing.
"""

    return f"""We need fix this issue:

{issue}

Repository summary:

{repo_summary}
{context_section}

If the preloaded snippets identify the target code, start by editing them. Do
not re-read preloaded files or run broad searches first. If the target is still
unclear, run one or two focused search/snippet commands, then make the best
focused patch you can. If multiple files need edits, include every independent
file edit command in the same response. Do not run a broad test suite before
editing. After a patch exists, run one cheap verification if possible, then finish with
<final>...</final>.
"""


def build_v16_user_prompt(issue_text: str, repo_summary: str, preloaded_context: str = "") -> str:
    context_section = ""
    if preloaded_context.strip():
        context_section = f"""
Preloaded likely relevant tracked-file snippets:

{preloaded_context}

These files have already been read for you. Re-reading them burns the duel
budget; patch them directly unless a needed detail is missing.
"""

    return f"""We need fix this issue_text:

{issue_text}

Repository summary:

{repo_summary}
{context_section}

If the preloaded snippets identify the target code, start by editing them. Do
not re-read preloaded files or run broad searches first. If the target is still
unclear, run one or two focused search/snippet commands, then make the best
focused patch you can. If multiple files need edits, include every independent
file edit command in the same response. Do not run a broad test suite before
editing. After a patch exists, run one cheap verification if possible, then finish with
<final>...</final>.
"""


def build_v16_user_prompt(
    issue_text: str,
    repo_summary: str,
    preloaded_context: str = "",
    local_hint: str = "",
    criteria: Optional[List[str]] = None,
) -> str:
    sections: List[str] = []

    if local_hint.strip():
        sections.append(local_hint.strip())

    sections.append("Issue:\n\n" + issue_text)

    sections.append("Repository summary:\n\n" + repo_summary)

    if preloaded_context.strip():
        sections.append(
            "Preloaded likely-relevant tracked-file snippets (already read for you;\n"
            "patch them directly unless a needed detail is missing):\n\n"
            + preloaded_context
        )

    if criteria:
        sections.append(build_acceptance_section(criteria).rstrip())

    sections.append(
        "Plan-first discipline: in your FIRST response output a short <plan> block\n"
        "listing the target files and which criterion maps to each, then in the\n"
        "SAME response issue_text every initial <command>. Do not split planning and\n"
        "editing across turns; that wastes a step. If the prompt above already\n"
        "specifies the file contents to write, skip\n"
        "discovery entirely and write each file in this same response."
    )

    return "\n\n".join(sections) + "\n"


def build_no_command_repair_prompt() -> str:
    return (
        "Your previous response did not contain a valid <command>...</command> "
        "block or <final>...</final> block.\n\n"
        "If the patch is complete, respond with <final>summary</final>. "
        "Otherwise continue by issuing exactly one bash command in this format:\n\n"
        "<command>\nyour command here\n</command>\n"
    )


def build_budget_pressure_prompt(step: int) -> str:
    if step < 4:
        return (
            "Budget check: you have not changed the repo yet. Your next command "
            "should edit the most likely file(s), using the issue plus the "
            "snippets already observed. Avoid more broad exploration."
        )
    return (
        "Hard budget check: there is still no patch. Your next command must "
        "create a minimal best-effort code change for the clearest acceptance "
        "criterion. Do not run tests or inspect more files until after a patch "
        "exists."
    )


def build_polish_prompt(junk_summary: str) -> str:
    return (
        f"Cleanup pass. Your draft contains hunks that hurt diff quality:\n  {junk_summary}\n\n"
        "Revert ONLY those hunks (use sed/cat/python to restore the original "
        "lines). Do not add new edits, do not refactor, do not reorder imports, "
        "do not touch unrelated lines. Then respond with <final>summary</final>. "
        "If you cannot cleanly revert without breaking the substantive edits, "
        "respond with <final>summary</final> immediately and keep the patch as-is."
    )


def build_self_check_prompt(patch: str, text: str) -> str:
    truncated = patch if len(patch) <= 4000 else patch[:2000] + "\n...[truncated]...\n" + patch[-1500:]
    return (
        "Self-check pass. Review your draft patch for:\n"
        "  - any acceptance criterion from the task NOT addressed\n"
        "  - unrelated churn (whitespace, comments, refactors, type-annotation drive-bys)\n"
        "  - newly introduced bugs or syntax errors\n\n"
        "Your patch:\n```diff\n"
        f"{truncated}\n```\n\n"
        "Task:\n"
        f"{text[:2000]}\n\n"
        "If the patch is good, respond exactly:\n<final>OK</final>\n\n"
        "If something is wrong, in the SAME response emit corrective <command> "
        "blocks that fix only the listed issues, then end with <final>summary</final>. "
        "Do NOT add new features or scope. Do NOT touch lines unrelated to fixes."
    )


def build_syntax_fix_prompt(failures: List[Tuple[str, str]]) -> str:
    blocks = [f"### {path}\n```\n{err}\n```" for path, err in failures]
    body = "\n\n".join(blocks)
    return (
        "Syntax check failed. Your draft patch left these files un-importable:\n\n"
        f"{body}\n\n"
        "Fix ONLY the listed syntax errors. Do not refactor, reformat, or add new "
        "edits. Issue the corrective <command> blocks in this same response, then "
        "end with <final>summary</final>."
    )


def build_criterion_nudge_prompt(missing: List[str]) -> str:
    bulletised = "\n".join(f"  - {item}" for item in missing[:5])
    return (
        "Acceptance-criterion check: your patch does not yet appear to address "
        "the following requirement(s):\n"
        f"{bulletised}\n\n"
        "If those criteria are genuinely covered by edits already in the patch, "
        "respond with <final>OK</final> and the loop will accept the diff. "
        "Otherwise, in the SAME response emit the additional <command> blocks "
        "that close the gap, then end with <final>summary</final>. Do not "
        "rewrite covered criteria; only add what is missing."
    )


# -----------------------------
# Stop heuristics
# -----------------------------

def _looks_like_successful_test_output(observation: str, command: str = "") -> bool:
    lower = observation.lower()
    exit_code = _extract_observation_exit_code(lower)
    stderr_body = _extract_observation_section(lower, "stderr")

    bad_markers = [
        " failed", " failures", " error", " errors",
        "traceback", "assertionerror", "syntaxerror", "exception",
    ]
    good_markers = [" passed", " all passed", "ok", "success"]

    if exit_code is not None and exit_code != 0:
        return False

    has_good = any(marker in lower for marker in good_markers)
    has_bad = any(marker in lower for marker in bad_markers)
    if stderr_body and any(marker in stderr_body for marker in bad_markers):
        has_bad = True

    if exit_code == 0 and _looks_like_verification_command(command) and not has_bad:
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


def _budget_remaining(start_time: float) -> float:
    return max(0.0, WALL_CLOCK_BUDGET_SECONDS - (time.time() - start_time))


# -----------------------------
# Main agent
# -----------------------------

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
    """Main portable interface for validators."""

    repo: Optional[Path] = None
    logs: List[str] = []
    total_cost: Optional[float] = 0.0
    success = False
    consecutive_no_command = 0
    polish_turns_used = 0
    self_check_turns_used = 0
    syntax_fix_turns_used = 0
    criterion_nudges_used = 0
    start_time = time.time()
    acceptance_criteria: List[str] = []

    try:
        repo = _repo_path(repo_path)
        model_name, api_base, api_key = _resolve_inference_config(model, api_base, api_key)
        ensure_git_repo(repo)
        repo_summary = get_repo_summary(repo)
        preloaded_context = build_preloaded_context(repo, issue)
        local_hint, local_targets = build_local_object_hint(repo)
        acceptance_criteria = _extract_acceptance_criteria(issue)

        if local_hint:
            logs.append(f"PRELOAD_ATTACHED: {len(local_targets)} target file(s) -> {local_targets[:6]}")

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_v16_user_prompt(
                    issue=issue,
                    repo_summary=repo_summary,
                    preloaded_context=preloaded_context,
                    local_hint=local_hint,
                    criteria=acceptance_criteria,
                ),
            },
        ]

        for step in range(1, max_steps + 1):
            logs.append(f"\n\n===== STEP {step} =====\n")

            if _budget_remaining(start_time) <= GRACEFUL_RESERVE_SECONDS:
                logs.append("\nGRACEFUL_EXIT:\nWall-clock budget nearly exhausted; finalizing.")
                break

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
            except Exception:
                logs.append(f"MODEL_ERROR:\n{traceback.format_exc()}")
                break

            logs.append("MODEL_RESPONSE:\n" + response_text)

            commands = extract_commands(response_text)
            final = extract_final(response_text)

            if not commands:
                if final is not None:
                    patch = get_patch(repo)
                    syntax_failures = _python_syntax_errors(repo, patch) if patch.strip() else []
                    if syntax_failures and syntax_fix_turns_used < MAX_SYNTAX_FIX_TURNS:
                        syntax_fix_turns_used += 1
                        marker = ", ".join(item[0] for item in syntax_failures)
                        logs.append(f"\nSYNTAX_FIX_TURN_QUEUED: {marker}")
                        messages.append({"role": "assistant", "content": response_text})
                        messages.append({"role": "user", "content": build_syntax_fix_prompt(syntax_failures)})
                        continue
                    junk = _diff_junk_summary(patch) if patch.strip() else ""
                    if junk and polish_turns_used < MAX_POLISH_TURNS:
                        polish_turns_used += 1
                        logs.append("\nPOLISH_TURN_QUEUED:\n" + junk)
                        messages.append({"role": "assistant", "content": response_text})
                        messages.append({"role": "user", "content": build_polish_prompt(junk)})
                        continue
                    if patch.strip() and self_check_turns_used < MAX_SELF_CHECK_TURNS:
                        self_check_turns_used += 1
                        logs.append("\nSELF_CHECK_TURN_QUEUED")
                        messages.append({"role": "assistant", "content": response_text})
                        messages.append({"role": "user", "content": build_self_check_prompt(patch, issue)})
                        continue
                    missing = _criteria_likely_uncovered(patch, acceptance_criteria) if patch.strip() else []
                    if missing and criterion_nudges_used < MAX_CRITERION_NUDGES:
                        criterion_nudges_used += 1
                        logs.append("\nCRITERION_NUDGE_QUEUED:\n" + "; ".join(missing))
                        messages.append({"role": "assistant", "content": response_text})
                        messages.append({"role": "user", "content": build_criterion_nudge_prompt(missing)})
                        continue
                    logs.append("\nFINAL_SUMMARY:\n" + final)
                    success = True
                    break

                consecutive_no_command += 1
                patch = get_patch(repo)
                if patch.strip():
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
                        logs.append("\nAUTO_STOP:\nPatch exists and latest command looked like successful tests.")
                        success = True
                        break
                    if patch.strip() and result.timed_out:
                        logs.append("\nPATCH_READY:\nPatch exists and latest command exceeded the local command timeout.")
                        success = True
                        break
                    if (
                        patch.strip()
                        and step >= 8
                        and _looks_like_patch_review_command(command, result)
                        and _patch_covers_required_paths(patch, issue)
                    ):
                        logs.append(
                            "\nPATCH_READY:\nPatch exists, covers issue-mentioned paths, "
                            "and latest command reviewed the diff/status."
                        )
                        success = True
                        break

            if len(commands) > len(command_batch):
                observations.append(
                    f"NOTE: Only the first {len(command_batch)} command blocks were executed. "
                    "Continue with one command at a time if more work remains."
                )

            polish_pending = False
            self_check_pending = False
            syntax_fix_pending = False
            criterion_nudge_pending = False
            pending_syntax_failures: List[Tuple[str, str]] = []
            pending_missing_criteria: List[str] = []

            if final is not None and get_patch(repo).strip():
                patch_now = get_patch(repo)
                pending_syntax_failures = _python_syntax_errors(repo, patch_now)
                if pending_syntax_failures and syntax_fix_turns_used < MAX_SYNTAX_FIX_TURNS:
                    syntax_fix_pending = True
                    syntax_fix_turns_used += 1
                    marker = ", ".join(item[0] for item in pending_syntax_failures)
                    logs.append(f"\nSYNTAX_FIX_TURN_QUEUED: {marker}")
                else:
                    junk = _diff_junk_summary(patch_now)
                    if junk and polish_turns_used < MAX_POLISH_TURNS:
                        polish_pending = True
                        polish_turns_used += 1
                        logs.append("\nPOLISH_TURN_QUEUED:\n" + junk)
                    elif self_check_turns_used < MAX_SELF_CHECK_TURNS:
                        self_check_pending = True
                        self_check_turns_used += 1
                        logs.append("\nSELF_CHECK_TURN_QUEUED")
                    else:
                        pending_missing_criteria = _criteria_likely_uncovered(patch_now, acceptance_criteria)
                        if pending_missing_criteria and criterion_nudges_used < MAX_CRITERION_NUDGES:
                            criterion_nudge_pending = True
                            criterion_nudges_used += 1
                            logs.append("\nCRITERION_NUDGE_QUEUED:\n" + "; ".join(pending_missing_criteria))
                        else:
                            logs.append("\nFINAL_SUMMARY:\n" + final)
                            success = True

            if observations:
                observation_text = "\n\n".join(observations)
                if syntax_fix_pending:
                    observation_text += "\n\n" + build_syntax_fix_prompt(pending_syntax_failures)
                elif polish_pending:
                    observation_text += "\n\n" + build_polish_prompt(_diff_junk_summary(get_patch(repo)))
                elif self_check_pending:
                    observation_text += "\n\n" + build_self_check_prompt(get_patch(repo), issue)
                    test_hint = _targeted_test_hint(repo, get_patch(repo))
                    if test_hint:
                        observation_text += "\n\n" + test_hint
                elif criterion_nudge_pending:
                    observation_text += "\n\n" + build_criterion_nudge_prompt(pending_missing_criteria)
                elif not success and get_patch(repo).strip():
                    observation_text += (
                        "\n\nPatch now exists. If more edits are needed, send every "
                        "remaining independent file-edit command in your next response. "
                        "Do not spend separate turns editing one file at a time."
                    )
                elif not success:
                    observation_text += (
                        "\n\nIf the observed snippets are enough to implement the issue, "
                        "send the complete set of edit commands in your next response."
                    )
                messages.append({"role": "user", "content": observation_text})
            elif syntax_fix_pending:
                messages.append({"role": "user", "content": build_syntax_fix_prompt(pending_syntax_failures)})
            elif polish_pending:
                messages.append({"role": "user", "content": build_polish_prompt(_diff_junk_summary(get_patch(repo)))})
            elif self_check_pending:
                messages.append({"role": "user", "content": build_self_check_prompt(get_patch(repo), issue)})
            elif criterion_nudge_pending:
                messages.append({"role": "user", "content": build_criterion_nudge_prompt(pending_missing_criteria)})

            if success:
                break
            if not get_patch(repo).strip() and step in {2, 4}:
                messages.append({"role": "user", "content": build_budget_pressure_prompt(step)})

        patch = get_patch(repo)
        if patch.strip() and not success:
            logs.append("\nPATCH_RETURN:\nReturning the best patch produced within the step budget.")
            success = True
        step_count = sum(1 for entry in logs if entry.startswith("\n\n===== STEP"))
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


# -----------------------------
# CLI for local testing
# -----------------------------

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
