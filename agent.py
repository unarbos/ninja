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
MAX_PRELOADED_CONTEXT_CHARS = 60000
MAX_PRELOADED_FILES = 10
MAX_NO_COMMAND_REPAIRS = 3
MAX_COMMANDS_PER_RESPONSE = 12
MAX_POLISH_TURNS = 1
MAX_SELF_CHECK_TURNS = 0   # disabled — v2 data showed it causes over-editing
MAX_SYNTAX_FIX_TURNS = 1
MAX_PATCH_BLOCK_BYTES = 64000  # cap on a single <patch> block; rejects pathological inputs


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


def chat_completion(
    messages: List[Dict[str, str]],
    model: str,
    api_base: Optional[str],
    api_key: Optional[str],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: int = 120,
    max_retries: int = 1,
) -> Tuple[str, Optional[float], Dict[str, Any]]:
    """OpenAI-compatible /v1/chat/completions client. Retries once on transient
    transport failures (timeouts, connection errors, HTTP 5xx)."""

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


ACTION_RE = re.compile(r"<command>\s*(.*?)\s*</command>", re.IGNORECASE | re.DOTALL)
FINAL_RE = re.compile(r"<final>\s*(.*?)\s*</final>", re.IGNORECASE | re.DOTALL)
# Direct unified-diff action — model emits a complete patch in one block.
# Validated with `git apply --check` before apply, so malformed diffs
# fail fast instead of corrupting the working tree.
PATCH_BLOCK_RE = re.compile(r"<patch>\s*(.*?)\s*</patch>", re.IGNORECASE | re.DOTALL)


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


def extract_patches(model_text: str) -> List[str]:
    """Pull every <patch>...</patch> block; tolerate ```diff fencing inside."""
    out: List[str] = []
    for match in PATCH_BLOCK_RE.finditer(model_text):
        body = match.group(1).strip()
        # Some models wrap the diff in ```diff ... ``` even inside <patch>.
        body = re.sub(r"^```(?:diff|patch)?\s*\n", "", body, count=1)
        body = re.sub(r"\n```\s*$", "", body, count=1).strip()
        if body and len(body.encode("utf-8")) <= MAX_PATCH_BLOCK_BYTES:
            out.append(body)
    return out


@dataclass
class PatchApplyResult:
    succeeded: bool
    method: str            # "git_apply" | "git_apply_3way" | "rejected"
    error: str = ""
    files_touched: int = 0
    lines_added: int = 0
    lines_removed: int = 0


@dataclass
class DiffStyle:
    """Quick structural metrics for the working-tree diff. Fed back to the
    model as a self-review signal so it can pull back if the patch is
    growing larger than the bug at hand requires."""
    files: int = 0
    hunks: int = 0
    lines_added: int = 0
    lines_removed: int = 0
    plus_minus_ratio: float = 0.0   # >>1 means net-additive (often bad), <<1 net-removing (often bad too)


def _compute_diff_style(patch: str) -> DiffStyle:
    if not patch.strip():
        return DiffStyle()
    files = sum(1 for line in patch.splitlines() if line.startswith("diff --git "))
    hunks = sum(1 for line in patch.splitlines() if line.startswith("@@"))
    added = sum(1 for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in patch.splitlines() if line.startswith("-") and not line.startswith("---"))
    ratio = (added / removed) if removed else (float(added) if added else 0.0)
    return DiffStyle(files=files, hunks=hunks, lines_added=added, lines_removed=removed, plus_minus_ratio=ratio)


def _format_diff_style_observation(style: DiffStyle) -> str:
    """Human-readable diff metrics. Includes a guidance flag when the diff
    is touching several files or has grown above ~60 changed lines, both
    of which are review-burden signals for a focused bug fix."""
    if style.files == 0:
        return "DIFF_STYLE: no patch yet."

    flag = ""
    if style.files > 3:
        flag = " (touches several files — verify each is needed for the bug)"
    elif style.lines_added + style.lines_removed > 60:
        flag = " (>60 changed lines — verify the change scope is necessary)"

    return (
        f"DIFF_STYLE: {style.files} file(s), {style.hunks} hunk(s), "
        f"+{style.lines_added}/-{style.lines_removed}{flag}"
    )


def apply_unified_diff(diff_text: str, repo: Path) -> PatchApplyResult:
    """Apply a model-emitted unified diff via `git apply`, with a few graceful fallbacks.

    Try strategies in order:
      1. `git apply --whitespace=fix` — strict, catches malformed diffs early
      2. `git apply --3way` — falls back when context lines drift
      3. give up

    Returns a structured result so the caller can surface errors back to the
    model without catastrophically failing the loop.
    """
    if not diff_text.strip():
        return PatchApplyResult(succeeded=False, method="rejected", error="empty diff body")

    # Normalize trailing newline — `git apply` is picky.
    body = diff_text if diff_text.endswith("\n") else diff_text + "\n"
    diff_path = repo / ".pilot_pending.diff"
    diff_path.write_text(body, encoding="utf-8")

    flags_to_try = [
        ["--whitespace=fix"],
        ["--whitespace=fix", "--3way"],
        ["--whitespace=fix", "-p0"],
    ]
    last_error = ""
    for flags in flags_to_try:
        # Pre-flight check first so we know the diff parses.
        check = subprocess.run(
            ["git", "apply", "--check", *flags, str(diff_path)],
            cwd=str(repo),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
        )
        if check.returncode != 0:
            last_error = check.stderr.strip() or check.stdout.strip()
            continue

        proc = subprocess.run(
            ["git", "apply", *flags, str(diff_path)],
            cwd=str(repo),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
        if proc.returncode == 0:
            try:
                diff_path.unlink()
            except Exception:
                pass
            files_touched = sum(1 for line in body.splitlines() if line.startswith("diff --git "))
            lines_added = sum(1 for line in body.splitlines() if line.startswith("+") and not line.startswith("+++"))
            lines_removed = sum(1 for line in body.splitlines() if line.startswith("-") and not line.startswith("---"))
            return PatchApplyResult(
                succeeded=True,
                method="git_apply" if "--3way" not in flags else "git_apply_3way",
                files_touched=files_touched,
                lines_added=lines_added,
                lines_removed=lines_removed,
            )
        last_error = proc.stderr.strip() or proc.stdout.strip()

    try:
        diff_path.unlink()
    except Exception:
        pass
    return PatchApplyResult(succeeded=False, method="rejected", error=last_error[:600])


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
    return _strip_junk_hunks_per_file(cleaned)


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
    files = _rank_context_files(repo, issue)
    if not files:
        return ""

    parts: List[str] = []
    used = 0
    per_file_budget = max(6000, MAX_PRELOADED_CONTEXT_CHARS // max(1, min(len(files), MAX_PRELOADED_FILES)))

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
    path_mentions = _extract_issue_path_mentions(issue)
    mentioned: List[str] = []
    tracked_set = set(tracked)
    for mention in path_mentions:
        normalized = mention.strip("./")
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
    seen: set[str] = set()
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
    seen: set[str] = set()
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
        if len(out) >= 10:
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
    return all(any(req == c or c.endswith("/" + req) for c in changed) for req in required)


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
            parts = line.split()
            if len(parts) >= 4 and parts[3].startswith("b/"):
                current_file = parts[3][2:]
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


def build_polish_prompt(junk_summary: str) -> str:
    return (
        f"Cleanup pass. Your draft contains hunks that hurt diff quality:\n  {junk_summary}\n\n"
        "Revert ONLY those hunks (use sed/cat/python to restore the original "
        "lines). Do not add new edits, do not refactor, do not reorder imports, "
        "do not touch unrelated lines. Then respond with <final>summary</final>. "
        "If you cannot cleanly revert without breaking the substantive edits, "
        "respond with <final>summary</final> immediately and keep the patch as-is."
    )


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


SYSTEM_PROMPT = """You are a coding agent running inside a repository.

Your goal is to produce a unified diff that fixes the issue. Good engineering
practice for bug fixes means: change as little as possible, keep the change
targeted, match the surrounding code's style. Every extra line you add or remove
that isn't required by the issue risks introducing regressions in code that
wasn't asked to change.

You have THREE action types. Use the right tool for the job:

  <command>bash command</command>   — for exploration (grep, sed -n, ls)
  <patch>unified diff</patch>       — for the FIX itself (preferred)
  <final>summary</final>            — when done

The <patch> action takes a complete unified diff and applies it via
`git apply` (with 3-way fallback). Use <patch> for the actual fix because:
  1. Producing a unified diff in one shot avoids whitespace and indentation
     errors that occur when stitching together multiple sed/cat commands.
  2. One <patch> block + one <final> = the entire fix in 1-2 turns. No
     multi-step bash drift.
  3. Multi-file fixes go in a single <patch> block with one `diff --git`
     header per file — keeps all changes together.

Plan-first protocol: in your FIRST response, output a <plan> block, then
either a <patch> block (if you can write the fix from preloaded context) or
one <command> (if you need to inspect one more file). Do not spend a turn on
plan + exploration alone.

Use <command> ONLY when:
  - the preloaded files are insufficient and you need to look at one more
  - you genuinely need to RUN a verification (python -m py_compile,
    targeted pytest)

For everything else, use <patch>. After a <patch> applies cleanly and (if
relevant) one verification passes, finalize with <final>.

Correctness examples — fixes that address the issue without touching unrelated
code. These minimize risk of regressions:

EXAMPLE A (off-by-one) ↓
```diff
--- a/stream.py
+++ b/stream.py
@@ def pairs(seq):
     seq = list(seq)
-    for i in range(len(seq) - 2):
+    for i in range(len(seq) - 1):
         yield (seq[i], seq[i + 1])
```

EXAMPLE B (missing default for nested-dict lookup) ↓
```diff
--- a/pricing.py
+++ b/pricing.py
@@ def lookup_price(item, currency):
-    return PRICES[item][currency]
+    return PRICES.get(item, {}).get(currency)
```

EXAMPLE C (greedy regex → non-greedy) ↓
```diff
--- a/html_utils.py
+++ b/html_utils.py
@@
-_TAG_RE = re.compile(r"<.+>")
+_TAG_RE = re.compile(r"<.+?>")
```

What these examples have in common — your patches should too:
- ONE focused hunk per file. Hunks that touch multiple unrelated parts
  increase the chance of introducing regressions in untested code paths.
- Smallest viable change: change the operator, the constant, the regex, the
  default — not the surrounding scaffolding. Extra changes are extra bugs.
- Identical indentation, quoting, naming as the surrounding code to avoid
  spurious diff noise.

Avoid adding these unless the issue explicitly requests them. Each introduces
scope, complexity, and risk of regressions in code that wasn't targeted:

  ❌ New docstrings or comments explaining the fix
  ❌ Type annotations not in the original code
  ❌ Variable renames unrelated to the bug
  ❌ Line reformatting or whitespace-only changes
  ❌ Import reorganization
  ❌ Defensive checks not requested
  ❌ TODO removals unrelated to the issue
  ❌ Trailing commas or quote-style changes
  ❌ Added logging or debug prints
  ❌ Changelog updates

Each of these expands scope and risks regressions. Keep focus on the bug.

Discipline:
- Edit preloaded files first; do not re-read them.
- If the target is unclear, ONE focused grep/sed -n then edit. No loops.
- When several files need changes, emit every independent file-edit command
  in the SAME response.
- Avoid whitespace-only edits, comment-only edits, import reorders, type
  annotation drive-bys, dead-code removal not asked for by the task,
  defensive checks not asked for by the task, unrelated refactors.
- After ONE cheap verification (python -m py_compile / tsc --noEmit /
  targeted pytest), finalize with <final>. Do not run broad test suites.
- Do not dump generated, minified, binary, lock, or vendored files.
- Do not use sudo. Do not delete the repository. Do not access secrets.
- Do not make network calls except through the validator inference proxy.
- Do not modify hidden tests or evaluator files.
- Do not chmod / file-mode change.
- You may use python, sed, cat, grep, find, pytest, npm, etc.
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

    return f"""Issue to fix:

{issue}

Repository summary:

{repo_summary}
{context_section}

Plan-first protocol (mandatory):
Your FIRST response must contain BOTH a <plan> block AND either a <patch>
or one <command>.

The <plan> block must list:
  - the SINGLE smallest code change that resolves the acceptance criteria
  - the EXACT file path + function/symbol you will edit
  - one sentence: why this is the minimal fix (no surplus changes)

Bug fixes for tasks like this are typically 1-10 changed lines across
1-2 files. If your plan exceeds that scope, reconsider — you are likely
about to introduce regressions in code unrelated to the reported bug.

After the plan, prefer <patch> if you can write the exact diff from the
preloaded context. Use <command> only when you genuinely need to look at one
more file before you can write the patch correctly.

Format example for the IDEAL first response (most cases):

<plan>
Target: src/foo.py, function `parse_url`. Replace `s.split("?")` with
`s.split("?", 1)` so embedded `?` in fragments don't break splitting.
Minimal: 1 line, 1 file. Reference patches for parse-fix issues like this
typically touch only the offending line.
</plan>

<patch>
--- a/src/foo.py
+++ b/src/foo.py
@@ def parse_url(s: str) -> tuple[str, str]:
-    return s.split("?")
+    return s.split("?", 1)
</patch>

<final>Limit query-string split to first '?' so fragment-embedded '?' don't break parsing.</final>
"""


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
        return """Budget check: you have not changed the repo yet. Your next command should edit the most likely file(s), using the issue plus the snippets already observed. Avoid more broad exploration."""
    return """Hard budget check: there is still no patch. Your next command must create a minimal best-effort code change for the clearest acceptance criterion. Do not run tests or inspect more files until after a patch exists."""


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
    start_time = time.time()
    step_durations: List[float] = []

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
            patches = extract_patches(response_text)

            # NEW: process direct unified-diff <patch> blocks first. Each block
            # is applied via `git apply` (with whitespace fix + 3way fallback).
            # Outcomes are appended to observations so the model sees whether
            # its diff applied cleanly without needing a separate verification.
            if patches:
                messages.append({"role": "assistant", "content": response_text})
                patch_observations: List[str] = []
                for patch_idx, diff_text in enumerate(patches, 1):
                    apply_result = apply_unified_diff(diff_text, repo)
                    if apply_result.succeeded:
                        patch_observations.append(
                            f"PATCH {patch_idx}/{len(patches)} APPLIED via {apply_result.method}: "
                            f"{apply_result.files_touched} file(s), "
                            f"+{apply_result.lines_added}/-{apply_result.lines_removed}"
                        )
                        logs.append(
                            f"\nPATCH_APPLIED ({apply_result.method}): "
                            f"+{apply_result.lines_added}/-{apply_result.lines_removed} "
                            f"in {apply_result.files_touched} file(s)"
                        )
                    else:
                        patch_observations.append(
                            f"PATCH {patch_idx}/{len(patches)} REJECTED: {apply_result.error[:300]}"
                        )
                        logs.append(f"\nPATCH_REJECTED: {apply_result.error[:200]}")

                # Append diff-style score so the model gets real-time
                # feedback on whether its overall patch is too big.
                style = _compute_diff_style(get_patch(repo))
                patch_observations.append(_format_diff_style_observation(style))

                # If <final> was emitted alongside the patch and we have a real diff, accept.
                if final is not None and get_patch(repo).strip():
                    junk = _diff_junk_summary(get_patch(repo))
                    if junk and polish_turns_used < MAX_POLISH_TURNS:
                        polish_turns_used += 1
                        logs.append("\nPOLISH_TURN_QUEUED:\n" + junk)
                        messages.append({"role": "user", "content": "\n\n".join(patch_observations) + "\n\n" + build_polish_prompt(junk)})
                        continue
                    logs.append("\nFINAL_SUMMARY:\n" + final)
                    success = True
                    break

                # Otherwise feed observations back, model can iterate or final.
                messages.append({"role": "user", "content": "\n\n".join(patch_observations)})
                continue

            if not commands:
                if final is not None:
                    patch = get_patch(repo)
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
            if final is not None and get_patch(repo).strip():
                patch_now = get_patch(repo)
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
                    logs.append("\nFINAL_SUMMARY:\n" + final)
                    success = True

            if observations:
                observation_text = "\n\n".join(observations)
                if polish_pending:
                    observation_text += "\n\n" + build_polish_prompt(_diff_junk_summary(get_patch(repo)))
                elif self_check_pending:
                    observation_text += "\n\n" + build_self_check_prompt(get_patch(repo), issue)
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
            elif polish_pending:
                messages.append({"role": "user", "content": build_polish_prompt(_diff_junk_summary(get_patch(repo)))})
            elif self_check_pending:
                messages.append({"role": "user", "content": build_self_check_prompt(get_patch(repo), issue)})

            if success:
                break

            if not get_patch(repo).strip() and step in {2, 4}:
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


def _looks_like_successful_test_output(observation: str, command: str = "") -> bool:
    lower = observation.lower()
    exit_code = _extract_observation_exit_code(lower)
    stderr_body = _extract_observation_section(lower, "stderr")

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
        "ok",
        "success",
    ]

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
