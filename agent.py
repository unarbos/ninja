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
DEFAULT_MAX_TOKENS = int(os.environ.get("AGENT_MAX_TOKENS", "4096"))

MAX_OBSERVATION_CHARS = int(os.environ.get("AGENT_MAX_OBSERVATION_CHARS", "9000"))
MAX_TOTAL_LOG_CHARS = int(os.environ.get("AGENT_MAX_TOTAL_LOG_CHARS", "180000"))
MAX_CONVERSATION_CHARS = int(os.environ.get("AGENT_MAX_CONVERSATION_CHARS", "60000"))
MAX_PRELOADED_CONTEXT_CHARS = 12000
MAX_PRELOADED_FILES = 4
MAX_NO_COMMAND_REPAIRS = 3
MAX_COMMANDS_PER_RESPONSE = 12
MAX_POLISH_TURNS = 1

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


_MAX_LINE_LENGTH = 500


def _sanitize_output(text: str) -> str:
    if not text:
        return text
    cleaned = text.replace("\x00", "")
    if any(ord(c) < 32 and c not in "\n\r\t" for c in cleaned[:2048]):
        cleaned = "".join(
            c if (c in "\n\r\t" or 32 <= ord(c) < 127 or ord(c) > 127) else "?"
            for c in cleaned
        )
    lines = cleaned.split("\n")
    truncated_lines = []
    for line in lines:
        if len(line) > _MAX_LINE_LENGTH:
            truncated_lines.append(line[:_MAX_LINE_LENGTH] + "...[line truncated]")
        else:
            truncated_lines.append(line)
    return "\n".join(truncated_lines)


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


_RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}
_MAX_API_RETRIES = 3


def chat_completion(
    messages: List[Dict[str, str]],
    model: str,
    api_base: Optional[str],
    api_key: Optional[str],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: int = 120,
) -> Tuple[str, Optional[float], Dict[str, Any]]:
    """
    Minimal OpenAI-compatible /v1/chat/completions client using urllib
    with retry on transient failures.
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

    last_error: Optional[Exception] = None
    for attempt in range(_MAX_API_RETRIES):
        req = urllib.request.Request(url=url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                data = json.loads(raw)
            break
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"HTTP {e.code} from model endpoint: {err_body}")
            if e.code not in _RETRYABLE_HTTP_CODES:
                raise last_error from e
            delay = min(2 ** attempt * 2, 15)
            time.sleep(delay)
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            last_error = RuntimeError(f"Model request failed: {e}")
            delay = min(2 ** attempt * 2, 15)
            time.sleep(delay)
        except Exception as e:
            raise RuntimeError(f"Model request failed: {e}") from e
    else:
        raise last_error or RuntimeError("Model request failed after retries")

    try:
        content = data["choices"][0]["message"]["content"] or ""
    except Exception as e:
        raise RuntimeError(f"Unexpected model response shape: {data}") from e

    usage = data.get("usage") or {}
    cost = 0.0 if usage else None
    return content, cost, data


_HEAVY_CMD_RE = re.compile(
    r"\b(?:pip\s+install|npm\s+install|npm\s+ci|yarn\s+install|yarn\s+add|"
    r"pnpm\s+install|pnpm\s+add|apt-get|brew\s+install|cargo\s+build|"
    r"mvn\s+(?:install|package)|gradle\s+build|make\s+all)\b",
    re.IGNORECASE,
)
_HEAVY_CMD_TIMEOUT = 30


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

    if _HEAVY_CMD_RE.search(command):
        timeout = min(timeout, _HEAVY_CMD_TIMEOUT)

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
            stdout=_truncate(_sanitize_output(proc.stdout or ""), MAX_OBSERVATION_CHARS),
            stderr=_truncate(_sanitize_output(proc.stderr or ""), MAX_OBSERVATION_CHARS),
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


_OUTPUT_TAG_RE = re.compile(r"<(/?(?:command|final))\b", re.IGNORECASE)


def _escape_output_tags(text: str) -> str:
    return _OUTPUT_TAG_RE.sub(r"[\1", text)


def format_observation(result: CommandResult) -> str:
    parts = [
        f"COMMAND: {result.command}",
        f"EXIT_CODE: {result.exit_code}",
    ]
    if result.timed_out:
        parts.append(f"TIMED_OUT: true (after {result.duration_sec:.1f}s)")
    if result.blocked:
        parts.append("BLOCKED: true (dangerous command)")

    stderr_text = _escape_output_tags(result.stderr.strip())
    stdout_text = _escape_output_tags(result.stdout.strip())

    if result.exit_code != 0 and stderr_text:
        parts.extend(["", "STDERR:", stderr_text])
        if stdout_text:
            stdout_budget = max(2000, MAX_OBSERVATION_CHARS - len(stderr_text))
            parts.extend(["", "STDOUT:", _truncate(stdout_text, stdout_budget)])
    else:
        if stdout_text:
            parts.extend(["", "STDOUT:", stdout_text])
        if stderr_text:
            parts.extend(["", "STDERR:", stderr_text])

    return "\n".join(parts) + "\n"


_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
ACTION_RE = re.compile(r"<command>\s*(.*?)\s*</command>", re.IGNORECASE | re.DOTALL)
FINAL_RE = re.compile(r"<final>\s*(.*?)\s*</final>", re.IGNORECASE | re.DOTALL)
_MARKDOWN_BASH_RE = re.compile(
    r"```(?:bash|sh|shell|zsh)\s*\n(.*?)```",
    re.IGNORECASE | re.DOTALL,
)
_MARKDOWN_BARE_RE = re.compile(
    r"```\s*\n(.*?)```",
    re.DOTALL,
)


def _strip_think_blocks(text: str) -> str:
    return _THINK_RE.sub("", text)


def extract_commands(model_text: str) -> List[str]:
    cleaned = _strip_think_blocks(model_text)

    commands = [m.group(1).strip() for m in ACTION_RE.finditer(cleaned) if m.group(1).strip()]
    if commands:
        return commands

    bash_blocks = [m.group(1).strip() for m in _MARKDOWN_BASH_RE.finditer(cleaned) if m.group(1).strip()]
    if bash_blocks:
        return bash_blocks

    bare_blocks = [m.group(1).strip() for m in _MARKDOWN_BARE_RE.finditer(cleaned) if m.group(1).strip()]
    result = []
    for block in bare_blocks:
        lines = block.strip().splitlines()
        if not lines:
            continue
        if any(line.strip().startswith(("cat ", "sed ", "grep ", "find ", "cd ", "mkdir ",
                "python", "pip ", "npm ", "git ", "echo ", "touch ", "mv ", "cp ",
                "ls ", "pwd", "make ", "cargo ", "go ", "pytest", "tsc", "node "))
                for line in lines[:3]):
            result.append(block)
    if result:
        return result

    return []


def extract_command(model_text: str) -> Optional[str]:
    commands = extract_commands(model_text)
    return commands[0] if commands else None


def extract_final(model_text: str) -> Optional[str]:
    cleaned = _strip_think_blocks(model_text)
    match = FINAL_RE.search(cleaned)
    if not match:
        return None
    return match.group(1).strip()


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


def _hunk_is_blank_only(added: List[str], removed: List[str]) -> bool:
    body = [line for line in added + removed if line.strip()]
    return not body and bool(added or removed)


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
    per_file_budget = max(1200, MAX_PRELOADED_CONTEXT_CHARS // max(1, min(len(files), MAX_PRELOADED_FILES)))

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


SYSTEM_PROMPT = """You are an expert coding agent running inside a repository. You fix issues by editing files.

You have a tight wall-clock budget. Make a useful patch quickly — do not exhaustively explore.

## Tool Format

To run a bash command, wrap it EXACTLY like this:

<command>
your bash command here
</command>

You may include multiple <command>...</command> blocks in one response to run
several commands. Each block is executed in order.

When you are finished, respond with:

<final>
short summary of what you changed
</final>

## Examples

Reading a file:
<command>
sed -n '1,50p' src/utils.py
</command>

Editing with sed:
<command>
sed -i 's/old_function/new_function/g' src/main.py
</command>

Using a python heredoc for multi-line edits:
<command>
python3 << 'PYEOF'
import pathlib
p = pathlib.Path("src/config.py")
text = p.read_text()
text = text.replace("DEBUG = True", "DEBUG = False")
p.write_text(text)
PYEOF
</command>

Multiple independent edits in one response:
<command>
sed -i 's/bug/fix/' src/a.py
</command>
<command>
sed -i 's/old/new/' src/b.py
</command>

## Strategy

- If relevant file snippets are already in the prompt, edit them directly.
  Do not re-read files that were preloaded.
- If the target is unclear, run 1-2 focused search commands, then edit.
- By your second response you should be editing the most likely files.
- When several files need changes, put all edit commands in the same response.
- Prefer small, targeted changes. Avoid unrelated formatting or refactoring.
- After patching, run one cheap verification (syntax check, single test, or diff
  review), then finalize with <final>...</final>.
- If a verification command is slow or dependencies are missing, keep the patch
  and finalize anyway.

## Constraints

- Do not use sudo. Do not delete the repository. Do not access secrets.
- Do not modify hidden tests or evaluator files.
- Do not make network calls outside the validator inference proxy.
- Avoid chmod/file mode changes and unrelated formatting churn.
- Do not explain without editing — always produce code changes.
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


def build_no_command_repair_prompt(model_response: str = "") -> str:
    snippet = ""
    if model_response.strip():
        lines = model_response.strip().splitlines()
        preview = "\n".join(lines[:8])
        if len(lines) > 8:
            preview += "\n..."
        snippet = f"\n\nYour previous output started with:\n{preview}\n"

    return f"""Your previous response did not contain a valid <command>...</command> or <final>...</final> block.{snippet}
You MUST wrap every bash command exactly like this (note the XML tags):

<command>
your bash command here
</command>

If the patch is already complete, respond with:

<final>
summary of changes
</final>

Do NOT put commands in markdown code blocks. Use <command>...</command> tags only.
"""


def build_budget_pressure_prompt(step: int, max_steps: int) -> str:
    remaining = max_steps - step
    if step == 2:
        return (
            f"Budget check ({remaining} steps left): you have not changed the repo yet. "
            "Your next command should edit the most likely file(s), using the issue plus "
            "the snippets already observed. Avoid more broad exploration."
        )
    if step == 4:
        return (
            f"Hard budget check ({remaining} steps left): there is still no patch. Your "
            "next command must create a minimal best-effort code change for the clearest "
            "acceptance criterion. Do not run tests or inspect more files until after a "
            "patch exists."
        )
    if remaining <= 3:
        return (
            f"FINAL WARNING ({remaining} steps left): wrap up now. If a patch exists, run "
            "one quick verification and respond with <final>...</final>. If no patch exists, "
            "make the single most impactful edit and finalize immediately."
        )
    return ""


_READ_ONLY_CMD_RE = re.compile(
    r"^\s*(?:cat|head|tail|less|more|grep|rg|ag|ack|find|fd|ls|tree|wc|file|stat|"
    r"git\s+(?:log|show|diff|status|ls-files|blame)|pwd|echo|type|which|"
    r"sed\s+-n|awk)\b",
    re.IGNORECASE,
)

_EDIT_CMD_RE = re.compile(
    r"(?:sed\s+-i|perl\s+-[ip]|python[23]?\s|tee\s|cat\s*>|echo\s.*>|"
    r"patch\s|git\s+(?:apply|checkout|reset|cherry-pick)|"
    r"mv\s|cp\s|mkdir\s|touch\s|printf\s.*>)",
    re.IGNORECASE,
)


def _is_read_only_command(command: str) -> bool:
    return bool(_READ_ONLY_CMD_RE.search(command)) and not _EDIT_CMD_RE.search(command)


def _normalize_command_for_dedup(command: str) -> str:
    return re.sub(r"\s+", " ", command.strip().lower())


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
    recent_commands: List[str] = []
    consecutive_read_only = 0
    polish_turns_used = 0

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

            if not response_text.strip():
                logs.append("\nEMPTY_RESPONSE:\nModel returned an empty response.")
                consecutive_no_command += 1
                if consecutive_no_command >= MAX_NO_COMMAND_REPAIRS:
                    logs.append("\nSTOPPED:\nModel repeatedly returned empty responses.")
                    break
                messages.append({"role": "assistant", "content": "(empty response)"})
                messages.append({"role": "user", "content": (
                    "Your response was empty. You must respond with a bash command "
                    "wrapped in <command>...</command> tags, or <final>...</final> "
                    "if the patch is complete."
                )})
                continue

            commands = extract_commands(response_text)
            final = extract_final(response_text)

            if not commands:
                if final is not None:
                    patch = get_patch(repo)
                    if patch.strip():
                        junk = _diff_junk_summary(patch)
                        if junk and polish_turns_used < MAX_POLISH_TURNS:
                            polish_turns_used += 1
                            messages.append({"role": "assistant", "content": response_text})
                            messages.append({"role": "user", "content": build_polish_prompt(junk)})
                            logs.append(f"\nPOLISH_TURN:\n{junk}")
                            continue
                        logs.append("\nFINAL_SUMMARY:\n" + final)
                        success = True
                        break
                    if step < max_steps:
                        logs.append("\nFALSE_FINAL:\nModel declared <final> but no patch exists yet.")
                        messages.append({"role": "assistant", "content": response_text})
                        messages.append({"role": "user", "content": (
                            "You responded with <final> but the repository has ZERO changes. "
                            "An empty patch guarantees a loss. You MUST edit at least one file. "
                            "Use <command>...</command> to make the most obvious fix from the "
                            "issue description, then finalize."
                        )})
                        consecutive_no_command += 1
                        continue
                    logs.append("\nFINAL_SUMMARY_EMPTY:\n" + final)
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
                messages.append({"role": "user", "content": build_no_command_repair_prompt(response_text)})
                continue

            consecutive_no_command = 0
            messages.append({"role": "assistant", "content": response_text})
            observations: List[str] = []
            command_batch = commands[:MAX_COMMANDS_PER_RESPONSE]

            all_read_only_this_step = True
            for command_index, command in enumerate(command_batch, 1):
                normalized = _normalize_command_for_dedup(command)
                if normalized in recent_commands[-6:]:
                    observations.append(
                        f"OBSERVATION {command_index}/{len(command_batch)}:\n"
                        f"SKIPPED: This command was already run recently. "
                        f"Try a different approach or edit a file instead.\n"
                    )
                    logs.append(f"\nSKIPPED_DUPLICATE:\n{command}")
                    continue
                recent_commands.append(normalized)
                if len(recent_commands) > 20:
                    recent_commands = recent_commands[-20:]

                if not _is_read_only_command(command):
                    all_read_only_this_step = False

                result = run_command(command, repo, timeout=command_timeout)
                observation = format_observation(result)
                observations.append(f"OBSERVATION {command_index}/{len(command_batch)}:\n{observation}")
                logs.append(f"\nOBSERVATION {command_index}/{len(command_batch)}:\n" + observation)

                if step >= 4 or command_index > 1:
                    patch = get_patch(repo)
                    if (
                        patch.strip()
                        and _looks_like_successful_test_output(observation, command)
                        and _patch_covers_required_paths(patch, issue)
                    ):
                        logs.append("\nAUTO_STOP:\nPatch exists and latest command looked like successful tests.")
                        success = True
                        break
                    if patch.strip() and result.timed_out:
                        logs.append("\nPATCH_READY:\nPatch exists and latest command exceeded the local command timeout.")
                        success = True
                        break
                    if patch.strip() and step >= 8 and _looks_like_patch_review_command(command, result):
                        logs.append("\nPATCH_READY:\nPatch exists and latest command reviewed the diff/status.")
                        success = True
                        break

            if len(commands) > len(command_batch):
                observations.append(
                    f"NOTE: Only the first {len(command_batch)} command blocks were executed. "
                    "Continue with one command at a time if more work remains."
                )

            if final is not None and get_patch(repo).strip():
                junk = _diff_junk_summary(get_patch(repo))
                if junk and polish_turns_used < MAX_POLISH_TURNS:
                    polish_turns_used += 1
                    messages.append({"role": "user", "content": build_polish_prompt(junk)})
                    logs.append(f"\nPOLISH_TURN:\n{junk}")
                else:
                    logs.append("\nFINAL_SUMMARY:\n" + final)
                    success = True

            if observations:
                observation_text = "\n\n".join(observations)
                if not success and get_patch(repo).strip():
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

            if success:
                break

            if all_read_only_this_step and not get_patch(repo).strip():
                consecutive_read_only += 1
            else:
                consecutive_read_only = 0

            if consecutive_read_only >= 3 and not get_patch(repo).strip():
                messages.append({"role": "user", "content": (
                    "WARNING: You have spent 3+ turns only reading/searching without "
                    "editing any files. An empty patch guarantees a loss. Your VERY NEXT "
                    "command must edit a file. Use sed -i, python3 heredoc, or cat > to "
                    "make the most obvious change for the issue."
                )})
                consecutive_read_only = 0

            remaining = max_steps - step
            needs_pressure = (
                (not get_patch(repo).strip() and step in {2, 4})
                or remaining <= 3
            )
            if needs_pressure:
                pressure = build_budget_pressure_prompt(step, max_steps)
                if pressure:
                    messages.append({"role": "user", "content": pressure})

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


_BAD_TEST_RE = re.compile(
    r"\b(?:fail(?:ed|ure|ures)|error(?:s)?|traceback|assertion\s*error|syntax\s*error|exception|FAIL)\b",
    re.IGNORECASE,
)
_GOOD_TEST_RE = re.compile(
    r"\b(?:passed|all\s+passed|tests?\s+passed|success(?:ful)?|OK)\b",
    re.IGNORECASE,
)


def _looks_like_successful_test_output(observation: str, command: str = "") -> bool:
    lower = observation.lower()
    exit_code = _extract_observation_exit_code(lower)
    stderr_body = _extract_observation_section(lower, "stderr")

    if exit_code is not None and exit_code != 0:
        return False

    has_good = bool(_GOOD_TEST_RE.search(observation))
    has_bad = bool(_BAD_TEST_RE.search(observation))
    if stderr_body and _BAD_TEST_RE.search(stderr_body):
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
    match = re.search(r"(?m)^exit_code:?\s*(-?\d+)", observation_lower)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _extract_observation_section(observation_lower: str, section: str) -> str:
    match = re.search(
        rf"(?ms)^{re.escape(section.lower())}:?\s*\n(.*?)(?:\n[a-z_]+:?\s|\Z)",
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
