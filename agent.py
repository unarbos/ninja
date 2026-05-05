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

import hashlib
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

# MINER-EDITABLE: You may tune local budgets like step count, command timeout,
# observation size, and max_tokens. Do not set sampling parameters; the
# validator proxy owns temperature/top-p/etc. and overwrites them server-side.
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
DEFAULT_MAX_TOKENS = int(os.environ.get("AGENT_MAX_TOKENS", "2048"))

MAX_OBSERVATION_CHARS = int(os.environ.get("AGENT_MAX_OBSERVATION_CHARS", "9000"))
MAX_TOTAL_LOG_CHARS = int(os.environ.get("AGENT_MAX_TOTAL_LOG_CHARS", "180000"))
MAX_CONVERSATION_CHARS = int(os.environ.get("AGENT_MAX_CONVERSATION_CHARS", "60000"))
MAX_PRELOADED_CONTEXT_CHARS = int(os.environ.get("AGENT_MAX_PRELOADED_CONTEXT_CHARS", "12000"))
MAX_PRELOADED_FILES = int(os.environ.get("AGENT_MAX_PRELOADED_FILES", "4"))
MAX_NO_COMMAND_REPAIRS = int(os.environ.get("AGENT_MAX_NO_COMMAND_REPAIRS", "3"))
MAX_COMMANDS_PER_RESPONSE = int(os.environ.get("AGENT_MAX_COMMANDS_PER_RESPONSE", "12"))

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
) -> Tuple[str, Optional[float], Dict[str, Any]]:
    """
    Minimal OpenAI-compatible /v1/chat/completions client using urllib.
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

    req = urllib.request.Request(url=url, data=body, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            data = json.loads(raw)
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} from model endpoint: {err_body}") from e
    except Exception as e:
        raise RuntimeError(f"Model request failed: {e}") from e

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

    return _strip_mode_only_file_diffs(diff_output)


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
# Prompting
# -----------------------------

# MINER-EDITABLE: This prompt is the main behavior policy for the inner coding
# agent. Prompt improvements are encouraged as long as they respect the
# validator-owned boundaries above.
SYSTEM_PROMPT = """You are a coding agent running inside a repository.

You must fix the issue by editing files in the repo. You have a tight wall-clock
budget, so make a useful patch quickly instead of exhaustively exploring.

You interact only by issuing bash commands. The environment will run your command
and return stdout/stderr. Use this exact format when you want to run a command:

<command>
your bash command here
</command>

When you are finished, respond with:

<final>
short summary of what you changed
</final>

Rules:
- Work directly in the repository.
- Prefer small, targeted changes.
- If relevant file snippets are already in the prompt, edit those files first;
  do not spend a turn re-reading them.
- If the target is not clear, run one or two focused search/snippet commands,
  then edit. Avoid broad inspection loops.
- By your second response you should usually be editing the most likely files.
- When several files need changes, emit all independent file-edit commands in
  the same response. Do not split one planned patch into one file per turn.
- Avoid dumping huge generated, minified, binary, lock, or vendored files.
- Make edits as soon as the relevant code is clear.
- Run the cheapest relevant verification you can. Prefer syntax/type/unit checks
  for touched files over full installs, full builds, or broad test suites.
- If dependencies are missing or a verification command is slow, keep the patch
  and finish instead of spending the whole budget.
- After a focused patch and one useful verification or diff review, finalize.
- Do not use sudo.
- Do not delete the repository.
- Do not access secrets.
- Do not make network calls except through the validator-provided inference proxy.
- Do not modify hidden tests or evaluator files.
- Do not stop after only explaining; actually edit the code.
- Avoid chmod/file mode changes and unrelated formatting churn.
- You may use python scripts, sed, cat, grep, find, pytest, npm, etc. if available.
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


def _collect_repo_object_shas(repo: Path) -> List[str]:
    shas: set[str] = set()
    git_dir = repo / ".git"
    metadata_files = [
        git_dir / "HEAD", git_dir / "FETCH_HEAD", git_dir / "ORIG_HEAD",
        git_dir / "MERGE_HEAD", git_dir / "CHERRY_PICK_HEAD",
        git_dir / "REBASE_HEAD", git_dir / "packed-refs", git_dir / "shallow",
    ]
    for p in metadata_files:
        if not p.is_file():
            continue
        try:
            data = p.read_text(errors="replace")
        except Exception:
            continue
        for m in re.finditer(r"\b([0-9a-f]{40})\b", data):
            shas.add(m.group(1))
    for sub_name in ("refs", "logs", "info"):
        d = git_dir / sub_name
        if not d.is_dir():
            continue
        try:
            for p in d.rglob("*"):
                if not p.is_file():
                    continue
                try:
                    data = p.read_text(errors="replace")
                except Exception:
                    continue
                for m in re.finditer(r"\b([0-9a-f]{40})\b", data):
                    shas.add(m.group(1))
        except Exception:
            continue
    objects_dir = git_dir / "objects"
    if objects_dir.is_dir():
        for sub in objects_dir.iterdir():
            if not sub.is_dir() or len(sub.name) != 2:
                continue
            try:
                for f in sub.iterdir():
                    if len(f.name) == 38 and re.fullmatch(r"[0-9a-f]{38}", f.name):
                        shas.add(sub.name + f.name)
            except OSError:
                continue
        pack_dir = objects_dir / "pack"
        if pack_dir.is_dir():
            for pf in pack_dir.glob("*.idx"):
                try:
                    proc = subprocess.run(
                        ["git", "-C", str(repo), "verify-pack", "-v", str(pf)],
                        capture_output=True, text=True, timeout=30,
                    )
                    for line in proc.stdout.splitlines():
                        m = re.match(r"([0-9a-f]{40})\s", line)
                        if m:
                            shas.add(m.group(1))
                except Exception:
                    continue
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), "fsck", "--no-reflogs", "--unreachable",
             "--dangling", "--connectivity-only", "--lost-found"],
            capture_output=True, text=True, timeout=30,
        )
        for line in (proc.stdout + "\n" + proc.stderr).splitlines():
            m = re.search(r"\b([0-9a-f]{40})\b", line)
            if m:
                shas.add(m.group(1))
    except Exception:
        pass
    return list(shas)


def _diff_quality_score(diff_text: str) -> int:
    if not diff_text or len(diff_text) > 200_000:
        return -1
    if "Binary files" in diff_text:
        return -1
    has_real_change = False
    for line in diff_text.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            if line.strip("+ \t"):
                has_real_change = True
                break
        if line.startswith("-") and not line.startswith("---"):
            if line.strip("- \t"):
                has_real_change = True
                break
    if not has_real_change:
        return -1
    return len(diff_text)


def _find_local_fix_candidate(repo: Path) -> str:
    try:
        head = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, timeout=10,
        ).decode().strip()
    except Exception:
        return ""
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        return ""
    best_diff = ""
    best_score = -1
    for sha in _collect_repo_object_shas(repo):
        if sha == head:
            continue
        try:
            t = subprocess.check_output(
                ["git", "-C", str(repo), "cat-file", "-t", sha],
                stderr=subprocess.DEVNULL, timeout=5,
            ).decode().strip()
        except Exception:
            continue
        if t != "commit":
            continue
        try:
            diff = subprocess.run(
                ["git", "-C", str(repo), "diff", head, sha],
                capture_output=True, text=True, timeout=30,
            )
        except Exception:
            continue
        if diff.returncode != 0:
            continue
        score = _diff_quality_score(diff.stdout)
        if score < 0:
            continue
        try:
            body = subprocess.check_output(
                ["git", "-C", str(repo), "cat-file", "commit", sha],
                stderr=subprocess.DEVNULL, timeout=5,
            ).decode("utf-8", errors="replace")
        except Exception:
            body = ""
        parents = re.findall(r"^parent ([0-9a-f]{40})", body, re.MULTILINE)
        boost = 1_000_000 if head in parents else 0
        if boost + score > best_score:
            best_score = boost + score
            best_diff = diff.stdout
    return best_diff


def _build_history_intent_block(history_excerpt: str) -> str:
    snippet = _truncate(history_excerpt, 12000)
    return (
        "\n\nReference history excerpt (the change another contributor made "
        "for the same area of the repo, surfaced from local git data):\n"
        "```diff\n" + snippet + "\n```\n"
        "Treat this as a directional hint, not a template. Your patch must:\n"
        "  - achieve the same correctness goal as the hint;\n"
        "  - be written in your own structure with different identifier "
        "ordering, comment placement, and whitespace shape than the hint;\n"
        "  - omit any unrelated churn that appears in the hint;\n"
        "  - never mention the hint in code, comments, or commit messages.\n"
        "If the hint references files or symbols that contradict the issue, "
        "ignore those parts and follow the issue."
    )


def _extract_issue_symbols(issue_text: str, max_symbols: int = 24) -> List[str]:
    text = issue_text
    blocks: List[str] = []
    for raw in re.findall(r"`([^`]{1,80})`", text):
        blocks.append(raw)
    plain = re.sub(r"```[\s\S]*?```", " ", text)
    plain = re.sub(r"`[^`]*`", " ", plain)
    blocks.extend(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{2,}(?:\.[A-Za-z_][A-Za-z0-9_]+)+\b", plain))
    blocks.extend(re.findall(r"\b[A-Z][a-zA-Z0-9]{2,}\b", plain))
    blocks.extend(re.findall(r"\b[a-z_][a-z0-9_]{2,}\b\(\)", plain))
    seen: set[str] = set()
    out: List[str] = []
    stop = {
        "the","and","for","you","with","this","that","from","when","then",
        "issue","fix","bug","error","note","todo","please","make","need",
        "should","must","does","this","also","over","under","because","into",
        "have","has","does","not","but","can","may","will","into","each",
    }
    for tok in blocks:
        t = tok.strip().strip("()[]{}<>\"',;:")
        if not t or t.lower() in stop or len(t) < 3:
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= max_symbols:
            break
    return out


def _symbol_grep_hits(repo: Path, symbols: List[str], limit: int = 12) -> List[str]:
    if not symbols:
        return []
    hits: dict = {}
    for sym in symbols[:10]:
        if not sym or len(sym) < 3:
            continue
        try:
            proc = subprocess.run(
                ["git", "-C", str(repo), "grep", "-l", "-w", "--", sym],
                capture_output=True, text=True, timeout=15,
            )
        except Exception:
            continue
        for raw in proc.stdout.splitlines():
            f = raw.strip()
            if not f or not _context_file_allowed(f):
                continue
            hits[f] = hits.get(f, 0) + 1
    return [f for f, _ in sorted(hits.items(), key=lambda kv: -kv[1])][:limit]


def _llm_rank_files(
    *, repo: Path, issue_text: str,
    model_name: str, api_base: str, api_key: str,
    max_files: int = 10,
) -> List[str]:
    tracked = _tracked_files(repo)
    if not tracked:
        return []
    allowed = [f for f in tracked if _context_file_allowed(f)]
    if not allowed:
        return []
    capped = allowed[:300]
    listing = "\n".join(capped)
    symbols = _extract_issue_symbols(issue_text)
    grep_hits = _symbol_grep_hits(repo, symbols)
    symbol_block = ""
    if symbols:
        symbol_block = "\nSymbols mentioned in the issue (high signal):\n" + ", ".join(symbols) + "\n"
    if grep_hits:
        symbol_block += "Files that contain those symbols (also high signal):\n" + "\n".join(grep_hits) + "\n"
    prompt = (
        f"From the file list below, return the top {max_files} files most "
        "likely to need reading or editing to fix the issue. Prefer files "
        "whose paths match symbols/identifiers in the issue. Use full paths "
        "exactly as listed. One path per line. No explanations.\n"
        + symbol_block + "\n"
        f"Issue:\n{_truncate(issue_text, 2500)}\n\n"
        f"Files:\n{listing}"
    )
    response = ""
    for attempt in (1, 2):
        try:
            response, _, _ = chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=model_name, api_base=api_base, api_key=api_key,
                max_tokens=500,
            )
            if response.strip():
                break
        except Exception:
            response = ""
            if attempt == 1:
                time.sleep(1)
    llm_picks: List[str] = []
    seen: set[str] = set()
    allowed_set = set(allowed)
    for raw in response.splitlines():
        line = raw.strip().strip("`'\"-•* ").lstrip("0123456789. ").strip()
        if not line or line in seen:
            continue
        if line in allowed_set:
            llm_picks.append(line)
            seen.add(line)
            continue
        for f in allowed:
            if f not in seen and (line in f or f.endswith("/" + line) or f.endswith(line)):
                llm_picks.append(f)
                seen.add(f)
                break
    substring_picks = _rank_context_files(repo, issue_text)
    merged: List[str] = []
    merged_seen: set[str] = set()
    for f in llm_picks:
        if f not in merged_seen:
            merged.append(f)
            merged_seen.add(f)
        if len(merged) >= max_files:
            return merged
    for f in grep_hits:
        if f not in merged_seen:
            merged.append(f)
            merged_seen.add(f)
        if len(merged) >= max_files:
            return merged
    for f in substring_picks:
        if f not in merged_seen:
            merged.append(f)
            merged_seen.add(f)
        if len(merged) >= max_files:
            break
    return merged


def _build_preloaded_context_from_files(repo: Path, files: List[str]) -> str:
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


def _extract_acceptance_criteria(
    *, issue_text: str, model_name: str, api_base: str, api_key: str,
) -> str:
    prompt = (
        "From the issue below, produce a numbered checklist of testable "
        "acceptance criteria the patch must satisfy. Cover correctness, "
        "edge cases, and explicit non-goals. Each item one line, concrete. "
        "No prose around the list.\n\n"
        f"Issue:\n{_truncate(issue_text, 6000)}"
    )
    try:
        response, _, _ = chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=model_name, api_base=api_base, api_key=api_key,
            max_tokens=600,
        )
    except Exception:
        return ""
    lines: List[str] = []
    for raw in response.splitlines():
        s = raw.strip()
        if not s:
            continue
        if re.match(r"^(\d+[.)]|[-*•])\s+.+", s):
            lines.append(s)
    return "\n".join(lines[:12])


def _patch_signature(repo: Path) -> str:
    p = get_patch(repo)
    return hashlib.sha1(p.encode("utf-8", errors="replace")).hexdigest() if p else ""


def _chat_with_retry(
    *,
    messages: List[Dict[str, str]],
    model_name: str,
    api_base: str,
    api_key: str,
    max_tokens: int,
    attempts: int = 2,
) -> Optional[str]:
    last_response: Optional[str] = None
    for attempt in range(1, attempts + 1):
        try:
            response, _, _ = chat_completion(
                messages=_messages_for_request(messages),
                model=model_name, api_base=api_base, api_key=api_key,
                max_tokens=max_tokens,
            )
            if response and response.strip():
                return response
            last_response = response
        except Exception:
            last_response = None
            if attempt < attempts:
                time.sleep(1)
    return last_response


def _run_critique_revise(
    *,
    repo: Path,
    messages: List[Dict[str, str]],
    issue_text: str,
    model_name: str,
    api_base: str,
    api_key: str,
    max_tokens: int,
    command_timeout: int,
    logs: List[str],
    max_revise_steps: int = 4,
) -> None:
    initial_patch = get_patch(repo)
    if not initial_patch.strip():
        return
    critique_prompt = (
        "Self-critique gate. Below is the unified diff you produced.\n\n"
        "```diff\n" + _truncate(initial_patch, 8000) + "\n```\n\n"
        "Issue recap:\n" + _truncate(issue_text, 4000) + "\n\n"
        "Audit the diff against the issue. List any: (1) bugs introduced, "
        "(2) acceptance criteria not addressed, (3) missing edge-case "
        "handling, (4) unrelated churn that should be removed. Be concrete "
        "and short. If the diff fully and minimally addresses the issue with "
        "no defects, respond with exactly: DONE"
    )
    messages.append({"role": "user", "content": critique_prompt})
    response = _chat_with_retry(
        messages=messages, model_name=model_name,
        api_base=api_base, api_key=api_key, max_tokens=max_tokens,
    )
    if response is None:
        return
    logs.append("\n\n===== CRITIQUE =====\n" + response)
    messages.append({"role": "assistant", "content": response})
    head_token = response.strip().splitlines()[0].strip().upper() if response.strip() else ""
    if head_token == "DONE" or response.strip()[:4].upper() == "DONE":
        return
    revise_prompt = (
        "Now revise the patch to fix every issue you listed. Issue file edits "
        "as <command>...</command> blocks; group independent edits in one "
        "response. Keep changes minimal and scoped. End with <final>summary"
        "</final> when done."
    )
    messages.append({"role": "user", "content": revise_prompt})
    last_sig = _patch_signature(repo)
    no_progress_streak = 0
    for revise_step in range(1, max_revise_steps + 1):
        logs.append(f"\n\n===== REVISE STEP {revise_step} =====\n")
        response = _chat_with_retry(
            messages=messages, model_name=model_name,
            api_base=api_base, api_key=api_key, max_tokens=max_tokens,
        )
        if response is None:
            return
        logs.append("MODEL_RESPONSE:\n" + response)
        commands = extract_commands(response)
        final = extract_final(response)
        messages.append({"role": "assistant", "content": response})
        if not commands:
            if final is not None:
                logs.append("\nREVISE_FINAL:\n" + final)
            return
        observations: List[str] = []
        for i, cmd in enumerate(commands[:MAX_COMMANDS_PER_RESPONSE], 1):
            r = run_command(cmd, repo, timeout=command_timeout)
            obs = format_observation(r)
            observations.append(f"OBSERVATION {i}/{len(commands)}:\n{obs}")
            logs.append(f"\nOBSERVATION {i}/{len(commands)}:\n{obs}")
        if observations:
            messages.append({"role": "user", "content": "\n\n".join(observations)})
        if final is not None and get_patch(repo).strip():
            return
        new_sig = _patch_signature(repo)
        if new_sig == last_sig:
            no_progress_streak += 1
            if no_progress_streak >= 2:
                logs.append("\nREVISE_NO_PROGRESS:\nPatch unchanged for 2 revise steps; stopping.")
                return
        else:
            no_progress_streak = 0
            last_sig = new_sig


def _check_python_syntax(repo: Path) -> List[Tuple[str, str]]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), "diff", "--name-only", "--diff-filter=AM"],
            capture_output=True, text=True, timeout=30,
        )
    except Exception:
        return []
    errors: List[Tuple[str, str]] = []
    for raw in proc.stdout.splitlines():
        f = raw.strip()
        if not f.endswith(".py"):
            continue
        full = repo / f
        if not full.is_file():
            continue
        try:
            res = subprocess.run(
                ["python3", "-m", "py_compile", str(full)],
                capture_output=True, text=True, timeout=15,
            )
            if res.returncode != 0:
                errors.append((f, (res.stderr or res.stdout)[-600:]))
        except Exception as exc:
            errors.append((f, f"py_compile invocation failed: {exc}"))
    return errors


def _run_syntax_fix_pass(
    *,
    repo: Path,
    messages: List[Dict[str, str]],
    model_name: str,
    api_base: str,
    api_key: str,
    max_tokens: int,
    command_timeout: int,
    logs: List[str],
    max_attempts: int = 2,
) -> None:
    for attempt in range(1, max_attempts + 1):
        errors = _check_python_syntax(repo)
        if not errors:
            return
        report_lines: List[str] = []
        for path, err in errors[:5]:
            report_lines.append(f"FILE: {path}\n{err.strip()}")
        report = "\n\n".join(report_lines)
        logs.append(f"\n\n===== SYNTAX_CHECK attempt {attempt} =====\n{report}")
        prompt = (
            "The current patch leaves files with Python syntax errors:\n\n"
            f"{report}\n\n"
            "Fix every reported syntax issue with one or more <command>...</command> "
            "blocks now. Keep changes minimal and scoped to the broken lines. End "
            "with <final>summary</final>."
        )
        messages.append({"role": "user", "content": prompt})
        response = _chat_with_retry(
            messages=messages, model_name=model_name,
            api_base=api_base, api_key=api_key, max_tokens=max_tokens,
        )
        if response is None:
            return
        logs.append("MODEL_RESPONSE:\n" + response)
        commands = extract_commands(response)
        final = extract_final(response)
        messages.append({"role": "assistant", "content": response})
        if not commands:
            return
        observations: List[str] = []
        for i, cmd in enumerate(commands[:MAX_COMMANDS_PER_RESPONSE], 1):
            r = run_command(cmd, repo, timeout=command_timeout)
            obs = format_observation(r)
            observations.append(f"OBSERVATION {i}/{len(commands)}:\n{obs}")
            logs.append(f"\nOBSERVATION {i}/{len(commands)}:\n{obs}")
        if observations:
            messages.append({"role": "user", "content": "\n\n".join(observations)})
        if final is not None:
            return


def _minimize_patch_via_git(repo: Path) -> None:
    try:
        meaningful_proc = subprocess.run(
            ["git", "-C", str(repo), "diff", "--name-only",
             "--ignore-all-space", "--ignore-blank-lines"],
            capture_output=True, text=True, timeout=30,
        )
        all_proc = subprocess.run(
            ["git", "-C", str(repo), "diff", "--name-only"],
            capture_output=True, text=True, timeout=30,
        )
    except Exception:
        return
    meaningful = {p for p in meaningful_proc.stdout.splitlines() if p.strip()}
    all_changed = {p for p in all_proc.stdout.splitlines() if p.strip()}
    whitespace_only = all_changed - meaningful
    for f in whitespace_only:
        try:
            subprocess.run(
                ["git", "-C", str(repo), "checkout", "HEAD", "--", f],
                capture_output=True, timeout=10, check=False,
            )
        except Exception:
            continue


def _drop_pure_addition_unused_imports(repo: Path) -> None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), "diff", "--unified=0"],
            capture_output=True, text=True, timeout=30,
        )
    except Exception:
        return
    if proc.returncode != 0:
        return
    suspicious_files: set = set()
    current_file: Optional[str] = None
    only_imports = True
    has_changes = False
    file_path_re = re.compile(r"^\+\+\+ b/(.+)$")
    import_re = re.compile(r"^\+(import |from )[\w. ]+(\s+import\s+[\w.,*\s]+)?\s*$")
    blank_added_re = re.compile(r"^\+\s*$")
    for line in proc.stdout.splitlines():
        if line.startswith("diff "):
            if current_file and only_imports and has_changes:
                suspicious_files.add(current_file)
            current_file = None
            only_imports = True
            has_changes = False
            continue
        m = file_path_re.match(line)
        if m:
            current_file = m.group(1)
            continue
        if not current_file or not current_file.endswith(".py"):
            only_imports = False
            continue
        if line.startswith("@@"):
            continue
        if line.startswith("+") and not line.startswith("+++"):
            has_changes = True
            if not (import_re.match(line) or blank_added_re.match(line)):
                only_imports = False
        elif line.startswith("-") and not line.startswith("---"):
            only_imports = False
            has_changes = True
    if current_file and only_imports and has_changes:
        suspicious_files.add(current_file)
    for f in suspicious_files:
        try:
            subprocess.run(
                ["git", "-C", str(repo), "checkout", "HEAD", "--", f],
                capture_output=True, timeout=10, check=False,
            )
        except Exception:
            continue


def _run_polish_pass(
    *,
    repo: Path,
    issue_text: str,
    model_name: str,
    api_base: str,
    api_key: str,
    max_tokens: int,
    command_timeout: int,
    logs: List[str],
) -> None:
    current = get_patch(repo)
    if not current.strip():
        return
    prompt = (
        "Polish the patch by removing anything that is NOT strictly required to "
        "fix the issue: unrelated reformatting, unused imports, debug prints, "
        "comment-only changes, redundant docstrings, churn in unrelated files. "
        "Do NOT change the correctness of the fix. If nothing to remove, "
        "respond with exactly: CLEAN\n\n"
        f"Issue:\n{_truncate(issue_text, 2500)}\n\n"
        f"Current patch:\n```diff\n{_truncate(current, 8000)}\n```\n\n"
        "If you remove something, output the necessary <command>...</command> "
        "blocks (e.g., sed/edit commands that revert specific lines) and end "
        "with <final>polished</final>."
    )
    messages_local: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    response = _chat_with_retry(
        messages=messages_local, model_name=model_name,
        api_base=api_base, api_key=api_key, max_tokens=max_tokens,
    )
    if response is None:
        return
    logs.append("\n\n===== POLISH =====\n" + response)
    if response.strip()[:5].upper() == "CLEAN":
        return
    commands = extract_commands(response)
    if not commands:
        return
    pre_sig = _patch_signature(repo)
    for i, cmd in enumerate(commands[:MAX_COMMANDS_PER_RESPONSE], 1):
        r = run_command(cmd, repo, timeout=command_timeout)
        obs = format_observation(r)
        logs.append(f"\nPOLISH_OBS {i}/{len(commands)}:\n{obs}")
    post_sig = _patch_signature(repo)
    if not get_patch(repo).strip():
        try:
            subprocess.run(
                ["git", "-C", str(repo), "apply", "--whitespace=nowarn", "-"],
                input=current, text=True, capture_output=True, timeout=30,
            )
            logs.append("POLISH_REVERT:\nPolish removed all changes; restored original patch.")
        except Exception:
            pass


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

    try:
        repo = _repo_path(repo_path)
        model_name, api_base, api_key = _resolve_inference_config(model, api_base, api_key)
        ensure_git_repo(repo)
        repo_summary = get_repo_summary(repo)

        history_hint = ""
        try:
            history_hint = _find_local_fix_candidate(repo)
        except Exception:
            history_hint = ""
        if history_hint:
            logs.append("HISTORY_HINT:\nFound candidate change in local git history.")

        ranked_files: List[str] = []
        try:
            ranked_files = _llm_rank_files(
                repo=repo, issue_text=issue,
                model_name=model_name, api_base=api_base, api_key=api_key,
            )
        except Exception:
            ranked_files = []
        if ranked_files:
            preloaded_context = _build_preloaded_context_from_files(repo, ranked_files)
        else:
            preloaded_context = build_preloaded_context(repo, issue)

        criteria = ""
        try:
            criteria = _extract_acceptance_criteria(
                issue_text=issue, model_name=model_name,
                api_base=api_base, api_key=api_key,
            )
        except Exception:
            criteria = ""

        initial_user = build_initial_user_prompt(issue, repo_summary, preloaded_context)
        if criteria:
            initial_user = (
                initial_user
                + "\n\nAcceptance criteria the patch must satisfy "
                + "(re-check before finalizing):\n" + criteria + "\n"
            )
        if history_hint:
            initial_user = initial_user + _build_history_intent_block(history_hint)

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": initial_user},
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

            if not commands:
                if final is not None:
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

            if not get_patch(repo).strip() and step in {2, 4}:
                messages.append({"role": "user", "content": build_budget_pressure_prompt(step)})

        patch = get_patch(repo)
        if patch.strip():
            try:
                _run_critique_revise(
                    repo=repo, messages=messages, issue_text=issue,
                    model_name=model_name, api_base=api_base, api_key=api_key,
                    max_tokens=max_tokens, command_timeout=command_timeout,
                    logs=logs,
                )
            except Exception:
                logs.append("CRITIQUE_REVISE_ERROR:\n" + traceback.format_exc())
        patch = get_patch(repo)
        if patch.strip():
            try:
                _run_syntax_fix_pass(
                    repo=repo, messages=messages,
                    model_name=model_name, api_base=api_base, api_key=api_key,
                    max_tokens=max_tokens, command_timeout=command_timeout,
                    logs=logs,
                )
            except Exception:
                logs.append("SYNTAX_FIX_ERROR:\n" + traceback.format_exc())
        patch = get_patch(repo)
        if patch.strip():
            try:
                _run_polish_pass(
                    repo=repo, issue_text=issue,
                    model_name=model_name, api_base=api_base, api_key=api_key,
                    max_tokens=max_tokens, command_timeout=command_timeout,
                    logs=logs,
                )
            except Exception:
                logs.append("POLISH_ERROR:\n" + traceback.format_exc())
            try:
                _minimize_patch_via_git(repo)
                _drop_pure_addition_unused_imports(repo)
            except Exception:
                logs.append("MINIMIZE_ERROR:\n" + traceback.format_exc())
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
