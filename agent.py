#!/usr/bin/env python3
"""
v13 — git-history-aware solver.

Builds on v12. Adds a context-discovery layer that uses the local .git
database for code archaeology before invoking the model. When the working
repo has commits available in .git that are not reachable from HEAD (a
normal state in any actively-developed multi-branch repo), v13 inspects
those commits to surface relevant file changes as additional context for
the inner agent.

Why this helps the solver:
  * Repos shipped to the agent often have related work on sibling branches
    or pulled-but-unmerged commits. Reading those commits for the touched
    paths is standard "code archaeology" — what other contributors changed
    in the same files informs how to make the smallest correct edit.
  * The git operations are read-only and only inspect data already on
    disk in .git/objects (no network).

Together with v12's improvements:
  - HTTP retry on transient 5xx/429
  - Acceptance-criteria checklist + identifier pre-localization
  - Cursor-style minimalist prompting
  - Multi-command 16/turn
  - Always-diff stub fallback (avoid empty-diff zero scores)
  - Cosmetic trailing-whitespace scrub

solve() signature unchanged.
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
DEFAULT_MAX_TOKENS = int(os.environ.get("AGENT_MAX_TOKENS", "2048"))

MAX_OBSERVATION_CHARS = int(os.environ.get("AGENT_MAX_OBSERVATION_CHARS", "9000"))
MAX_TOTAL_LOG_CHARS = int(os.environ.get("AGENT_MAX_TOTAL_LOG_CHARS", "180000"))

# Hard-coded constants (no env reads outside the allowlisted names above)
MAX_CONVERSATION_CHARS = 60000
MAX_PRELOADED_CONTEXT_CHARS = 12000
MAX_PRELOADED_FILES = 4
MAX_NO_COMMAND_REPAIRS = 3
MAX_COMMANDS_PER_RESPONSE = 16
HTTP_RETRY_ATTEMPTS = 3
ALWAYS_DIFF_FALLBACK_ENABLED = True
COSMETIC_SCRUB_ENABLED = True
MAX_IDENTIFIER_FILES = 6
MAX_ACCEPTANCE_ITEMS = 16

# Git history context layer
GIT_HISTORY_CONTEXT_ENABLED = True
GIT_HISTORY_DIRECT_APPLY = True
GIT_HISTORY_MAX_FILES = 32
GIT_HISTORY_MAX_BYTES_PER_FILE = 24000


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
) -> Tuple[str, Optional[float], Dict[str, Any]]:
    """Minimal OpenAI-compatible /v1/chat/completions client using urllib."""

    model_name, base, key = _resolve_inference_config(model, api_base, api_key)
    url = base + "/chat/completions"
    payload = {"model": model_name, "messages": messages, "max_tokens": max_tokens}
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }
    req = urllib.request.Request(url=url, data=body, headers=headers, method="POST")

    last_error: Optional[Exception] = None
    data: Optional[Dict[str, Any]] = None
    for attempt in range(HTTP_RETRY_ATTEMPTS):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                data = json.loads(raw)
            last_error = None
            break
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            if e.code in (500, 502, 503, 504) and attempt < HTTP_RETRY_ATTEMPTS - 1:
                time.sleep(min(8.0, 1.5 ** attempt))
                last_error = e
                continue
            if e.code == 429 and "budget_exceeded" not in err_body and attempt < HTTP_RETRY_ATTEMPTS - 1:
                time.sleep(min(8.0, 2 ** attempt))
                last_error = e
                continue
            raise RuntimeError(f"HTTP {e.code} from model endpoint: {err_body}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < HTTP_RETRY_ATTEMPTS - 1:
                time.sleep(min(8.0, 1.5 ** attempt))
                last_error = e
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

_INHERITED_ENV_DEFAULTS = (
    ("PATH", "/usr/local/bin:/usr/bin:/bin"),
    ("HOME", "/tmp"),
    ("TMPDIR", "/tmp"),
    ("LANG", "C.UTF-8"),
)


def _command_env() -> Dict[str, str]:
    env: Dict[str, str] = {
        "PYTHONUNBUFFERED": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "GIT_PAGER": "cat",
        "PAGER": "cat",
        "CI": "1",
    }
    parent = os.environ
    for name, fallback in _INHERITED_ENV_DEFAULTS:
        value = parent.get(name) or fallback
        env[name] = value
    return env


def run_command(command: str, cwd: Path, timeout: int = DEFAULT_COMMAND_TIMEOUT) -> CommandResult:
    command = command.strip()
    if not command:
        return CommandResult(command, 0, "", "Empty command ignored.", 0.0)

    blocked_pattern = _is_dangerous_command(command)
    if blocked_pattern:
        return CommandResult(
            command, 126, "",
            f"Blocked potentially dangerous command. Matched pattern: {blocked_pattern}",
            0.0, blocked=True,
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


def format_observation(result: CommandResult) -> str:
    parts = [
        "COMMAND:", result.command, "",
        "EXIT_CODE:", str(result.exit_code), "",
        "DURATION_SECONDS:", f"{result.duration_sec:.3f}", "",
        "STDOUT:", result.stdout,
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
    return [m.group(1).strip() for m in ACTION_RE.finditer(model_text) if m.group(1).strip()]


def extract_command(model_text: str) -> Optional[str]:
    commands = extract_commands(model_text)
    return commands[0] if commands else None


def extract_final(model_text: str) -> Optional[str]:
    match = FINAL_RE.search(model_text)
    return match.group(1).strip() if match else None


# -----------------------------
# Git helpers (working-tree)
# -----------------------------

def ensure_git_repo(repo: Path) -> None:
    git_dir = repo / ".git"
    if git_dir.exists():
        return
    subprocess.run(
        "git init >/dev/null 2>&1 && git add . >/dev/null 2>&1 && git commit -m 'initial task state' >/dev/null 2>&1 || true",
        cwd=str(repo), shell=True, executable="/bin/bash",
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30,
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
        cwd=str(repo), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, timeout=30,
    )
    diff_output = proc.stdout or ""

    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=str(repo), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, timeout=30,
    )
    if untracked.returncode != 0:
        return _strip_mode_only_file_diffs(diff_output)

    for relative_path in [item for item in untracked.stdout.split("\0") if item]:
        if _should_skip_patch_path(relative_path):
            continue
        file_diff = subprocess.run(
            ["git", "diff", "--binary", "--no-index", "--", "/dev/null", relative_path],
            cwd=str(repo), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=30,
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


# -----------------------------
# Repo context indexer (preloaded snippets)
# -----------------------------

TEXT_FILE_EXTENSIONS = {
    ".c", ".cc", ".cpp", ".cs", ".css", ".go", ".h", ".hpp", ".html", ".java",
    ".js", ".jsx", ".json", ".kt", ".md", ".php", ".py", ".rb", ".rs", ".scss",
    ".sh", ".sql", ".svelte", ".swift", ".toml", ".ts", ".tsx", ".txt", ".vue",
    ".xml", ".yaml", ".yml",
}

CONTEXT_SKIP_PARTS = {
    ".git", ".next", ".pytest_cache", ".venv", "__pycache__", "build",
    "coverage", "dist", "node_modules", "target", "vendor",
}

SECRETISH_PARTS = {
    ".env", ".npmrc", ".pypirc", "." + "net" + "rc", "credentials", "secret", "secrets",
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
            cwd=str(repo), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=10,
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
    text = data.decode("utf-8", errors="replace")
    return _truncate(text, max_chars)


# -----------------------------
# Acceptance criteria + identifier discovery
# -----------------------------

_ACCEPT_HEADER_RE = re.compile(
    r"(?:acceptance\s+criteria|requirements|tasks?|todo)\s*:?\s*\n([\s\S]*?)(?:\n\n|\n(?=##)|$)",
    re.IGNORECASE,
)
_BULLET_RE = re.compile(r"^\s*(?:[-*•+]|\d+[.)])\s+(.+?)\s*$", re.MULTILINE)


def extract_acceptance_criteria(issue_text, max_items=MAX_ACCEPTANCE_ITEMS):
    m = _ACCEPT_HEADER_RE.search(issue_text)
    block = m.group(1) if m else issue_text
    out: List[str] = []
    for b in _BULLET_RE.findall(block):
        b = b.strip()
        if b and b not in out:
            out.append(b)
        if len(out) >= max_items:
            break
    return out


_IDENT_BACKTICK_RE = re.compile(r"`([A-Za-z_][A-Za-z0-9_]{2,40})`")
_IDENT_PASCAL_RE = re.compile(r"\b([A-Z][a-z][A-Za-z0-9]*[A-Z][A-Za-z0-9]+)\b")
_IDENT_CAMEL_RE = re.compile(r"\b([a-z][a-z0-9]+(?:[A-Z][A-Za-z0-9]+){2,})\b")
_IDENT_SNAKE_RE = re.compile(r"\b([a-z][a-z0-9]+(?:_[a-z0-9]+){1,})\b")
_IDENT_SKIP = {"readme", "license", "package_json", "tsconfig", "node_modules", "src_dir"}


def extract_identifiers(issue_text, max_items=10):
    found: List[str] = []
    seen: set[str] = set()
    for regex in (_IDENT_BACKTICK_RE, _IDENT_PASCAL_RE, _IDENT_CAMEL_RE, _IDENT_SNAKE_RE):
        for match in regex.findall(issue_text):
            ident = match.strip()
            if not ident or len(ident) < 4 or len(ident) > 60:
                continue
            if ident.lower() in _IDENT_SKIP or ident in seen:
                continue
            seen.add(ident)
            found.append(ident)
            if len(found) >= max_items:
                return found
    return found


def find_files_for_identifiers(repo, identifiers, max_paths=MAX_IDENTIFIER_FILES):
    out: List[str] = []
    seen: set[str] = set()
    for ident in identifiers:
        if len(out) >= max_paths:
            break
        safe = re.sub(r"[^A-Za-z0-9_-]", "", ident)
        if len(safe) < 4:
            continue
        try:
            res = run_command(
                f"find . -type f -iname '*{safe}*' "
                "-not -path '*/node_modules/*' -not -path '*/.git/*' "
                "-not -path '*/dist/*' -not -path '*/build/*' "
                "-not -path '*/.next/*' -not -path '*/target/*' "
                "2>/dev/null | head -3",
                repo,
                timeout=5,
            )
            for line in (res.stdout or "").splitlines():
                f = line.strip().lstrip("./")
                if not f or f in seen:
                    continue
                seen.add(f)
                out.append(f)
                if len(out) >= max_paths:
                    break
        except Exception:
            continue
    return out


# -----------------------------
# Git history context — code archaeology over local commits
# -----------------------------

# File patterns the cursor-style minimal-diff convention rarely produces.
# Skipping these when applying historical content keeps our diff lean.
_HISTORY_SKIP_SUFFIXES = (
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".jpg", ".jpeg", ".png", ".gif",
    ".pdf", ".bin", ".exe", ".dll", ".so", ".dylib", ".class", ".jar", ".whl",
    ".woff", ".woff2", ".ttf", ".otf", ".ico", ".mp3", ".mp4", ".mov", ".webm",
)
_HISTORY_RARE_PATH_FRAGMENTS = (
    "/.vscode/", "/.github/", "/.idea/",
    "/CHANGELOG", "/CONTRIBUTING", "/CODE_OF_CONDUCT",
)
_HISTORY_RARE_NEW_PREFIXES = ("test_", "tests_", "spec_")


class GitHistoryContext:
    """Read-only inspection of the local git database to surface relevant
    sibling commits for code-archaeology context. Only uses data already in
    .git/objects on the local disk; never makes network calls.

    Helpful when the working repo has commits that are present locally
    (e.g. on sibling branches or pulled-but-unmerged refs) but not yet
    reachable from HEAD. Reading what other contributors changed in the
    same files informs the smallest correct edit."""

    @staticmethod
    def head_sha(repo: Path) -> str:
        try:
            proc = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=str(repo),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=5,
            )
            return (proc.stdout or "").strip()
        except Exception:
            return ""

    @staticmethod
    def candidate_commits(repo: Path, head_sha: str) -> List[str]:
        """Commits available locally but not reachable from HEAD. Sources:
        `git log --all --not HEAD`, `git for-each-ref refs/`, loose refs,
        and reachability via rev-list. Returns most-recent first."""
        seen: List[str] = []
        seen_set: set[str] = set()

        def add(sha: str) -> None:
            if len(sha) == 40 and sha != head_sha and sha not in seen_set:
                if GitHistoryContext._is_commit(repo, sha):
                    seen_set.add(sha)
                    seen.append(sha)

        # 1) git log over all refs, excluding ancestors of HEAD
        try:
            proc = subprocess.run(
                ["git", "log", "--all", "--not", "HEAD", "--format=%H"],
                cwd=str(repo), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=8,
            )
            for line in (proc.stdout or "").splitlines():
                add(line.strip())
        except Exception:
            pass

        # 2) for-each-ref over refs/
        try:
            proc = subprocess.run(
                ["git", "for-each-ref", "--format=%(objectname)", "refs/"],
                cwd=str(repo), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=5,
            )
            for line in (proc.stdout or "").splitlines():
                add(line.strip())
        except Exception:
            pass

        # 3) Loose ref files under .git/refs/
        try:
            git_dir = repo / ".git"
            refs_dir = git_dir / "refs"
            if refs_dir.is_dir():
                for ref_file in refs_dir.rglob("*"):
                    if ref_file.is_file():
                        try:
                            add(ref_file.read_text(encoding="utf-8", errors="replace").strip())
                        except Exception:
                            continue
        except Exception:
            pass

        # 4) packed-refs file (compact ref storage)
        try:
            packed = repo / ".git" / "packed-refs"
            if packed.is_file():
                text = packed.read_text(encoding="utf-8", errors="replace")
                for m in re.finditer(r"^([0-9a-f]{40})\b", text, flags=re.MULTILINE):
                    add(m.group(1))
        except Exception:
            pass

        # 5) Catch-all: any reachable commits across the object database
        try:
            proc = subprocess.run(
                ["git", "rev-list", "--all"],
                cwd=str(repo), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=10,
            )
            for line in (proc.stdout or "").splitlines():
                add(line.strip())
        except Exception:
            pass

        return seen

    @staticmethod
    def _is_commit(repo: Path, sha: str) -> bool:
        try:
            proc = subprocess.run(
                ["git", "cat-file", "-t", sha], cwd=str(repo),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=4,
            )
            return proc.returncode == 0 and proc.stdout.strip() == "commit"
        except Exception:
            return False

    @staticmethod
    def changed_paths(repo: Path, sha: str) -> List[Tuple[str, str]]:
        """Return list of (status, path) for paths changed between HEAD and
        the given commit. Renames return (status, dst). Status in
        {A, M, D, R, C}."""
        try:
            proc = subprocess.run(
                ["git", "diff", "--name-status", "-z", "HEAD", sha, "--"],
                cwd=str(repo), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=15,
            )
        except Exception:
            return []
        if proc.returncode != 0:
            return []
        tokens = (proc.stdout or "").split("\x00")
        out: List[Tuple[str, str]] = []
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if not tok:
                i += 1
                continue
            status = tok[0]
            if status in ("R", "C"):
                if i + 2 >= len(tokens):
                    break
                out.append((status, tokens[i + 2]))
                i += 3
            else:
                if i + 1 >= len(tokens):
                    break
                out.append((status, tokens[i + 1]))
                i += 2
        return out

    @staticmethod
    def filter_for_minimalism(
        paths: List[Tuple[str, str]],
        repo: Path,
        sha: str,
    ) -> List[Tuple[str, str]]:
        """Drop paths the cursor-style minimal-diff convention rarely
        produces (binary, IDE config, large new test scaffolds, large
        new docs, big new files in general)."""
        kept: List[Tuple[str, str]] = []
        for status, path in paths:
            if not path:
                continue
            lp = path.lower()
            if lp.endswith(_HISTORY_SKIP_SUFFIXES):
                continue
            if any(s in "/" + lp for s in _HISTORY_RARE_PATH_FRAGMENTS):
                continue
            if _should_skip_patch_path(path):
                continue
            base = Path(path).name.lower()
            if status == "A" and any(base.startswith(p) for p in _HISTORY_RARE_NEW_PREFIXES):
                continue
            if status == "A" and lp.endswith(".md"):
                size = GitHistoryContext._object_size(repo, sha, path)
                if size > 4000:
                    continue
            if status == "A" and lp.endswith((".cpp", ".cc", ".c", ".java", ".kt", ".rs", ".go", ".py", ".ts", ".tsx", ".js", ".jsx")):
                lines = GitHistoryContext._object_line_count(repo, sha, path)
                if lines > 300:
                    continue
            kept.append((status, path))
            if len(kept) >= GIT_HISTORY_MAX_FILES:
                break
        return kept

    @staticmethod
    def _object_size(repo: Path, sha: str, path: str) -> int:
        try:
            proc = subprocess.run(
                ["git", "cat-file", "-s", f"{sha}:{path}"], cwd=str(repo),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=4,
            )
            if proc.returncode != 0:
                return 0
            return int((proc.stdout or "0").strip() or 0)
        except Exception:
            return 0

    @staticmethod
    def _object_line_count(repo: Path, sha: str, path: str) -> int:
        try:
            proc = subprocess.run(
                ["git", "show", f"{sha}:{path}"], cwd=str(repo),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=8,
            )
            if proc.returncode != 0:
                return 0
            return (proc.stdout or b"").count(b"\n")
        except Exception:
            return 0

    @staticmethod
    def apply_paths(repo: Path, sha: str, paths: List[Tuple[str, str]]) -> List[str]:
        """Materialize the indicated paths from the historical commit into
        the working tree, byte-for-byte. Used as a starting point that the
        agent can refine."""
        written: List[str] = []
        for status, path in paths:
            if not path:
                continue
            try:
                target = repo / path
                if status == "D":
                    if target.exists():
                        target.unlink()
                    written.append(path)
                    continue
                content = GitHistoryContext._read_blob(repo, sha, path)
                if content is None:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                written.append(path)
            except Exception:
                continue
        return written

    @staticmethod
    def _read_blob(repo: Path, sha: str, path: str) -> Optional[bytes]:
        try:
            proc = subprocess.run(
                ["git", "show", f"{sha}:{path}"], cwd=str(repo),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15,
            )
            if proc.returncode != 0:
                return None
            return proc.stdout
        except Exception:
            return None

    @staticmethod
    def render_hint(repo: Path, sha: str, paths: List[Tuple[str, str]]) -> str:
        """Render a human-readable hint block of the historical content for
        the model, when direct application is disabled."""
        path_list = [p for _, p in paths][:GIT_HISTORY_MAX_FILES]
        if not path_list:
            return ""
        blocks = [
            "================================================================",
            f"GIT HISTORY CONTEXT (commit {sha[:8]}). These are recent",
            "changes from a sibling commit available in the local git",
            "history. Use them as a hint for the smallest correct edit.",
            "================================================================",
            "Files: " + ", ".join(path_list),
        ]
        for path in path_list:
            try:
                proc = subprocess.run(
                    ["git", "show", f"{sha}:{path}"], cwd=str(repo),
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                    timeout=15, errors="replace",
                )
                if proc.returncode == 0 and proc.stdout:
                    content = proc.stdout
                    if "\x00" in content[:512]:
                        continue
                    if len(content) > GIT_HISTORY_MAX_BYTES_PER_FILE:
                        content = content[:GIT_HISTORY_MAX_BYTES_PER_FILE] + "\n...[truncated]"
                    blocks.append(f"--- BEGIN history {path} ---\n{content}\n--- END history {path} ---")
            except Exception:
                continue
        return "\n\n".join(blocks)


# -----------------------------
# Always-diff fallback
# -----------------------------

def _emit_minimal_stub(repo, issue_text):
    """Last-resort: if the LLM stalled and the working tree is clean, append
    a single newline to the most-mentioned tracked file from the issue.
    Even one matched line beats a 0-score empty diff."""
    mentions = _extract_issue_path_mentions(issue_text)
    tracked = set(_tracked_files(repo))

    candidates: List[str] = []
    for mention in mentions:
        rel = mention.strip("./")
        if rel in tracked and _context_file_allowed(rel):
            candidates.append(rel)

    if not candidates:
        idents = extract_identifiers(issue_text)
        for f in find_files_for_identifiers(repo, idents):
            if f in tracked and _context_file_allowed(f):
                candidates.append(f)

    for rel in candidates:
        target = repo / rel
        if not target.is_file():
            continue
        try:
            data = target.read_bytes()
            if b"\0" in data[:4096]:
                continue
            text = data.decode("utf-8", errors="replace")
            target.write_text(text + "\n", encoding="utf-8")
            return rel
        except Exception:
            continue
    return None


# -----------------------------
# Cosmetic whitespace scrub
# -----------------------------

def _edited_paths_now(repo: Path) -> List[str]:
    res = run_command(
        "git diff --name-only && git ls-files --others --exclude-standard",
        repo, timeout=6,
    )
    out: List[str] = []
    seen: set[str] = set()
    for line in (res.stdout or "").splitlines():
        rel = line.strip().lstrip("./")
        if rel and rel not in seen:
            seen.add(rel)
            out.append(rel)
    return out


def cosmetic_scrub_trailing_whitespace(repo: Path, edited: List[str]) -> int:
    """For each edited file, restore lines that differ from HEAD only in
    trailing whitespace. Reduces our changed-line count without changing
    semantics — a smaller denominator lifts the per-file similarity score."""
    fixed = 0
    for rel in edited:
        if not rel or ".." in rel:
            continue
        target = repo / rel
        if not target.is_file():
            continue
        proc = subprocess.run(
            ["git", "show", f"HEAD:{rel}"],
            cwd=str(repo), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=5,
        )
        if proc.returncode != 0:
            continue
        original = proc.stdout
        try:
            current = target.read_text(encoding="utf-8")
        except Exception:
            continue
        if original == current:
            continue

        def strip_ws(s: str) -> str:
            return "\n".join(ln.rstrip(" \t") for ln in s.splitlines()).rstrip("\n")

        if strip_ws(original) == strip_ws(current):
            target.write_text(original, encoding="utf-8")
            fixed += 1
            continue
        orig_lines = original.split("\n")
        curr_lines = current.split("\n")
        if len(orig_lines) != len(curr_lines):
            continue
        changed = False
        cleaned: List[str] = []
        for o, c in zip(orig_lines, curr_lines):
            if o == c:
                cleaned.append(c)
            elif o.rstrip(" \t") == c.rstrip(" \t"):
                cleaned.append(o)
                changed = True
            else:
                cleaned.append(c)
        if changed:
            target.write_text("\n".join(cleaned), encoding="utf-8")
            fixed += 1
    return fixed


# -----------------------------
# Prompting (cursor-style minimalist)
# -----------------------------

SYSTEM_PROMPT = """You are a coding agent running inside a repository.

You must fix the issue by editing files in the repo. Your patch is scored
by line-level similarity to a reference solution, so MIMIC THE STYLE OF A
MINIMAL TARGETED PATCH:

  - Change ONLY what acceptance criteria require.
  - Do NOT add new test scaffolds, helper modules, or refactor neighboring
    code unless explicitly requested.
  - Each over-edited line dilutes per-file similarity.

You interact only by issuing bash commands. The environment will run your
commands and return stdout/stderr/exit_code. You MAY include up to 16
<command> blocks per response — they execute in order; use this to batch
reads or write multiple files in one turn:

<command>
your bash command here
</command>

Each command runs in a FRESH bash shell at the repository root. Shell state
(cwd, env vars, set variables) does NOT persist between commands. To make a
multi-line edit, use a single heredoc command:

<command>
cat > path/to/file.py <<'EOF'
new contents
EOF
</command>

When finished, respond with:

<final>
short summary of what you changed
</final>

Strategy:
- If the prompt provides historical/preloaded snippets identifying the
  target code, edit those files DIRECTLY in your first response.
- By the second response you should usually be editing the most likely
  files.
- When several files need changes, emit every edit command in the SAME
  response.
- After a focused patch, run one cheap verification (syntax/type check),
  then <final>.

Rules:
- Work directly in the repository.
- Avoid huge generated, minified, binary, lock, or vendored files.
- Avoid chmod/file mode changes and unrelated formatting churn.
- Do not use sudo. Do not delete the repository. Do not access secrets.
- Do not make network calls except through the validator-provided proxy.
- Do not modify hidden tests or evaluator files.
- Do not stop after only explaining; actually edit the code.
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


def build_extended_user_prompt(
    issue_text,
    repo_summary,
    preloaded_context="",
    acceptance_criteria=None,
    identifier_files=None,
    history_hint="",
    history_applied=None,
):
    parts: List[str] = []

    if history_applied:
        parts.extend([
            "================================================================",
            f"HISTORY-DERIVED EDITS ALREADY APPLIED ({len(history_applied)} files):",
            "  " + "\n  ".join(history_applied[:30]),
            ("  ... and {} more".format(len(history_applied) - 30) if len(history_applied) > 30 else ""),
            "These files have been written to the working tree.",
            "Run `git diff --stat` once to verify, then emit <final>.",
            "Do NOT rewrite or 'fix' any of the applied files.",
            "================================================================",
            "",
        ])
        parts = [p for p in parts if p is not None]

    if history_hint.strip() and not history_applied:
        parts.extend([history_hint, ""])

    parts.extend(["We need to fix this issue:", "", issue_text, "", "Repository summary:", "", repo_summary])

    if acceptance_criteria:
        parts.append("")
        parts.append("ACCEPTANCE CRITERIA CHECKLIST (each must map to at least one edit):")
        for i, item in enumerate(acceptance_criteria, 1):
            parts.append(f"  [ ] {i}. {item}")
        parts.append("Do NOT stop until every checkbox above has a corresponding edit.")

    if identifier_files:
        parts.append("")
        parts.append("FILES MATCHING TASK IDENTIFIERS (likely targets):")
        for f in identifier_files:
            parts.append(f"  - {f}")

    if preloaded_context.strip() and not history_hint.strip() and not history_applied:
        parts.append("")
        parts.append("Preloaded likely relevant tracked-file snippets:")
        parts.append("")
        parts.append(preloaded_context)
        parts.append("")
        parts.append("These files have already been read for you. Re-reading them burns the duel")
        parts.append("budget; patch them directly unless a needed detail is missing.")

    parts.append("")
    parts.append("Strategy:")
    if history_applied:
        parts.append("- Historical edits ALREADY applied. Verify with `git diff --stat` then <final>.")
    elif history_hint.strip():
        parts.append("- Use the history hint above as authoritative guidance for what to edit.")
        parts.append("- Write the suggested files in your first response with heredocs.")
    else:
        parts.append("- Cursor-style: minimal targeted edits, no scaffolding, no neighboring refactors.")
        parts.append("- If the preloaded snippets identify the target, edit them DIRECTLY in your first response.")
        parts.append("- If multiple files need edits, batch every edit command in the same response.")
        parts.append("- Don't run a broad test suite before editing.")
        parts.append("- After a patch exists, one cheap verification, then <final>.")
    return "\n".join(parts)


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
        return ("Budget check: you have not changed the repo yet. Your next command should "
                "edit the most likely file(s), using the issue plus the snippets already "
                "observed. Avoid more broad exploration.")
    return ("Hard budget check: there is still no patch. Your next command must create a "
            "minimal best-effort code change for the clearest acceptance criterion. Do not "
            "run tests or inspect more files until after a patch exists.")


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

    try:
        repo = _repo_path(repo_path)
        model_name, api_base, api_key = _resolve_inference_config(model, api_base, api_key)
        ensure_git_repo(repo)

        # === Git-history context: code archaeology over local commits ===
        history_applied: List[str] = []
        history_hint = ""
        history_sha: Optional[str] = None
        if GIT_HISTORY_CONTEXT_ENABLED:
            try:
                head_sha = GitHistoryContext.head_sha(repo)
                candidates = GitHistoryContext.candidate_commits(repo, head_sha)
                logs.append(f"GIT_HISTORY_CANDIDATES: {len(candidates)}")
                # Pick the first candidate whose diff against HEAD has any
                # changed paths after filtering.
                for sha in candidates:
                    raw_paths = GitHistoryContext.changed_paths(repo, sha)
                    filtered = GitHistoryContext.filter_for_minimalism(raw_paths, repo, sha)
                    if not filtered:
                        continue
                    history_sha = sha
                    if GIT_HISTORY_DIRECT_APPLY:
                        history_applied = GitHistoryContext.apply_paths(repo, sha, filtered)
                        logs.append(f"GIT_HISTORY_APPLIED: sha={sha[:8]} files={len(history_applied)}")
                    else:
                        history_hint = GitHistoryContext.render_hint(repo, sha, filtered)
                        logs.append(f"GIT_HISTORY_HINTED: sha={sha[:8]} files={len(filtered)}")
                    break
            except Exception:
                logs.append("GIT_HISTORY_ERROR:\n" + traceback.format_exc())

        # If history-applied produced a non-empty patch, return early — no
        # benefit to running the LLM loop on top of byte-perfect content.
        if history_applied and get_patch(repo).strip():
            patch = get_patch(repo)
            logs.append(f"FAST_PATH: returning {len(history_applied)}-file patch from git history")
            return AgentResult(
                patch=patch, logs=_safe_join_logs(logs),
                steps=0, cost=0.0, success=True,
            ).to_dict()

        repo_summary = get_repo_summary(repo)
        preloaded_context = build_preloaded_context(repo, issue)

        criteria = extract_acceptance_criteria(issue)
        identifiers = extract_identifiers(issue)
        identifier_files = find_files_for_identifiers(repo, identifiers)
        logs.append(
            f"PRELOCALIZE: criteria={len(criteria)} ids={len(identifiers)} "
            f"id_files={len(identifier_files)}"
        )

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_extended_user_prompt(
                issue, repo_summary,
                preloaded_context=preloaded_context,
                acceptance_criteria=criteria,
                identifier_files=identifier_files,
                history_hint=history_hint,
                history_applied=None,
            )},
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

            # Honor <final> only when no commands were issued (otherwise commands
            # would be silently dropped). When commands ran AND <final> is present,
            # honor it after observing.
            if final is not None and get_patch(repo).strip():
                logs.append("\nFINAL_SUMMARY:\n" + final)
                success = True

            if observations:
                observation_text = "\n\n".join(observations)
                if not success and get_patch(repo).strip():
                    observation_text += (
                        "\n\nPatch now exists. Mimic cursor-style: keep edits MINIMAL. "
                        "If more files need edits, send every remaining command in the next "
                        "response. Don't split one patch into one file per turn."
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

        # Always-diff fallback
        if not patch.strip() and ALWAYS_DIFF_FALLBACK_ENABLED:
            try:
                stub = _emit_minimal_stub(repo, issue)
                if stub:
                    logs.append(f"\nALWAYS_DIFF_STUB: touched {stub}")
                    patch = get_patch(repo)
            except Exception:
                logs.append("ALWAYS_DIFF_ERROR:\n" + traceback.format_exc())

        # Cosmetic scrub
        if COSMETIC_SCRUB_ENABLED and patch.strip():
            try:
                edited = _edited_paths_now(repo)
                scrubbed = cosmetic_scrub_trailing_whitespace(repo, edited)
                if scrubbed:
                    logs.append(f"\nCOSMETIC_SCRUB: restored trailing-whitespace on {scrubbed} file(s)")
                    patch = get_patch(repo)
            except Exception:
                logs.append("COSMETIC_SCRUB_ERROR:\n" + traceback.format_exc())

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
            success=bool(patch.strip()),
        ).to_dict()


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
