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

Architecture:
    The agent is decomposed into a small layered class graph. The free
    function `solve(...)` at the bottom is the validator-contracted entry
    point and is preserved verbatim; everything else is implementation.

        Layer 6 (PUBLIC)        solve()                    <- validator entry
        Layer 5 (ORCHESTRATION) Agent, RefinementOrchestrator
        Layer 4 (GATES)         FinalChecker               <- LLM judge
        Layer 3 (BUILDERS)      ContextBuilder, Prompts
        Layer 2 (DOMAIN)        RepoContext, DiffAnalyzer, IssueAnalyzer
        Layer 1 (PRIMITIVES)    LLMClient, ShellRunner, ActionParser,
                                ObservationInspector, AgentConfig

Miner editing guide:
    You are expected to improve this file. Good areas to edit include
    prompting (Prompts), context gathering (ContextBuilder, IssueAnalyzer),
    command selection / safety (ShellRunner, DANGEROUS_PATTERNS), tool/result
    parsing (ActionParser, ObservationInspector), stopping logic (Agent.run,
    RefinementOrchestrator), patch generation, and how the agent uses its
    step budget.

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

import ast as _ast
import json
import os
import re
import shutil
import subprocess
import sys
import time
import textwrap
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass, replace as _dataclass_replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Config
# -----------------------------

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


# ============================================================================
# Data carriers
# ============================================================================


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


@dataclass
class RefinementTurn:
    """A queued refinement: `marker` is the journal label, `prompt` is the
    corrective user message appended after the assistant's draft."""

    marker: str
    prompt: str


@dataclass
class FinalVerdict:
    """Structured judge result. ``passed`` already incorporates the
    explicit VERDICT plus the score-vs-threshold check, so callers only
    need to read this one bool to decide whether to ship.

    ``score`` is 0\u2013100 (clamped). ``reason`` is the judge's prose
    summary. ``fix_list`` is the list of concrete bullets the judge
    asked the agent to address; empty on PASS, normally non-empty on
    FAIL (an empty fix_list on FAIL is treated as PASS upstream to
    avoid a wasted retry).
    """

    passed: bool
    score: int
    reason: str
    fix_list: List[str]


# ============================================================================
# Layer 1: AgentConfig
# ============================================================================


@dataclass
class AgentConfig:
    """Every tunable knob in one place.

    `from_env()` mirrors the historical module-level DEFAULT_* constants;
    `with_overrides(**kw)` produces a copy with the validator-supplied
    `solve()` arguments slotted in.
    """

    # Inference
    model: str = ""
    api_base: str = ""
    api_key: str = ""
    max_tokens: int = 8192

    # Loop budget
    max_steps: int = 30
    command_timeout: int = 15

    # HTTP retry
    http_max_retries: int = 3
    http_retry_base_backoff: float = 1.0
    max_step_retries: int = 2

    # Char budgets
    max_observation_chars: int = 9000
    max_total_log_chars: int = 180000
    max_conversation_chars: int = 80000
    # Cap for the ranked file-snippets section only. Recent commits get
    # their own ``recent_commit_block_budget`` below; they used to spill
    # over this cap when bundled into a single "preloaded_context" string.
    max_preloaded_context_chars: int = 32000
    max_preloaded_files: int = 10
    max_no_command_repairs: int = 3
    max_commands_per_response: int = 12
    style_hint_budget: int = 600

    # Section toggles for the initial-user prompt assembler. Each flag turns
    # an entire named section on/off so the cost of an experiment is one
    # boolean flip rather than a code edit.
    enable_preloaded_context: bool = True
    enable_recent_commits: bool = True
    # repo_summary safety net: GitHub-mined tasks always start from a clean
    # snapshot so `git status --short` is empty and was producing 11 lines of
    # boilerplate per run. Off by default; flip on if the harness ever stages
    # partially-applied state and we want it surfaced.
    enable_uncommitted_status: bool = False

    # Edit-pressure nudges. Once the step loop reaches `edit_warn_first_step`
    # without any patch on disk, inject a weak nudge; from there on, every
    # `edit_warn_interval` steps inject a strong nudge. So with the defaults
    # (5, 2) the schedule is: weak@5, strong@7, strong@9, strong@11 ...
    edit_warn_first_step: int = 5
    edit_warn_interval: int = 2

    # Refinement is now a single LLM "final check" call. The judge gives a
    # 0\u2013100 SCORE plus VERDICT plus a FIX_LIST; we retry once on FAIL
    # (the judge is treated as advice, not gospel \u2014 a flaky judge should
    # never block shipping). A patch passes only when the judge's explicit
    # VERDICT is PASS *and* the score >= ``final_check_pass_threshold``.
    max_final_check_turns: int = 1
    final_check_pass_threshold: int = 70

    # Recent-commit examples. v1.2 tightens both caps after the v1 default
    # (3500/4500) was filling 4-5k chars with a single drive-by patch whose
    # touched files often had nothing to do with the issue. With (A)
    # relevance filter + (B) issue-aware ranking we now keep one or two
    # *targeted* style anchors instead of two long random ones.
    recent_commit_max_insertions: int = 30
    recent_commit_max_diff_chars: int = 1800
    recent_commit_block_budget: int = 2800

    @classmethod
    def from_env(cls) -> "AgentConfig":
        return cls(
            model=(os.environ.get("AGENT_MODEL") or os.environ.get("NINJA_MODEL", "")).strip(),
            api_base=(
                os.environ.get("AGENT_API_BASE")
                or os.environ.get("NINJA_INFERENCE_BASE_URL")
                or os.environ.get("OPENAI_BASE_URL", "")
            ).strip(),
            api_key=(
                os.environ.get("AGENT_API_KEY")
                or os.environ.get("NINJA_INFERENCE_API_KEY")
                or os.environ.get("OPENAI_API_KEY", "")
            ).strip(),
            max_steps=int(os.environ.get("AGENT_MAX_STEPS", "30")),
            command_timeout=int(os.environ.get("AGENT_COMMAND_TIMEOUT", "15")),
            max_tokens=int(os.environ.get("AGENT_MAX_TOKENS", "8192")),
            max_observation_chars=int(os.environ.get("AGENT_MAX_OBSERVATION_CHARS", "9000")),
            max_total_log_chars=int(os.environ.get("AGENT_MAX_TOTAL_LOG_CHARS", "180000")),
        )

    def with_overrides(self, **kwargs: Any) -> "AgentConfig":
        clean = {k: v for k, v in kwargs.items() if v is not None}
        return _dataclass_replace(self, **clean)


# ============================================================================
# Module-level constants
# ============================================================================
#
# Constants referenced from multiple layers live at module scope rather than
# being duplicated as class fields. Anti-whiff knobs (HTTP retry counts,
# refinement budgets, char limits) live on AgentConfig instead.

# MINER-EDITABLE: shell-command blocklist. You may make it stricter or
# smarter. Do not weaken it to permit destructive host/container operations.
DANGEROUS_PATTERNS: Tuple[str, ...] = (
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
)

TEXT_FILE_EXTENSIONS: frozenset = frozenset({
    # Web/script
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
})

CONTEXT_SKIP_PARTS: frozenset = frozenset({
    ".git", ".next", ".pytest_cache", ".venv", "__pycache__",
    "build", "coverage", "dist", "node_modules", "target", "vendor",
})

SECRETISH_PARTS = {
    ".env",
    ".npmrc",
    ".pypirc",
    ".netrc",
    "credentials",
    "secret",
    "secrets",
}

# v1.2: when ranking recent commits as style anchors, pull this many candidates
# that pass the per-commit filters (insertions, diff size, source-file touched)
# before scoring against the issue. 6 gives us enough headroom that a clearly
# relevant top-2 can outrank pure-recency without paying for a full 20-commit
# pass on every run.
_RECENT_COMMIT_POOL_SIZE: int = 6

_TEST_PARTNER_TEMPLATES: Tuple[Tuple[str, str], ...] = (
    # Python — most common shapes.
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

# ============================================================================
# Generic helpers (used across layers)
# ============================================================================


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


def _resolve_repo_path(path: str | Path) -> Path:
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"repo_path does not exist: {p}")
    if not p.is_dir():
        raise NotADirectoryError(f"repo_path is not a directory: {p}")
    return p


def _shell_quote(value: str) -> str:
    """Single-quote-escape for embedding in a bash command string."""
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _has_executable(name: str) -> bool:
    """True if `name` is on PATH. Uses shutil.which (stdlib).

    The earlier impl invoked `command -v` via subprocess with shell=False, but
    `command` is a bash builtin and not a standalone binary on python:3.11-slim,
    so the subprocess call always raised FileNotFoundError and returned False.
    Net effect: every gate that depended on this check (e.g. JS/TS
    `node --check`, pytest discovery) silently no-op'd in production.
    shutil.which is the portable equivalent.
    """
    try:
        return shutil.which(name) is not None
    except Exception:
        return False


def _context_file_allowed(relative_path: str) -> bool:
    path = Path(relative_path)
    parts_lower = {part.lower() for part in path.parts}
    name_lower = path.name.lower()
    if parts_lower & CONTEXT_SKIP_PARTS:
        return False
    if (
        name_lower.startswith(".env")
        or name_lower in SECRETISH_PARTS
        or parts_lower & SECRETISH_PARTS
    ):
        return False
    if path.suffix.lower() not in TEXT_FILE_EXTENSIONS:
        return False
    return True


# v1.4 path-atom inverted-index helpers. The corpus study showed v1.3's
# token extractors recovered only ~15% of changed files across the
# workspace tasks; the remaining ~85% live in tasks where the LLM
# described changes in prose without naming specific identifiers, but
# whose changed-file *paths* still contain the domain words the issue
# repeatedly mentions (test4: ``Pos`` / ``Customer`` / ``Service`` /
# ``Role`` / ``Discount`` / ``Booking``, etc.). v1.4 attacks that by
# breaking each tracked path into lowercased ≥3-char atoms and scoring
# overlap with the issue's token atoms.

# Atoms that appear in nearly every path of every repo and therefore
# convey no domain signal. Kept short on purpose — false-positive
# overlap on a single generic atom only adds a +8 / file (roughly the
# per-term weight today), which is the noise floor we already accept.
_PATH_ATOM_STOP: frozenset = frozenset({
    "src", "app", "lib", "pkg", "pkgs", "cmd", "cmds", "main", "index",
    "init", "common", "shared", "core", "tools",
    "public", "private", "protected", "static",
})


def _atomize(token: str) -> List[str]:
    """Split a single name (path segment, identifier, term) into lowercased
    atoms ≥3 chars.

    ``PartnershipContractContent`` → ``["partnership", "contract", "content"]``
    ``senior_discount_percent``    → ``["senior", "discount", "percent"]``
    ``user-agent``                 → ``["user", "agent"]``
    ``api.helpers.ts``             → ``["api", "helpers"]`` (ext stripped via
    the non-letter split below; the caller handles extensions separately).

    The CamelCase split keeps adjacent uppercase runs as one atom
    (``XMLHttpRequest`` → ``["xml", "http", "request"]``) so an issue
    word like ``http`` matches the ``Http`` part of a path.
    """
    if not token:
        return []
    out: List[str] = []
    # Split on every non-alphanumeric so snake / kebab / dot all collapse.
    for chunk in re.split(r"[^A-Za-z0-9]+", token):
        if not chunk:
            continue
        # CamelCase split inside the chunk. Order matters:
        #   1. acronym runs followed by a lowercase letter
        #   2. PascalCase / camelCase (first cap + lowers)
        #   3. all-lowercase fallback
        for sub in re.findall(
            r"[A-Z]+(?=[A-Z][a-z])|[A-Z][a-z]+|[A-Z]+|[a-z]+|[0-9]+",
            chunk,
        ):
            sub_lower = sub.lower()
            if len(sub_lower) < 3:
                continue
            out.append(sub_lower)
    return out


def _atoms_match(a: str, b: str) -> bool:
    """Soft equality used by the path-atom overlap scorer.

    Strict equality plus a substring-match fallback when the shorter atom
    is ≥4 chars. Handles common plural / suffix differences cheaply
    (``service`` ↔ ``services``, ``user`` ↔ ``users``,
    ``auth`` ↔ ``authentication``) without a stemmer. The 4-char floor
    blocks accidental matches like ``pos`` ↔ ``position``.
    """
    if a == b:
        return True
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if len(short) < 4:
        return False
    return short in long


def _split_path_parts(relative_path: str) -> List[str]:
    """Decompose a tracked-file path into its domain atoms.

    Keeps directory atoms and filename-stem atoms; drops the suffix
    (extensions like ``.tsx`` / ``.php`` carry no domain signal). Atoms
    in ``_PATH_ATOM_STOP`` (``src`` / ``app`` / ``main`` / ``index`` /
    ``public`` / ``private`` / etc.) are filtered out so they don't
    inflate the overlap-count denominator.
    """
    if not relative_path:
        return []
    p = Path(relative_path)
    out: List[str] = []
    for segment in p.parts[:-1]:
        out.extend(_atomize(segment))
    out.extend(_atomize(p.stem))
    return [a for a in out if a not in _PATH_ATOM_STOP]


def _parse_shortstat_insertions(stat_output: str) -> int:
    """Pull the insertion count out of ``git show --shortstat`` output.

    The shortstat line looks like ``" 3 files changed, 17 insertions(+), 4
    deletions(-)"``. We only need the insertion count for the
    recent-commit examples filter, so we keep a tiny dedicated parser
    instead of reaching for ``git show --numstat`` which would force one
    line per file. Returns 0 when no insertion clause is present
    (deletion-only commits, parse failures, empty input).
    """
    for line in stat_output.splitlines():
        if "insertion" not in line:
            continue
        for word in line.split(","):
            if "insertion" not in word:
                continue
            try:
                return int(word.strip().split()[0])
            except (ValueError, IndexError):
                return 0
        return 0
    return 0


# ============================================================================
# Layer 1: LLMClient
# ============================================================================

# MINER-EDITABLE WITH BOUNDARIES: You may change request formatting, retry
# behavior, response parsing, or model-message strategy here. Keep all
# requests pointed at the api_base/api_key supplied by solve(); the validator
# proxy rewrites the model and sampling parameters server-side.
class LLMClient:
    """OpenAI-compatible /v1/chat/completions client.

    Owns retry/backoff, payload shaping, and response parsing. Constructed
    once per attempt with the validator-resolved (model, api_base, api_key);
    never reads module globals at call time.

    Retries with exponential backoff on transient transport failures
    (timeout, connection reset, HTTP 5xx, HTTP 429). Client-side 4xx (other
    than 429) bail out immediately because retrying won't change the outcome.
    """

    def __init__(self, config: AgentConfig):
        model, base, key = self._resolve(config.model, config.api_base, config.api_key)
        self.model = model
        self.api_base = base
        self.api_key = key
        self.max_tokens = config.max_tokens
        self.max_retries = config.http_max_retries
        self.base_backoff = config.http_retry_base_backoff
        self.max_conversation_chars = config.max_conversation_chars

    @staticmethod
    def _normalize_api_base(api_base: str) -> str:
        base = api_base.rstrip("/")
        if base.endswith("/chat/completions"):
            return base[: -len("/chat/completions")]
        if base.endswith("/v1"):
            return base
        return base + "/v1"

    @classmethod
    def _resolve(cls, model: str, api_base: str, api_key: str) -> Tuple[str, str, str]:
        m = (model or "").strip()
        b = (api_base or "").strip()
        k = (api_key or "").strip()
        if not m:
            raise ValueError(
                "model is required; validators must pass the centrally managed model id"
            )
        if not b:
            raise ValueError(
                "api_base is required; validators must pass the managed inference proxy URL"
            )
        if not k:
            raise ValueError(
                "api_key is required; validators must pass the per-run proxy token"
            )
        return m, cls._normalize_api_base(b), k

    @staticmethod
    def _message_chars(messages: List[Dict[str, str]]) -> int:
        return sum(len(m.get("content") or "") + 32 for m in messages)

    def trim_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Drop oldest non-system messages once the conversation exceeds the
        char budget. The system prompt and the first user prompt are kept."""
        if self._message_chars(messages) <= self.max_conversation_chars:
            return messages

        head = messages[:2]
        tail: List[Dict[str, str]] = []
        budget = max(8000, self.max_conversation_chars - self._message_chars(head) - 400)
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

    def complete(
        self,
        messages: List[Dict[str, str]],
        timeout: int = 120,
    ) -> Tuple[str, Optional[float], Dict[str, Any]]:
        """Send messages, return (content, cost, raw response)."""
        url = self.api_base + "/chat/completions"
        payload = {
            "model": self.model,
            "messages": self.trim_messages(messages),
            "max_tokens": self.max_tokens,
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        data: Optional[Dict[str, Any]] = None
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(url=url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
                    data = json.loads(raw)
                break
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="replace")
                retryable = (500 <= e.code < 600) or e.code == 429
                if retryable and attempt < self.max_retries:
                    last_error = e
                    time.sleep(self.base_backoff * (2 ** attempt))
                    continue
                raise RuntimeError(f"HTTP {e.code} from model endpoint: {err_body}") from e
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                if attempt < self.max_retries:
                    last_error = e
                    time.sleep(self.base_backoff * (2 ** attempt))
                    continue
                raise RuntimeError(f"Model request failed: {e}") from e
            except json.JSONDecodeError as e:
                if attempt < self.max_retries:
                    last_error = e
                    time.sleep(self.base_backoff * (2 ** attempt))
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


# ============================================================================
# Layer 1: ShellRunner
# ============================================================================

# MINER-EDITABLE: This is the bash tool surface your agent uses inside the
# task repo. You may improve command validation, environment handling,
# timeouts, and output shaping. Keep commands scoped to the repo and avoid
# secrets or network access outside the validator inference proxy.
class ShellRunner:
    """Validator-scoped bash surface.

    Owns the dangerous-command filter, env scrubbing, timeout handling, and
    observation formatting. The agent's only interaction with the host
    filesystem (besides git) goes through `run`.
    """

    DANGEROUS_PATTERNS: Tuple[str, ...] = DANGEROUS_PATTERNS

    def __init__(self, repo: Path, config: AgentConfig):
        self.repo = repo
        self.default_timeout = config.command_timeout
        self.max_observation_chars = config.max_observation_chars

    @classmethod
    def is_dangerous(cls, command: str) -> Optional[str]:
        lowered = command.strip()
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, lowered):
                return pattern
        return None

    @staticmethod
    def sanitised_env() -> Dict[str, str]:
        return _command_env()

    def run(self, command: str, timeout: Optional[int] = None) -> CommandResult:
        command = command.strip()
        if not command:
            return CommandResult(
                command=command,
                exit_code=0,
                stdout="",
                stderr="Empty command ignored.",
                duration_sec=0.0,
            )

        blocked_pattern = self.is_dangerous(command)
        if blocked_pattern:
            return CommandResult(
                command=command,
                exit_code=126,
                stdout="",
                stderr=f"Blocked potentially dangerous command. Matched pattern: {blocked_pattern}",
                duration_sec=0.0,
                blocked=True,
            )

        deadline = timeout if timeout is not None else self.default_timeout
        start = time.time()
        try:
            proc = subprocess.run(
                command,
                cwd=str(self.repo),
                shell=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=deadline,
                executable="/bin/bash",
                env=self.sanitised_env(),
            )
            return CommandResult(
                command=command,
                exit_code=proc.returncode,
                stdout=_truncate(proc.stdout or "", self.max_observation_chars),
                stderr=_truncate(proc.stderr or "", self.max_observation_chars),
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
                stdout=_truncate(stdout, self.max_observation_chars),
                stderr=_truncate(
                    stderr + f"\nCommand timed out after {deadline}s.",
                    self.max_observation_chars,
                ),
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

    @staticmethod
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


# ============================================================================
# Layer 1: ActionParser
# ============================================================================


class ActionParser:
    """Extracts <command> and <final> blocks from model responses."""

    ACTION_RE = re.compile(r"<command>\s*(.*?)\s*</command>", re.IGNORECASE | re.DOTALL)
    FINAL_RE = re.compile(r"<final>\s*(.*?)\s*</final>", re.IGNORECASE | re.DOTALL)

    @classmethod
    def extract_commands(cls, model_text: str) -> List[str]:
        return [
            match.group(1).strip()
            for match in cls.ACTION_RE.finditer(model_text)
            if match.group(1).strip()
        ]

    @classmethod
    def extract_command(cls, model_text: str) -> Optional[str]:
        commands = cls.extract_commands(model_text)
        return commands[0] if commands else None

    @classmethod
    def extract_final(cls, model_text: str) -> Optional[str]:
        match = cls.FINAL_RE.search(model_text)
        if not match:
            return None
        return match.group(1).strip()


# ============================================================================
# Layer 1: ObservationInspector
# ============================================================================


class ObservationInspector:
    """Static helpers for inspecting CommandResult / observation strings.

    Used by the agent's auto-stop heuristics: did the command look like a
    successful test run? A patch review (git diff/status)? Etc.
    """

    _VERIFICATION_PATTERNS: Tuple[str, ...] = (
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
    )

    _BAD_MARKERS: Tuple[str, ...] = (
        " failed", " failures", " error", " errors", "traceback",
        "assertionerror", "syntaxerror", "exception",
    )

    _GOOD_MARKERS: Tuple[str, ...] = (" passed", " all passed", "ok", "success")

    @classmethod
    def is_verification_command(cls, command: str) -> bool:
        lowered = command.lower()
        return any(re.search(pattern, lowered) for pattern in cls._VERIFICATION_PATTERNS)

    @classmethod
    def looks_like_successful_test(cls, observation: str, command: str = "") -> bool:
        lower = observation.lower()
        exit_code = cls._extract_exit_code(lower)
        stderr_body = cls._extract_section(lower, "stderr")

        if exit_code is not None and exit_code != 0:
            return False

        has_good = any(marker in lower for marker in cls._GOOD_MARKERS)
        has_bad = any(marker in lower for marker in cls._BAD_MARKERS)
        if stderr_body and any(marker in stderr_body for marker in cls._BAD_MARKERS):
            has_bad = True

        if exit_code == 0 and cls.is_verification_command(command) and not has_bad:
            return True

        return (exit_code == 0 or has_good) and has_good and not has_bad

    @staticmethod
    def is_patch_review(command: str, result: CommandResult) -> bool:
        if result.exit_code != 0:
            return False
        lowered = command.lower().strip()
        return bool(
            re.search(r"\bgit\s+(diff|status)\b", lowered)
            or re.search(r"\bgit\s+show\s+--stat\b", lowered)
        )

    @staticmethod
    def _extract_exit_code(observation_lower: str) -> Optional[int]:
        match = re.search(r"(?m)^exit_code:\n(-?\d+)", observation_lower)
        if not match:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    @staticmethod
    def _extract_section(observation_lower: str, section: str) -> str:
        match = re.search(
            rf"(?ms)^{re.escape(section.lower())}:\n(.*?)(?:\n[a-z_]+:\n|\Z)",
            observation_lower,
        )
        return match.group(1).strip() if match else ""


# ============================================================================
# Layer 2: DiffAnalyzer
# ============================================================================


class DiffAnalyzer:
    """Pure-functional unified-diff inspector. Hunk hygiene + path extraction.

    No I/O, no repo handle \u2014 every method takes a diff string. Used at
    patch-return time to drop low-signal hunks (whitespace-only,
    comment-only, blank-line-only) before the validator sees the diff.
    """

    _COMMENT_LINE_PREFIXES: Tuple[str, ...] = ("#", "//", ";", "--", "%")
    _BLOCK_COMMENT_RE = re.compile(r"^\s*(\*|/\*|\*/)")

    @classmethod
    def line_is_comment(cls, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        if any(stripped.startswith(p) for p in cls._COMMENT_LINE_PREFIXES):
            return True
        if cls._BLOCK_COMMENT_RE.match(line):
            return True
        if stripped.startswith('"""') or stripped.startswith("'''"):
            return True
        return False

    @staticmethod
    def hunk_is_blank_only(added: List[str], removed: List[str]) -> bool:
        body = [line for line in added + removed if line.strip()]
        return not body and bool(added or removed)

    @staticmethod
    def hunk_is_whitespace_only(added: List[str], removed: List[str]) -> bool:
        if not added and not removed:
            return False
        a = sorted(line.strip() for line in added if line.strip())
        r = sorted(line.strip() for line in removed if line.strip())
        if not a and not r:
            return True
        return a == r

    @classmethod
    def hunk_is_comment_only(cls, added: List[str], removed: List[str]) -> bool:
        body = [line for line in added + removed if line.strip()]
        if not body:
            return False
        return all(cls.line_is_comment(line) for line in body)

    @staticmethod
    def strip_mode_only_blocks(diff_output: str) -> str:
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

    @classmethod
    def strip_low_signal_hunks(cls, diff_output: str) -> str:
        """Drop blank-only / whitespace-only / comment-only hunks per file.

        Whole-file blocks with no @@ markers are kept verbatim because they
        are file-create / file-delete / binary patches that the hunk
        classifier can't reason about.
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
                    cls.hunk_is_blank_only(added, removed)
                    or cls.hunk_is_whitespace_only(added, removed)
                    or cls.hunk_is_comment_only(added, removed)
                ):
                    continue
                substantive.append(hunk_text)
            if substantive:
                out.append(header + "".join(substantive))
        result = "".join(out)
        if diff_output.endswith("\n") and result and not result.endswith("\n"):
            result += "\n"
        return result

    @staticmethod
    def changed_files(patch: str) -> List[str]:
        """Return the list of `b/` paths touched by a unified diff, in order."""
        seen: List[str] = []
        for match in re.finditer(r"^diff --git a/(.+?) b/(.+?)$", patch, flags=re.MULTILINE):
            path = match.group(2)
            if path and path not in seen:
                seen.append(path)
        return seen


# ============================================================================
# Layer 2: IssueAnalyzer
# ============================================================================


class IssueAnalyzer:
    """Pure-functional issue text mining.

    Lazy-cached properties so each derived list is computed once per attempt:
    `path_mentions`, `terms`, `symbols`, plus the typed symbol pools used
    by ContextBuilder for relevance ranking.
    """

    _PATH_MENTION_RE = re.compile(
        r"(?<![\w.-])([\w./-]+\.(?:c|cc|cpp|cs|css|go|h|hpp|html|java|js|jsx|json|kt|md|php|py|rb|rs|scss|sh|sql|svelte|swift|toml|ts|tsx|txt|vue|xml|ya?ml))(?![\w.-])",
        re.IGNORECASE,
    )

    _TERM_STOP: frozenset = frozenset({
        "about", "after", "also", "before", "change", "code", "file", "from",
        "have", "issue", "make", "need", "should", "that", "their", "there",
        "this", "update", "using", "when", "with",
    })

    # v1.3: typed symbol-pool extraction. The single-regex / cap-12 design in
    # v1.2 caught only ~9% of changed files in the workspace task corpus
    # because every "interesting" identifier had to fight for one of 12
    # slots. We now extract several pools in parallel, each with its own
    # regex envelope (backtick spans, paren tokens, PascalCase, snake_case,
    # quoted strings, CLI flags) and let downstream ranking weight them by
    # the typographic confidence each envelope conveys.
    _BACKTICK_SPAN_RE = re.compile(r"`([^`\n]+)`")
    # Paren spans are restricted to **no-whitespace** payloads so the pool
    # stays high-precision. Allowing spaces matched whole sub-clauses like
    # ``(which uses DigitalPicture interface)`` and the resulting
    # split-to-identifiers pulled in English filler (`which`, `uses`,
    # `interface`). Restricting to ``[^\s()]`` matches only the
    # ``(EntityList.vue)`` / ``(CropModal)`` / ``(Foo)`` shape that the
    # corpus study showed as 25% directly file-matching.
    _PAREN_SPAN_RE    = re.compile(r"\(([^\s()]{1,60})\)")
    _DQ_SPAN_RE       = re.compile(r'"([^"\n]{2,40})"')
    _SQ_SPAN_RE       = re.compile(r"'([^'\n]{2,40})'")
    _KEBAB_FLAG_RE    = re.compile(r"(--[a-z][a-z0-9-]+)")
    # PascalCase: covers both camel-bounded forms (TestPicture17) and
    # single-Cap nouns ≥3 chars (Picture, Pixel, Restore). The second
    # alternative requires at least one lowercase boundary after the cap so
    # all-caps acronyms (URL, API) are not double-counted (they almost
    # always show up unquoted in prose where they would be noise).
    _PASCAL_RE = re.compile(
        r"\b([A-Z][a-z]+(?:[A-Z][A-Za-z0-9]*)+|[A-Z][A-Z0-9]*[a-z][A-Za-z0-9]*)\b"
    )
    _SNAKE_RE = re.compile(r"\b([a-z]+(?:_[a-z0-9]+)+)\b")
    _IDENT_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")

    # PascalCase stoplist — sentence starters / English nouns / pronouns
    # that the corpus study showed as the dominant noise source for the
    # pascal pool. Words that double as legitimate code identifiers (Build,
    # Header, Request, Response, Service, Settings, ...) are intentionally
    # NOT here; the per-pool grep weight + dedup handles their precision
    # penalty downstream.
    _PASCAL_STOP: frozenset = frozenset({
        # Articles / determiners
        "The", "This", "That", "These", "Those", "Each", "Every",
        "Some", "Any", "Most", "All", "Other", "Another",
        # Conjunctions / connectives at sentence start
        "When", "If", "Then", "Otherwise", "Important", "Note", "Notes",
        "First", "Second", "Third", "Final", "Last", "Next", "Previous",
        "Currently", "However", "Additionally", "Finally", "Meanwhile",
        "Furthermore", "Moreover", "Therefore", "Hence", "Thus",
        "Initially", "Eventually", "Subsequently", "Recently", "Now",
        "After", "Before", "Since", "Until", "While", "During",
        # English booleans / null
        "True", "False", "None", "Yes", "No",
        # Common verbs at sentence start
        "Add", "Remove", "Get", "Set", "Use", "Make", "Send", "Save",
        "Show", "Hide", "Click", "Choose", "Update", "Create", "Delete",
        "Returns", "Return", "Replace", "Restore", "Implement",
        "Ensure", "Confirm", "Allow", "Apply", "Choose", "Pick",
        "Read", "Write", "Open", "Close",
        # Pronouns
        "I", "We", "You", "They", "He", "She", "It", "Us", "Them",
        "Your", "Our", "Their", "His", "Her", "Its",
        # Section labels common in LLM-generated specs
        "Acceptance", "Background", "Context", "Goal", "Goals",
        "Summary", "Overview", "Description", "Details", "Major",
        "Minor", "Per", "Hey", "Existing", "New", "Old",
    })

    # Per-pool extraction caps. Total worst-case ≈ 86 unique symbols, deduped
    # across pools before grep so each unique symbol is git-grep'd at most
    # once. The 12-cap on the legacy ``symbols`` property survives so
    # ``recent_commit_examples`` scoring stays cheap.
    _MAX_BACKTICK = 20
    _MAX_PAREN    = 10
    _MAX_PASCAL   = 24
    _MAX_SNAKE    = 16
    _MAX_STRING   = 16

    # Legacy union cap kept at 12 so ``recent_commit_examples`` scoring (the
    # only remaining consumer of ``IssueAnalyzer.symbols``) does not balloon
    # its per-commit substring scan. The new typed pools use their own caps.
    _MAX_SYMBOLS = 12

    def __init__(self, issue_text: str):
        self.text = issue_text or ""
        self._path_mentions: Optional[List[str]] = None
        self._terms: Optional[List[str]] = None
        self._symbols: Optional[List[str]] = None
        self._backtick_symbols: Optional[List[str]] = None
        self._paren_symbols: Optional[List[str]] = None
        self._pascal_symbols: Optional[List[str]] = None
        self._snake_symbols: Optional[List[str]] = None
        self._string_literals: Optional[List[str]] = None
        self._issue_atoms: Optional[frozenset] = None

    @property
    def path_mentions(self) -> List[str]:
        if self._path_mentions is None:
            mentions: List[str] = []
            for match in self._PATH_MENTION_RE.finditer(self.text):
                value = match.group(1).strip("`'\"()[]{}:,;")
                if value and value not in mentions:
                    mentions.append(value)
            self._path_mentions = mentions
        return self._path_mentions

    @property
    def terms(self) -> List[str]:
        # v1.3: cap lowered 40 → 20. The corpus study showed 39% of v1.2's
        # ``terms`` were pure English noise that produced a faint score
        # boost on every file containing the word; tightening the cap
        # halves that noise floor without losing real path-substring
        # signal (the top-ranked terms are typically domain nouns).
        if self._terms is None:
            seen: List[str] = []
            for raw in re.findall(r"[A-Za-z_][A-Za-z0-9_-]{2,}", self.text.lower()):
                if raw in self._TERM_STOP or raw in seen:
                    continue
                seen.append(raw)
            self._terms = seen[:20]
        return self._terms

    @classmethod
    def _split_to_idents(cls, value: str) -> List[str]:
        """Split a multi-token span (e.g. backtick body) into identifier-like
        sub-tokens. ``mirrorVertical(Picture source)`` becomes ``["mirrorVertical",
        "Picture", "source"]``; we then drop tokens shorter than 3 chars.
        """
        return [t for t in cls._IDENT_TOKEN_RE.findall(value) if len(t) >= 3]

    def _extract_pool(
        self,
        regex: "re.Pattern[str]",
        max_count: int,
        *,
        split: bool = False,
        stop: Optional[frozenset] = None,
    ) -> List[str]:
        """Pull tokens captured by ``regex.group(1)`` from the issue text.

        Shared engine for the typed pools so each property is one line of
        configuration instead of a copy-pasted loop. ``split`` runs the
        body through ``_split_to_idents`` (used for backtick / paren
        spans whose payload is an arbitrary phrase). ``stop`` is the
        per-pool stoplist applied to the final tokens, not the raw span.
        """
        seen: set = set()
        out: List[str] = []
        for match in regex.finditer(self.text):
            value = (match.group(1) or "").strip()
            if not value:
                continue
            candidates = self._split_to_idents(value) if split else [value]
            for cand in candidates:
                if cand in seen:
                    continue
                if len(cand) < 3:
                    continue
                if stop and cand in stop:
                    continue
                seen.add(cand)
                out.append(cand)
                if len(out) >= max_count:
                    return out
        return out

    @property
    def backtick_symbols(self) -> List[str]:
        """Identifier-like tokens lifted from `code` spans in the issue.

        Highest-precision symbol pool: 100% of backtick tokens in the
        corpus study appeared somewhere in the reference patch. Backtick
        bodies frequently contain whole expressions (``mirrorVertical(Picture
        source)``) so the body is split into sub-identifiers before dedup.
        """
        if self._backtick_symbols is None:
            self._backtick_symbols = self._extract_pool(
                self._BACKTICK_SPAN_RE, self._MAX_BACKTICK, split=True
            )
        return self._backtick_symbols

    @property
    def paren_symbols(self) -> List[str]:
        """Identifier-like tokens from ``(parenthesised)`` spans.

        Highest "useful%" of any new pool in the corpus study (25% of
        paren tokens directly matched a changed file's stem/basename).
        Examples: ``(EntityList.vue)``, ``(CropModal)`` — the LLM
        consistently uses parens to clarify which file/module it means.
        """
        if self._paren_symbols is None:
            self._paren_symbols = self._extract_pool(
                self._PAREN_SPAN_RE, self._MAX_PAREN, split=True
            )
        return self._paren_symbols

    @property
    def pascal_symbols(self) -> List[str]:
        """PascalCase tokens from the issue prose.

        Highest absolute file-match count of any pool in the corpus study
        (17 stem matches across the 11 tasks vs. 1 for ``cur:path_mentions``).
        Bulk pool — wider net than backtick / paren but with a small
        sentence-starter stoplist to keep noise in check.
        """
        if self._pascal_symbols is None:
            self._pascal_symbols = self._extract_pool(
                self._PASCAL_RE, self._MAX_PASCAL, stop=self._PASCAL_STOP
            )
        return self._pascal_symbols

    @property
    def snake_symbols(self) -> List[str]:
        """snake_case tokens (Python functions, DB columns, CLI subcommands)."""
        if self._snake_symbols is None:
            self._snake_symbols = self._extract_pool(
                self._SNAKE_RE, self._MAX_SNAKE
            )
        return self._snake_symbols

    @property
    def string_literals(self) -> List[str]:
        """Quoted UI strings + CLI flag tokens.

        Combines double-quoted spans, single-quoted spans, and ``--kebab``
        flags. Used as a separate grep pool: literals match against
        verbatim string contents in source (e.g. ``"Restore"`` matches
        a button-label in TSX), which is a different signal from
        identifier-shaped grep.
        """
        if self._string_literals is None:
            seen: set = set()
            out: List[str] = []
            for regex in (self._DQ_SPAN_RE, self._SQ_SPAN_RE, self._KEBAB_FLAG_RE):
                for match in regex.finditer(self.text):
                    value = (match.group(1) or "").strip()
                    if not value or len(value) < 2:
                        continue
                    if value in seen:
                        continue
                    seen.add(value)
                    out.append(value)
                    if len(out) >= self._MAX_STRING:
                        break
                if len(out) >= self._MAX_STRING:
                    break
            self._string_literals = out
        return self._string_literals

    @property
    def symbol_pools(self) -> Dict[str, List[str]]:
        """Convenience accessor for ranking code: a dict of every typed pool.

        Order is preserved (``backtick``, ``paren``, ``pascal``, ``snake``,
        ``string``) so iterators that dedupe by first-occurrence give the
        highest-precision pool the first claim on each token.
        """
        return {
            "backtick": self.backtick_symbols,
            "paren":    self.paren_symbols,
            "pascal":   self.pascal_symbols,
            "snake":    self.snake_symbols,
            "string":   self.string_literals,
        }

    @property
    def symbols(self) -> List[str]:
        """Backwards-compat union of identifier-shaped tokens.

        v1.2 used a single regex + ``_SYMBOL_STOP`` filter capped at 12. v1.3
        rebuilds the property as a dedup'd union of the typed pools above
        (still capped at 12) so the only remaining consumer
        (``RepoContext._score_commit_for_issue``) gets richer extraction
        for free without us having to update its scoring formula.
        """
        if self._symbols is None:
            seen: set = set()
            out: List[str] = []
            for pool_key in ("backtick", "paren", "pascal", "snake"):
                for token in self.symbol_pools[pool_key]:
                    if token in seen:
                        continue
                    seen.add(token)
                    out.append(token)
                    if len(out) >= self._MAX_SYMBOLS:
                        break
                if len(out) >= self._MAX_SYMBOLS:
                    break
            self._symbols = out
        return self._symbols

    @property
    def issue_atoms(self) -> frozenset:
        """Domain atoms derived from every typed pool plus ``terms``.

        v1.4 lever (1) + (2): the path-atom inverted-index ranker uses
        this set to score how many of the issue's domain words appear
        inside a tracked file's path atoms (directory + stem segments,
        camel/snake/kebab-split, ≥3 chars). Examples of what ends up in
        the set for a test4-shaped issue: ``pos``, ``customer``,
        ``service``, ``role``, ``permission``, ``unit``, ``conversion``,
        ``delivery``, ``discount``, ``cashier``, ``catalog``,
        ``variant``, ``slug``, ``booking``, ``fulfillment``.

        Sourced from (in order, highest-confidence first):
        ``path_mentions``, ``backtick_symbols``, ``paren_symbols``,
        ``pascal_symbols``, ``snake_symbols``, ``string_literals``,
        ``terms``. Each contributing token is run through ``_atomize``,
        the union is taken, and entries in ``_PATH_ATOM_STOP`` are
        dropped. Cached once per ``IssueAnalyzer`` lifetime; the result
        is a ``frozenset`` so the ranking inner loop can use ``in``
        without copying.
        """
        if self._issue_atoms is None:
            atoms: set = set()
            sources = [
                self.path_mentions,
                self.backtick_symbols,
                self.paren_symbols,
                self.pascal_symbols,
                self.snake_symbols,
                self.string_literals,
                self.terms,
            ]
            for source in sources:
                for token in source:
                    atoms.update(_atomize(token))
            atoms = {a for a in atoms if a not in _PATH_ATOM_STOP}
            self._issue_atoms = frozenset(atoms)
        return self._issue_atoms

# ============================================================================
# Layer 2: RepoContext
# ============================================================================


class RepoContext:
    """Filesystem + git facade.

    Owns:
      - tracked-file enumeration (cached per attempt)
      - patch generation with low-signal hygiene applied
      - safe in-repo file reads
      - recent-commit examples for in-context style anchors
      - symbol-grep ranking signal for ContextBuilder
    """

    def __init__(self, repo: Path, shell: ShellRunner, config: AgentConfig):
        self.repo = repo
        self.shell = shell
        self.config = config
        self._tracked_cache: Optional[List[str]] = None

    # ---- low-level git ----

    def ensure_git_repo(self) -> None:
        git_dir = self.repo / ".git"
        if git_dir.exists():
            return
        subprocess.run(
            "git init >/dev/null 2>&1 && "
            "git add . >/dev/null 2>&1 && "
            "git commit -m 'initial task state' >/dev/null 2>&1 || true",
            cwd=str(self.repo),
            shell=True,
            executable="/bin/bash",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )

    # ---- patch generation ----

    @staticmethod
    def _should_skip_patch_path(relative_path: str) -> bool:
        path = Path(relative_path)
        if path.suffix == ".pyc":
            return True
        return any(
            part in {"__pycache__", ".pytest_cache", "node_modules", ".git"}
            for part in path.parts
        )

    def get_patch(self) -> str:
        exclude_pathspecs = [
            ":(exclude,glob)**/*.pyc",
            ":(exclude,glob)**/__pycache__/**",
            ":(exclude,glob)**/.pytest_cache/**",
            ":(exclude,glob)**/node_modules/**",
            ":(exclude).git",
        ]
        proc = subprocess.run(
            ["git", "diff", "--binary", "--", ".", *exclude_pathspecs],
            cwd=str(self.repo),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
        diff_output = proc.stdout or ""

        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=str(self.repo),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
        if untracked.returncode != 0:
            return diff_output

        for relative_path in [item for item in untracked.stdout.split("\0") if item]:
            if self._should_skip_patch_path(relative_path):
                continue
            file_diff = subprocess.run(
                ["git", "diff", "--binary", "--no-index", "--", "/dev/null", relative_path],
                cwd=str(self.repo),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30,
            )
            if file_diff.returncode in (0, 1):
                diff_output += file_diff.stdout or ""

        cleaned = DiffAnalyzer.strip_mode_only_blocks(diff_output)
        return DiffAnalyzer.strip_low_signal_hunks(cleaned)

    # ---- repo introspection ----

    def tracked_files(self, *, refresh: bool = False) -> List[str]:
        if refresh:
            self._tracked_cache = None
        if self._tracked_cache is not None:
            return self._tracked_cache
        try:
            proc = subprocess.run(
                ["git", "ls-files"],
                cwd=str(self.repo),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10,
            )
        except Exception:
            self._tracked_cache = []
            return self._tracked_cache
        if proc.returncode != 0:
            self._tracked_cache = []
            return self._tracked_cache
        self._tracked_cache = [
            line.strip() for line in proc.stdout.splitlines() if line.strip()
        ]
        return self._tracked_cache

    def read_file(self, relative_path: str, max_chars: int) -> str:
        path = (self.repo / relative_path).resolve()
        try:
            path.relative_to(self.repo.resolve())
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

    def summary(self) -> str:
        """Plain-markdown description of the repo state at agent start.

        Replaces the previous `format_observation`-based output which produced
        ~250 lines of `COMMAND: / EXIT_CODE: / DURATION_SECONDS:` shell-log
        boilerplate per run. Concretely:

          * cwd is taken from `self.repo` directly \u2014 no `pwd` shell call.
          * The file list is filtered through `_context_file_allowed` so the
            agent sees only readable source/text/config files. Build artifacts
            (`.class`, `.jar`, `.pyc`, ...) and binary assets (`.jpg`, `.gif`,
            `.eps`, ...) are excluded; they were prompt-budget noise.
          * `git status --short` is dropped by default (empty for mined tasks);
            the `enable_uncommitted_status` toggle re-adds a one-line note
            when the working tree is not clean.

        Cap of 250 listed entries keeps the summary bounded on huge repos.
        """
        tracked = self.tracked_files()
        readable = [p for p in tracked if _context_file_allowed(p)]

        lines: List[str] = [
            f"Working directory: {self.repo}",
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

        if self.config.enable_uncommitted_status:
            status = self._uncommitted_status_short()
            if status:
                lines.append("")
                lines.append("Uncommitted changes (working tree is not clean):")
                lines.extend(status.splitlines()[:20])

        return "\n".join(lines)

    def _uncommitted_status_short(self) -> str:
        """Return ``git status --short`` output, stripped, or ``""`` on error.

        Used only when ``AgentConfig.enable_uncommitted_status`` is set. Kept
        as a tiny dedicated helper so ``summary()`` doesn't mix shell I/O with
        formatting logic.
        """
        try:
            proc = subprocess.run(
                ["git", "status", "--short"],
                cwd=str(self.repo),
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

    def recent_commit_examples(
        self, issue: Optional["IssueAnalyzer"] = None
    ) -> str:
        """Read recent small-diff commits and format them as in-context
        style anchors. Returns empty string when the repo has no real
        history (single synthetic commit in pilot snapshots) — silent
        no-op locally, real lift live where the validator clones the
        upstream repo with full history.

        The model imitates concrete examples better than abstract rules.
        Cursor's reference patch IS a one-off commit in this codebase's
        style; showing the model 1-2 real recent commits gives it the same
        anchor.

        v1.2 changes vs v1.1:
          * (A) relevance filter — drop commits whose touched files are
            all binary/lockfile/generated, since their diff body teaches
            the model nothing about how *source* in this repo gets edited.
          * (B) issue-aware ranking — when an ``IssueAnalyzer`` is
            available, score each candidate by file-path overlap with the
            issue's ``path_mentions`` / ``terms`` and by symbol overlap
            with the diff body, then pick the top-scoring ones (recency
            breaks ties). Without an ``IssueAnalyzer`` we fall back to
            recency.
          * tighter per-diff and total-block char budgets (see config).
        """
        try:
            proc = subprocess.run(
                ["git", "log", "--no-merges", "--pretty=format:%H", "-n", "20"],
                cwd=str(self.repo),
                capture_output=True,
                text=True,
                timeout=10,
            )
            if proc.returncode != 0 or not proc.stdout.strip():
                return ""
            shas = [s.strip() for s in proc.stdout.splitlines() if s.strip()]
            if len(shas) < 2:
                return ""

            # Build a small candidate pool from the most recent commits
            # that pass the per-commit filters. We bound the pool size to
            # cap the cost of ranking; without (B) the original code
            # essentially used pool=2 and stopped, which was effectively
            # "first two recent commits that fit" - fine when history
            # was tiny, lossy on real repos.
            candidates: List[Tuple[int, int, str]] = []
            for recency_idx, sha in enumerate(shas):
                if len(candidates) >= _RECENT_COMMIT_POOL_SIZE:
                    break
                stat_proc = subprocess.run(
                    ["git", "show", "--no-merges", "--shortstat", "--pretty=format:", sha],
                    cwd=str(self.repo),
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if stat_proc.returncode != 0:
                    continue
                insertions = _parse_shortstat_insertions(stat_proc.stdout)
                if insertions == 0 or insertions > self.config.recent_commit_max_insertions:
                    continue

                # NOTE: passing --pretty=format:%s caused git to emit the
                # commit subject in place of the standard header but then
                # still appended the diff. After the >=100 char filter the
                # only commits that survived were those with very long
                # subjects (e.g. squash messages); their wrapped output was
                # subject + diff, which is noise. --pretty=format: empties
                # the header entirely so we keep just the diff body.
                diff_proc = subprocess.run(
                    ["git", "show", "--no-merges", "--pretty=format:", sha],
                    cwd=str(self.repo),
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if diff_proc.returncode != 0:
                    continue
                diff_text = diff_proc.stdout.strip()
                if (
                    len(diff_text) < 100
                    or len(diff_text) > self.config.recent_commit_max_diff_chars
                ):
                    continue

                # (A) relevance filter: at least one touched file must be
                # something a coding agent could plausibly edit. Drops
                # lockfile-only / binary-only / CI-only commits whose diff
                # bodies are not useful style anchors.
                touched = DiffAnalyzer.changed_files(diff_text)
                if not touched or not any(
                    _context_file_allowed(p) for p in touched
                ):
                    continue

                # (B) score by overlap with the issue. Higher = more
                # relevant style anchor for this specific task.
                score = self._score_commit_for_issue(
                    touched=touched, diff_text=diff_text, issue=issue
                )
                candidates.append((score, recency_idx, diff_text))

            if not candidates:
                return ""

            # Sort by score desc, then by recency asc (lower idx == more
            # recent). When no IssueAnalyzer is wired in every score is 0
            # so ordering collapses back to pure recency - safe behaviour
            # for unit tests and bare-CLI invocations.
            candidates.sort(key=lambda t: (-t[0], t[1]))

            examples: List[str] = []
            budget_used = 0
            for _score, _recency_idx, diff_text in candidates:
                block = (
                    "```diff\n"
                    + diff_text[: self.config.recent_commit_max_diff_chars]
                    + "\n```"
                )
                if budget_used + len(block) > self.config.recent_commit_block_budget:
                    continue
                examples.append(block)
                budget_used += len(block)
                if len(examples) >= 2:
                    break

            if not examples:
                return ""
            # Body only - the section header is owned by Prompts.initial_user
            # so headers don't double up when this string is rendered.
            return "\n\n".join(examples)
        except Exception:
            return ""

    @staticmethod
    def _score_commit_for_issue(
        touched: List[str],
        diff_text: str,
        issue: Optional["IssueAnalyzer"],
    ) -> int:
        """Score how well a candidate commit matches the active issue.

        Without an ``IssueAnalyzer`` every commit gets 0 and the caller
        falls back to recency ordering. With one we reward, in order:

        * exact path overlap with ``issue.path_mentions`` (heaviest - a
          commit that previously edited the same file is the best style
          anchor we can offer);
        * substring overlap between touched paths and ``issue.terms``;
        * occurrences of ``issue.symbols`` in the diff body itself, since
          a commit that already mentions the same identifiers is likely
          working in the same neighbourhood of the codebase.
        """
        if issue is None:
            return 0
        score = 0
        try:
            issue_paths = {p.strip("./") for p in issue.path_mentions}
        except Exception:
            issue_paths = set()
        try:
            issue_terms = list(issue.terms)
        except Exception:
            issue_terms = []
        try:
            issue_symbols = list(issue.symbols)
        except Exception:
            issue_symbols = []
        for touched_path in touched:
            if touched_path in issue_paths:
                score += 50
                continue
            touched_lower = touched_path.lower()
            for term in issue_terms:
                if term and term in touched_lower:
                    score += 2
        if issue_symbols:
            diff_lower = diff_text.lower()
            for sym in issue_symbols:
                if sym and sym.lower() in diff_lower:
                    score += 3
        return score

    # ---- symbol grep (used by ContextBuilder for ranking) ----

    def symbol_grep_hits(self, symbols: List[str]) -> Dict[str, int]:
        """Count how many extracted symbols each tracked file references.

        Skips on git-grep failure to keep the cycle cheap; symbol-grep is a
        *boost* to ranking, never the only signal.
        """
        if not symbols:
            return {}
        tracked_set = set(self.tracked_files())
        hits: Dict[str, int] = {}
        for symbol in symbols:
            try:
                proc = subprocess.run(
                    ["git", "grep", "-l", "-F", "--", symbol],
                    cwd=str(self.repo),
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

    # ---- recent-modification cohort (v1.4 ranking signal) ----

    def recently_touched_files(self, commit_limit: int = 30) -> Dict[str, int]:
        """Return tracked files that appeared in the last ``commit_limit``
        non-merge commits and how many times each was touched.

        v1.4 lever (3): files the project has been actively editing are
        more likely to be the next edit target than untouched legacy
        code, even when the issue text doesn't mention them. The
        per-file count lets the ranker reward "hot zone" files
        (touched by 3+ recent commits) more than "incidentally
        touched" ones (1 recent commit). Filtered through
        ``_context_file_allowed`` so generated / vendored / minified
        files don't sneak back in. Returns ``{}`` when ``git log``
        fails or the repo has no commits, since this is a *boost* on
        top of the existing ranking, never a requirement.

        ``commit_limit`` defaults to 30 — large enough to cover a
        feature branch's recent activity, small enough to keep stale
        refactors from drowning the current focus area.
        """
        if commit_limit <= 0:
            return {}
        try:
            proc = subprocess.run(
                [
                    "git", "log",
                    "--no-merges",
                    f"-n{commit_limit}",
                    "--name-only",
                    "--pretty=format:",
                ],
                cwd=str(self.repo),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=4,
            )
        except Exception:
            return {}
        if proc.returncode != 0:
            return {}
        tracked_set = set(self.tracked_files())
        hits: Dict[str, int] = {}
        for line in proc.stdout.splitlines():
            relative_path = line.strip()
            if not relative_path or relative_path not in tracked_set:
                continue
            if not _context_file_allowed(relative_path):
                continue
            hits[relative_path] = hits.get(relative_path, 0) + 1
        return hits


# ============================================================================
# Layer 3: ContextBuilder
# ============================================================================


@dataclass(frozen=True)
class PreloadedSections:
    """Decomposed user-prompt sections built by ContextBuilder.

    The previous API returned one bundled string with file snippets and
    recent commits glued together under a single "tracked-file snippets"
    header. Splitting them lets ``Prompts.initial_user`` give each section
    its own header and budget, and keeps room for future named sections
    (acceptance criteria, language idioms, etc.) without further string
    surgery. Each field is the body only \u2014 no header text \u2014 so the prompt
    template owns presentation entirely.
    """

    file_snippets: str = ""
    recent_commits: str = ""


class ContextBuilder:
    """Builds the preloaded-snippets blob the agent's first user prompt
    contains. Combines:
      1. Issue-derived file ranking (path mentions, term overlap, symbol grep)
      2. Companion-test partner inclusion (source + test slotted together)
      3. Recent-commit reference patches

    Each of the two top-level sections (file_snippets, recent_commits) is
    gated by an ``AgentConfig.enable_*`` toggle so an experiment turning one
    off is a one-line config flip rather than a code edit.
    """

    def __init__(self, repo: RepoContext, issue: IssueAnalyzer, config: AgentConfig):
        self.repo = repo
        self.issue = issue
        self.config = config

    def build(self) -> PreloadedSections:
        """Build the named sections that feed Prompts.initial_user.

        Each section has its own char budget (file snippets: per-section cap;
        recent commits: ``recent_commit_block_budget``) so a fat file-snippets
        section can no longer silently evict the recent-commits block.
        """
        file_snippets = (
            self._build_file_snippets() if self.config.enable_preloaded_context else ""
        )
        recent_commits = (
            self.repo.recent_commit_examples(self.issue)
            if self.config.enable_recent_commits
            else ""
        )
        return PreloadedSections(
            file_snippets=file_snippets,
            recent_commits=recent_commits,
        )

    def _build_file_snippets(self) -> str:
        """Rank-and-read the highest-scoring tracked files plus companion tests.

        Two improvements over a vanilla rank-and-read loop:

          1. Companion test files (tests/test_X.py for X.py, X.test.ts for
             X.ts, X_test.go for X.go, etc.) are slotted in right after their
             source partner. Real GitHub-derived tasks almost always need
             source+test changes together; without the test in context the
             agent patches only the source and misses the companion test
             update.

          2. Files that match identifier-shaped symbols extracted from the
             issue text get a substantial rank boost via `symbol_grep_hits`.
             This catches the common case where the bug is described by
             function or class name without mentioning the file path.
        """
        files = self._rank_files()
        if not files:
            return ""

        tracked_set = set(self.repo.tracked_files())
        files = self._augment_with_test_partners(files, tracked_set)

        parts: List[str] = []
        used = 0
        max_chars = self.config.max_preloaded_context_chars
        max_files = self.config.max_preloaded_files
        per_file_budget = max(
            1500,
            max_chars // max(1, min(len(files), max_files)),
        )

        for relative_path in files[:max_files]:
            snippet = self.repo.read_file(relative_path, per_file_budget)
            if not snippet.strip():
                continue
            block = f"### {relative_path}\n```\n{snippet}\n```"
            if parts and used + len(block) > max_chars:
                break
            parts.append(block)
            used += len(block)

        return "\n\n".join(parts)

    # ---- ranking ----

    # Per-pool weights for the v1.3 multi-pool symbol grep ranking. Each
    # tuple is (base, per-extra, cap) — same shape as v1.2's single
    # ``+60 + min(40, 8 * count)`` formula. Higher base means a single grep
    # hit from this pool is enough to outrank a single grep hit from a
    # weaker pool, even before extra-hit boosts kick in. Order is highest-
    # confidence first because ``_weighted_grep_score`` deduplicates symbols
    # by first-claim, so a token in two pools is grep'd once and scored at
    # the strongest pool's weight.
    _GREP_POOL_WEIGHTS: Dict[str, Tuple[int, int, int]] = {
        "paren":    (95, 12, 45),
        "string":   (80, 10, 40),
        "backtick": (70,  9, 35),
        "pascal":   (45,  6, 24),
        "snake":    (35,  5, 20),
    }

    def _weighted_grep_score(self) -> Dict[str, int]:
        """Aggregate per-file grep score across all typed symbol pools.

        Each pool runs through the existing ``RepoContext.symbol_grep_hits``
        machinery (one ``git grep -F`` per token) but contributes a
        different (base, per-extra, cap) so high-confidence pools (paren,
        string, backtick) outrank bulk pools (pascal, snake) even on equal
        per-file hit counts. Symbols appearing in two pools are claimed
        greedily by the higher-weight pool so each unique token is git-
        grep'd at most once across the whole pass.
        """
        pools = self.issue.symbol_pools
        seen_symbols: set = set()
        file_score: Dict[str, int] = {}
        for pool_name, weights in self._GREP_POOL_WEIGHTS.items():
            symbols = [s for s in pools.get(pool_name, []) if s not in seen_symbols]
            if not symbols:
                continue
            seen_symbols.update(symbols)
            base, per_extra, cap = weights
            hits = self.repo.symbol_grep_hits(symbols)
            for path, count in hits.items():
                # ``count`` is "how many of THIS pool's symbols hit this file"
                # so the base reward fires once per pool / file combination,
                # and additional hits within the same pool stack up to the cap.
                bonus = base + min(cap, per_extra * (count - 1))
                file_score[path] = file_score.get(path, 0) + bonus
        return file_score

    # ---- v1.4 path-atom + recent-modification scorers ----

    # Atom-overlap reward curve. Tuned against the workspace tasks: 1
    # match is "any one domain word in the path" — common, mild +8 boost
    # so a single overlap can't outrank a real grep hit. 2 matches is
    # "two domain words from the same issue both appear" — meaningful
    # phrase coherence (test4 has 33/59 changed-file paths matching
    # exactly this pattern). 3+ matches is rare and a strong signal so
    # the boost grows steeply.
    _ATOM_OVERLAP_BOOSTS: Tuple[int, int, int, int, int] = (0, 8, 22, 38, 50)
    _ATOM_OVERLAP_MAX: int = 70  # cap once overlap ≥ 5

    def _path_atom_overlap_score(
        self, tracked: List[str], issue_atoms: frozenset
    ) -> Dict[str, int]:
        """Per-file boost from path-atom ∩ issue-atom overlap.

        v1.4 lever (1) + (2) combined into one pass: each tracked path
        is decomposed into ≥3-char atoms via ``_split_path_parts`` and
        scored by how many distinct issue atoms appear in (or
        substring-overlap with) the path's atom set. Empty issue-atom
        set returns ``{}`` cheaply so pure-prose tasks where extraction
        produced nothing skip the loop entirely.
        """
        if not issue_atoms:
            return {}
        out: Dict[str, int] = {}
        for relative_path in tracked:
            if not _context_file_allowed(relative_path):
                continue
            path_atoms = _split_path_parts(relative_path)
            if not path_atoms:
                continue
            seen_path_atoms = set(path_atoms)
            overlap = 0
            for issue_atom in issue_atoms:
                if any(_atoms_match(issue_atom, pa) for pa in seen_path_atoms):
                    overlap += 1
            if overlap == 0:
                continue
            if overlap < len(self._ATOM_OVERLAP_BOOSTS):
                out[relative_path] = self._ATOM_OVERLAP_BOOSTS[overlap]
            else:
                # ≥5 overlapping atoms — clip to the cap so a path that
                # happens to contain many domain words doesn't crowd out
                # files that genuinely match the *issue's* phrasing.
                out[relative_path] = self._ATOM_OVERLAP_MAX
        return out

    # Recent-modification reward curve. Conservative: a "hot zone" file
    # (touched by 4+ recent commits) gets at most +20 — well below the
    # ``+35`` of a single weighted grep hit so we never let recency
    # alone dominate semantic match signals.
    _RECENT_TOUCH_LIMIT: int = 30
    _RECENT_TOUCH_BASE: int = 5
    _RECENT_TOUCH_PER_EXTRA: int = 6
    _RECENT_TOUCH_CAP: int = 20

    def _recent_modification_score(self) -> Dict[str, int]:
        """Per-file boost from how recently / frequently the file was touched.

        v1.4 lever (3): files actively edited in the last
        ``_RECENT_TOUCH_LIMIT`` commits are more likely to be the next
        edit target than untouched legacy code. The boost is capped low
        so it acts as a tiebreaker, not a primary ranker. Falls back to
        ``{}`` silently when the repo has no log (fresh init, broken
        history); the rest of the ranking is unaffected.
        """
        touched = self.repo.recently_touched_files(self._RECENT_TOUCH_LIMIT)
        if not touched:
            return {}
        out: Dict[str, int] = {}
        for path, count in touched.items():
            bonus = self._RECENT_TOUCH_BASE + min(
                self._RECENT_TOUCH_CAP - self._RECENT_TOUCH_BASE,
                self._RECENT_TOUCH_PER_EXTRA * (count - 1),
            )
            out[path] = bonus
        return out

    def _rank_files(self) -> List[str]:
        """Score every tracked source file and return them in rank order.

        v1.4 changes vs v1.3:

          * Path-atom inverted-index overlap (``_path_atom_overlap_score``).
            Each tracked path is split into ≥3-char domain atoms (camel /
            snake / kebab segments minus ``_PATH_ATOM_STOP``) and scored
            by how many issue atoms (terms ∪ all five typed pools)
            overlap. Recovers prose-only English tasks (test4) where
            v1.3's symbol-extraction pools came up short — the issue
            mentions ``Customer`` and ``Service`` and ``Pos`` as ordinary
            English words but the *paths* still spell those out.
          * Recent-modification boost (``_recent_modification_score``).
            Files touched by the last 30 non-merge commits get a small
            +5 to +20 tiebreaker. Capped well below grep weight so this
            never overrides semantic match signals; it just breaks ties
            between two domain-equivalent files.

        v1.3 (still in effect):

          * Multi-pool weighted symbol-grep boost.
          * No test-file ``+2 per term`` double-count.
        """
        tracked = self.repo.tracked_files()
        if not tracked:
            return []

        issue_lower = self.issue.text.lower()
        path_mentions = self.issue.path_mentions
        tracked_set = set(tracked)

        mentioned: List[str] = []
        for mention in path_mentions:
            normalized = mention.strip("./")
            if normalized in tracked_set and _context_file_allowed(normalized):
                mentioned.append(normalized)

        terms = self.issue.terms
        grep_score = self._weighted_grep_score()
        atom_score = self._path_atom_overlap_score(tracked, self.issue.issue_atoms)
        recent_score = self._recent_modification_score()

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
            score += grep_score.get(relative_path, 0)
            score += atom_score.get(relative_path, 0)
            score += recent_score.get(relative_path, 0)
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

    # ---- companion-test partners ----

    @staticmethod
    def find_test_partner(relative_path: str, tracked: set) -> Optional[str]:
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

    @classmethod
    def _augment_with_test_partners(cls, files: List[str], tracked: set) -> List[str]:
        """Slot each ranked source file's companion test in immediately after it."""
        if not tracked:
            return files
        augmented: List[str] = []
        seen: set = set()
        for relative_path in files:
            if relative_path not in seen:
                augmented.append(relative_path)
                seen.add(relative_path)
            partner = cls.find_test_partner(relative_path, tracked)
            if partner and partner not in seen:
                augmented.append(partner)
                seen.add(partner)
        return augmented

# ============================================================================
# Layer 3: Prompts
# ============================================================================

# MINER-EDITABLE: All prompt strings + builders. Pure string ops; no I/O.
# Prompt improvements are encouraged as long as they respect the
# validator-owned boundaries above.
class Prompts:
    """All system + builder prompts in one namespace."""

    SYSTEM_PROMPT = textwrap.dedent("""
        You are a surgical coding agent. Your patch is scored two ways, each worth 50%:
        1. Cursor similarity \u2014 how closely your diff matches the reference in the files touched, line regions changed, and tokens added/removed.
        2. LLM judge \u2014 scores your patch 0-100 for correctness, completeness, and alignment with the task and reference patch. A patch that is correct and complete scores high here even when similarity is modest.

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

        **Read the issue once** and identify every requirement. Missing any one loses points.

        **Edit early.** If the preloaded snippets already show the target, edit immediately \u2014 do not re-read or grep first. If the target is unclear, run at most 1\u20132 focused greps to locate it, then edit.

        **Edit surgically.** Each response emits exactly one `<command>`, so edit one location per turn. Change only the lines that implement the fix.
        - One-line substitutions: `sed -i 's/old/new/' file`
        - Small block replacements: `python -c "import pathlib; p=pathlib.Path('file'); p.write_text(p.read_text().replace('''old''', '''new'''))"`
        - Larger edits: a minimal Python script or heredoc
        - Never rewrite an entire function when only 1\u20133 lines need changing
        - Multi-file work proceeds across consecutive turns; do not try to batch. Preloaded companion tests are just another file in that sequence \u2014 edit them next turn if your source change affects them.

        **Finish.** Emit `<final>summary</final>` as soon as the patch is correct and complete. Do not re-read files. Do not run tests \u2014 the sandbox does not provide a working test runner; reason about correctness from the diff and ship.

        ## Acceptance criteria

        Every requirement listed in the issue is an unmet change you must make. None of them are already satisfied in the repo \u2014 that is why the task exists. Even if you see related or similar code, do NOT assume a criterion is already implemented; the criterion is listed because its implementation is missing, incomplete, or wrong.

        Before emitting `<final>`:
        - Re-read every acceptance criterion / requirement bullet in the issue.
        - Confirm your patch contains a concrete edit that addresses each one.
        - If any criterion is unaddressed, do not finish \u2014 issue the missing edit.

        ## Scope

        **Do:**
        - Edit exactly what the issue names ("fix X in function Y" \u2192 only function Y; "add feature Z to class C" \u2192 only what Z requires inside C).
        - Use the EXACT variable / function / class names already in the codebase.
        - Add new imports at the same location as existing imports.

        **Don't:**
        - Whitespace-only, comment-only, or blank-line-only edits.
        - Imports, type annotations, error handling, or logging not required by the fix.
        - Refactoring, renaming, reordering, or new helpers / abstractions the issue does not ask for.
        - New files unless the issue explicitly requires them.
        - Test files unless the issue explicitly requires editing them.

        ## Comment + structure preservation

        Preserve EVERY comment from the surrounding code unless the task explicitly
        removes it. Section-grouping comments (`// Member 1 availability`) are
        high-signal to the judge. Removing comments while refactoring tanks judge
        score.

        ## Language-specific completeness rules

        **Java:** Write complete method bodies \u2014 never use '// similar logic' stubs.
        Cascade all call-site changes when modifying signatures. Include all imports.

        **C/C++:** Edit both .h header AND .cpp implementation for each changed
        function. Include full signatures and all required #include changes.

        **TypeScript/C#:** Cascade interface and type changes to ALL implementing
        classes, components, and function parameters. Missing one = lower score.

        **Go/Rust:** Update every struct field usage. Provide complete Rust lifetime
        annotations on modified functions.

        **Multi-file tasks:** Complete ALL affected files in the same diff \u2014 never
        leave a related file partially edited. When in doubt, include more files.

        ## Style matching

        Copy indentation, quote style, brace style, trailing commas, blank-line patterns, AND control flow (loop vs. unrolled, comprehension vs. map) exactly from adjacent code. The reference patches in the preloaded section are the best style anchor \u2014 mirror their shape.

        ## Preloaded snippets

        Preloaded files are the most likely edit targets. Edit them directly \u2014 do not re-read them.

        ## Safety

        No sudo. No file deletion. No network access outside the validator proxy. No host secrets. No modifying hidden test or evaluator files.
        """).strip()

    @staticmethod
    def initial_user(
        issue_text: str,
        repo_summary: str,
        preloaded_context: str = "",
        recent_commits: str = "",
    ) -> str:
        """Assemble the first user-turn prompt from named, optional sections.

        Each section is emitted only when its body is non-empty, so toggling
        a section off via ``AgentConfig.enable_*`` (or passing an empty
        string at the call site) cleanly removes it from the rendered prompt
        \u2014 no orphan headers, no double blank lines.

        Section order is chosen to put the most-attended content at the
        prompt's start (issue + repo summary) and the model's freshest
        context at the end (recent commits + closing strategy).
        """
        sections: List[str] = [
            f"Fix this issue:\n\n{issue_text}",
            f"## Repository summary\n\n{repo_summary}",
        ]

        if preloaded_context.strip():
            sections.append(
                "## Preloaded code context\n"
                "(already read for you \u2014 do not re-read)\n\n"
                f"{preloaded_context}"
            )

        if recent_commits.strip():
            sections.append(
                "## Recent reference patches\n"
                "(style anchors \u2014 match the shape, scale, and conventions of "
                "these real recent commits when writing your patch)\n\n"
                f"{recent_commits}"
            )

        sections.append(textwrap.dedent("""
            Before planning, read the ENTIRE issue above and identify every requirement (there may be more than one). Your patch must satisfy ALL of them \u2014 the LLM judge penalizes incomplete solutions.

            Strategy: the fix is typically in ONE specific function or block. Identify it precisely, then make the minimal edit that fixes the ROOT CAUSE.

            If the preloaded snippets show the target code, edit them directly \u2014 do not re-read or run broad searches first. If the target is unclear, run ONE or TWO focused grep/sed -n commands to locate it, then edit immediately.

            When multiple files need edits, include EVERY independent edit command in the SAME response. Do not split edits across turns.

            After patching, run the most targeted test available (`pytest tests/test_X.py -x -q`, `go test ./...`, etc.) to verify correctness. Then finish with <final>...</final>.
        """).strip())

        return "\n\n".join(sections)

    @staticmethod
    def no_command_repair() -> str:
        return textwrap.dedent("""
            Your previous response did not contain a valid <command>...</command> block or <final>...</final> block.

            If the patch is complete, respond with <final>summary</final>. Otherwise continue
            by issuing exactly one bash command in this format:

            <command>
            your command here
            </command>
        """).strip()

    @staticmethod
    def budget_pressure_weak(step: int) -> str:
        return textwrap.dedent(f"""
            Step {step} budget check: still no repo change.
            Your next command should edit the most likely file using what you
            already know from the issue and preloaded snippets. A precise sed
            or python -c is better than another grep. Stop exploring.
        """).strip()

    @staticmethod
    def budget_pressure_strong(step: int) -> str:
        return textwrap.dedent(f"""
            Step {step} hard budget check: still no patch after multiple turns.
            Your next command MUST make a code change \u2014 even a best-effort
            minimal edit to the most obvious location. Do not read files or
            run tests until after a patch exists. Use `sed -i` or a python
            one-liner to make the targeted edit now.
        """).strip()

    FINAL_CHECK_SYSTEM = textwrap.dedent("""
        You are a strict, concise final-review judge. You receive:
          - the original task (issue text),
          - the agent's last message,
          - the agent's proposed patch as a unified diff.

        Score the patch 0\u2013100 using this rubric, in priority order:

        1. ACCEPTANCE CRITERIA (highest weight). Does the patch contain a
           concrete edit that addresses every requirement / criterion in
           the issue? Missing or partial coverage of any criterion is the
           heaviest penalty. Do NOT assume a criterion is already
           satisfied just because related code exists in the repo \u2014 if a
           criterion is listed it was unmet at the start of the task.

        2. CORRECTNESS. Does the patch edit the right file / function /
           symbol, without obvious bugs (off-by-one, wrong condition,
           swapped args, broken call signature, missing import)? Obvious
           syntax breaks fall here too.

        3. STYLE + SCOPE. Does the patch match the surrounding code's
           indentation / quoting / brace / control-flow conventions, use
           the EXACT names already in the codebase, and avoid unnecessary
           changes (whitespace-only edits, gratuitous refactoring, new
           helpers, error handling not asked for)?

        Score anchors:
        - 90\u2013100: every criterion addressed correctly, clean style, minimal scope.
        - 70\u201389:  every criterion addressed but with minor style or scope issues.
        - 50\u201369:  most criteria addressed; one missing/partial OR a real bug.
        - 0\u201349:   multiple missing criteria OR wrong-target edits OR broken syntax.

        Be precise and concrete. Prefer FAIL when in doubt about ACCEPTANCE
        CRITERIA coverage; STYLE issues alone are rarely a FAIL.

        Respond with exactly the format requested by the user message and
        nothing else.
    """).strip()

    @staticmethod
    def final_check(issue_text: str, last_response: str, patch: str) -> str:
        """User message for the final-review LLM judge.

        Truncates each section so the judge call stays within token budget
        even on large issues / sprawling diffs.
        """
        issue_short = issue_text[:4000]
        response_short = last_response[:2000]
        if len(patch) <= 8000:
            diff_view = patch
        else:
            diff_view = patch[:5000] + "\n...[diff truncated]...\n" + patch[-2500:]
        return textwrap.dedent(f"""
            TASK / ISSUE:
            {issue_short}

            AGENT'S LAST MESSAGE:
            {response_short}

            PATCH (unified diff):
            ```diff
            {diff_view}
            ```

            Score and judge the patch using the rubric in your system
            prompt. Reply using EXACTLY this format and nothing else:

            SCORE: <integer 0-100>
            VERDICT: PASS|FAIL
            REASON: <one or two sentences; what is right and what is wrong>
            FIX_LIST:
            - <concrete bullet naming the file / symbol / criterion to fix>
            - <bullet>
            - ...

            Rules:
            - Use VERDICT: PASS only when every requirement is addressed
              correctly and there are no obvious bugs or wrong-target edits.
            - On VERDICT: FAIL the FIX_LIST must contain at least one
              concrete bullet that names the file / symbol / criterion to
              fix; vague bullets ("improve quality") are not actionable.
            - On VERDICT: PASS the FIX_LIST may be empty (write a single
              "- (none)" bullet).
        """).strip()

    @staticmethod
    def final_fix(
        score: int,
        reason: str,
        fix_list: List[str],
        issue_text: str,
    ) -> str:
        """User-message refinement prompt fed back when the judge fails.

        Quotes the judge's score, prose reason, and concrete fix bullets
        verbatim, then asks the agent to fix only those items and emit
        ``<final>``.
        """
        issue_short = issue_text[:1500]
        if fix_list:
            bullets = "\n".join(f"- {item}" for item in fix_list)
        else:
            bullets = "- (judge gave no concrete bullets; use REASON above)"
        return textwrap.dedent(f"""
            Final check FAILED (score {score}/100). Judge said:

            {reason}

            Things to fix:
            {bullets}

            Address every item above with one or more <command> blocks
            (use sed or `python -c` for surgical edits) and then end with
            <final>summary</final>. Do NOT add scope the task did not
            ask for; fix only the named gaps and bugs.

            Task (for reference):
            {issue_short}
        """).strip()


# ============================================================================
# Layer 4: FinalChecker
# ============================================================================


class FinalChecker:
    """LLM-based final-review judge.

    ``judge(issue, last_response, patch)`` asks the model to score the
    patch 0\u2013100 against three rubric dimensions (acceptance criteria,
    correctness, style + scope), to declare PASS or FAIL, and to emit a
    concrete FIX_LIST. Returns a :class:`FinalVerdict`.

    A patch is considered ``passed`` only when BOTH:
      - the judge's explicit VERDICT is PASS, and
      - the score >= ``config.final_check_pass_threshold``.

    Permissive on failure (user-explicit policy: a flaky judge must never
    block shipping). Each of these falls back to ``passed=True`` with a
    placeholder reason:
      - LLM endpoint raises an exception
      - response is empty / whitespace
      - response is missing SCORE or VERDICT
      - SCORE is not an integer in [0, 100]
      - VERDICT is FAIL but no actionable fix bullets were emitted
        (nothing for the agent to act on, so a retry would be wasted)
    """

    _SCORE_RE = re.compile(r"(?im)^\s*SCORE\s*:\s*(\d{1,3})\s*$")
    _VERDICT_RE = re.compile(r"(?im)^\s*VERDICT\s*:\s*(PASS|FAIL)\s*$")
    _REASON_RE = re.compile(
        r"(?ims)^\s*REASON\s*:\s*(.+?)(?:\n\s*FIX[_ ]LIST\s*:|\n\s*VERDICT\s*:|\n\s*SCORE\s*:|\Z)"
    )
    _FIX_LIST_RE = re.compile(r"(?ims)^\s*FIX[_ ]LIST\s*:\s*\n(.+?)\Z")
    _BULLET_RE = re.compile(r"(?m)^\s*-\s+(.+?)\s*$")
    _EMPTY_BULLETS = frozenset({"(none)", "none", "n/a", "na", "-"})

    def __init__(self, llm: LLMClient, config: AgentConfig):
        self.llm = llm
        self.config = config

    def judge(
        self,
        issue_text: str,
        last_response: str,
        patch: str,
    ) -> FinalVerdict:
        messages = [
            {"role": "system", "content": Prompts.FINAL_CHECK_SYSTEM},
            {
                "role": "user",
                "content": Prompts.final_check(issue_text, last_response, patch),
            },
        ]
        try:
            response_text, _, _ = self.llm.complete(messages=messages)
        except Exception as exc:
            return FinalVerdict(
                passed=True,
                score=100,
                reason=f"final check unavailable ({exc.__class__.__name__})",
                fix_list=[],
            )
        return self._parse(response_text)

    def _parse(self, text: str) -> FinalVerdict:
        if not text or not text.strip():
            return FinalVerdict(True, 100, "judge returned empty response", [])

        score_match = self._SCORE_RE.search(text)
        verdict_match = self._VERDICT_RE.search(text)
        if score_match is None or verdict_match is None:
            return FinalVerdict(
                True, 100, "judge produced no parseable score/verdict", []
            )

        try:
            score = max(0, min(100, int(score_match.group(1))))
        except ValueError:
            return FinalVerdict(True, 100, "judge score was not an integer", [])

        verdict_pass = verdict_match.group(1).upper() == "PASS"
        threshold_pass = score >= self.config.final_check_pass_threshold

        reason_match = self._REASON_RE.search(text)
        reason = reason_match.group(1).strip() if reason_match else text.strip()

        fix_list: List[str] = []
        fix_list_match = self._FIX_LIST_RE.search(text)
        if fix_list_match is not None:
            for bullet_match in self._BULLET_RE.finditer(fix_list_match.group(1)):
                bullet = bullet_match.group(1).strip()
                if bullet and bullet.lower() not in self._EMPTY_BULLETS:
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


# ============================================================================
# Layer 5: RefinementOrchestrator
# ============================================================================


class RefinementOrchestrator:
    """Single-shot LLM-judge gate.

    Replaces the v2.3-and-earlier 7-check ladder (hail-mary / polish /
    syntax / test / coverage / criteria / self-check) with one call to
    ``FinalChecker.judge``. Behaviour:

      - patch is empty            -> return None (judge has nothing to
        evaluate; caller falls through to its normal "stop only on
        <final>" behaviour, no refinement queued).
      - judge already fired N times (cap = ``max_final_check_turns``)
        -> return None (we accept whatever the agent has).
      - judge ``passed == True`` (PASS verdict AND score >= threshold)
        -> return None (caller declares success).
      - judge ``passed == False`` AND ``fix_list`` is non-empty
        -> return a RefinementTurn whose prompt carries the score,
        prose reason, and fix bullets, so the agent gets one more step
        to fix.

    Permissive: judge errors, unparseable responses, and FAILs without an
    actionable fix list are mapped to PASS by FinalChecker, so they never
    surface here as work to do.
    """

    def __init__(
        self,
        config: AgentConfig,
        issue: IssueAnalyzer,
        checker: FinalChecker,
    ):
        self.config = config
        self.issue = issue
        self.checker = checker
        self.checks_used = 0
        self.last_verdict: Optional[FinalVerdict] = None

    def next_turn(self, patch: str, last_response: str) -> Optional[RefinementTurn]:
        if not patch.strip():
            return None
        if self.checks_used >= self.config.max_final_check_turns:
            return None

        self.checks_used += 1
        verdict = self.checker.judge(self.issue.text, last_response, patch)
        self.last_verdict = verdict
        if verdict.passed:
            return None

        truncated_reason = (
            verdict.reason
            if len(verdict.reason) <= 600
            else verdict.reason[:600] + "..."
        )
        return RefinementTurn(
            marker=f"FINAL_CHECK_FAILED (score {verdict.score}/100):\n  {truncated_reason}",
            prompt=Prompts.final_fix(
                verdict.score,
                verdict.reason,
                verdict.fix_list,
                self.issue.text,
            ),
        )


# ============================================================================
# Layer 5: Agent (single attempt)
# ============================================================================


class Agent:
    """One full solve attempt. Owns the step loop, the messages list, the
    logs, the time budget, the model-error retry, and the auto-stop
    heuristics. Returns AgentResult.

    Wires together: LLMClient, ShellRunner, RepoContext, IssueAnalyzer,
    ContextBuilder, Prompts, RefinementOrchestrator, ObservationInspector.
    """

    def __init__(
        self,
        config: AgentConfig,
        repo: RepoContext,
        llm: LLMClient,
        shell: ShellRunner,
        issue: IssueAnalyzer,
        context_builder: ContextBuilder,
        refinement: RefinementOrchestrator,
    ):
        self.config = config
        self.repo = repo
        self.llm = llm
        self.shell = shell
        self.issue = issue
        self.context_builder = context_builder
        self.refinement = refinement
        self.messages: List[Dict[str, str]] = []
        self.logs: List[str] = []
        self.total_cost: Optional[float] = 0.0
        self.success = False
        self.consecutive_no_command = 0
        self.consecutive_model_errors = 0

    # ---- helpers ----

    def _safe_join_logs(self) -> str:
        joined = "\n".join(self.logs)
        return _truncate(joined, self.config.max_total_log_chars)

    def _queue_refinement(self, assistant_text: str) -> bool:
        """If the final-review judge wants the agent to keep working,
        queue a refinement turn. Returns True if a turn was queued
        (caller should keep looping)."""
        patch = self.repo.get_patch()
        turn = self.refinement.next_turn(patch, assistant_text)
        if turn is None:
            return False
        self.logs.append(f"\n{turn.marker}\n")
        self.messages.append({"role": "assistant", "content": assistant_text})
        self.messages.append({"role": "user", "content": turn.prompt})
        return True

    # ---- main loop ----

    def run(self) -> AgentResult:
        try:
            self.repo.ensure_git_repo()
            repo_summary = self.repo.summary()
            preloaded = self.context_builder.build()

            self.messages = [
                {"role": "system", "content": Prompts.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": Prompts.initial_user(
                        issue_text=self.issue.text,
                        repo_summary=repo_summary,
                        preloaded_context=preloaded.file_snippets,
                        recent_commits=preloaded.recent_commits,
                    ),
                },
            ]

            for step in range(1, self.config.max_steps + 1):
                self.logs.append(f"\n\n===== STEP {step} =====\n")

                response_text = self._call_model_with_retry(step)
                if response_text is None:
                    if self._handle_model_error():
                        break
                    continue

                self.consecutive_model_errors = 0
                self.logs.append("MODEL_RESPONSE:\n" + response_text)

                commands = ActionParser.extract_commands(response_text)
                final = ActionParser.extract_final(response_text)

                if not commands:
                    if self._handle_no_commands(response_text, final):
                        break
                    continue

                should_break = self._handle_commands(
                    response_text, commands, final
                )
                if should_break:
                    break

                self._maybe_nudge_for_edit(step)

            patch = self.repo.get_patch()
            if patch.strip() and not self.success:
                self.logs.append(
                    "\nPATCH_RETURN:\nReturning the best patch produced within the step budget."
                )
                self.success = True
            step_count = len([x for x in self.logs if x.startswith("\n\n===== STEP")])
            return AgentResult(
                patch=patch,
                logs=self._safe_join_logs(),
                steps=min(self.config.max_steps, step_count),
                cost=self.total_cost,
                success=self.success and bool(patch.strip()),
            )

        except Exception:
            self.logs.append("FATAL_ERROR:\n" + traceback.format_exc())
            patch = ""
            try:
                patch = self.repo.get_patch()
            except Exception:
                pass
            return AgentResult(
                patch=patch,
                logs=self._safe_join_logs(),
                steps=0,
                cost=self.total_cost,
                success=False,
            )

    # ---- step pieces ----

    def _call_model_with_retry(self, step: int) -> Optional[str]:
        for retry_attempt in range(self.config.max_step_retries + 1):
            try:
                response_text, cost, _raw = self.llm.complete(messages=self.messages)
                if cost is not None and self.total_cost is not None:
                    self.total_cost += cost
                return response_text
            except Exception as exc:
                self.logs.append(
                    f"MODEL_ERROR (step {step}, attempt {retry_attempt + 1}/"
                    f"{self.config.max_step_retries + 1}):\n{exc}"
                )
                if retry_attempt < self.config.max_step_retries:
                    time.sleep(self.config.http_retry_base_backoff * (2 ** retry_attempt))
                    continue
                return None
        return None

    def _maybe_nudge_for_edit(self, step: int) -> None:
        """Inject an edit-pressure prompt once the patch is still empty after
        ``edit_warn_first_step`` steps, then every ``edit_warn_interval``
        steps after that. The first nudge is gentle; later ones are firm.
        """
        if self.repo.get_patch().strip():
            return
        first = self.config.edit_warn_first_step
        interval = self.config.edit_warn_interval
        if step < first or interval <= 0:
            return
        offset = step - first
        if offset == 0:
            self.messages.append(
                {"role": "user", "content": Prompts.budget_pressure_weak(step)}
            )
        elif offset > 0 and offset % interval == 0:
            self.messages.append(
                {"role": "user", "content": Prompts.budget_pressure_strong(step)}
            )

    def _handle_model_error(self) -> bool:
        """Return True if the loop should break."""
        self.consecutive_model_errors += 1
        # If we already have any patch staged in the repo, stop early and
        # return that patch rather than wiping everything because the proxy
        # hiccuped. Empty patches score 0; partial patches can still earn
        # cursor-similarity credit.
        if self.repo.get_patch().strip():
            self.logs.append(
                "MODEL_ERROR_RECOVER:\nReturning best partial patch "
                "after persistent model errors."
            )
            self.success = True
            return True
        if self.consecutive_model_errors >= 3:
            self.logs.append(
                "MODEL_ERROR_GIVE_UP:\nNo patch and persistent model "
                "errors -- ending loop."
            )
            return True
        # No patch yet but step budget remains; ride out and try again.
        return False

    def _handle_no_commands(self, response_text: str, final: Optional[str]) -> bool:
        """Handle responses with no <command>. Return True if the outer loop
        should break (success or hard stop)."""
        if final is not None:
            if self._queue_refinement(response_text):
                return False
            self.logs.append("\nFINAL_SUMMARY:\n" + final)
            self.success = True
            return True
        self.consecutive_no_command += 1
        patch = self.repo.get_patch()
        if patch.strip():
            if self._queue_refinement(response_text):
                return False
            self.logs.append(
                "\nPATCH_READY:\nModel stopped issuing commands after creating a patch."
            )
            self.success = True
            return True
        if self.consecutive_no_command >= self.config.max_no_command_repairs:
            self.logs.append(
                "\nSTOPPED:\nModel repeatedly failed to produce a command or final answer."
            )
            return True
        self.messages.append({"role": "assistant", "content": response_text})
        self.messages.append({"role": "user", "content": Prompts.no_command_repair()})
        return False

    def _handle_commands(
        self,
        response_text: str,
        commands: List[str],
        final: Optional[str],
    ) -> bool:
        """Run the model's commands, log + feed back the observations,
        and stop only when the model emits ``<final>``.

        Returns True if the outer loop should break (success), False to
        continue. There is no auto-stop heuristic: the loop trusts the
        model to declare completion via ``<final>``. The refinement gate
        still fires on ``<final>`` so we get one chance to polish the
        patch before shipping.
        """
        self.consecutive_no_command = 0
        observations: List[str] = []
        command_batch = commands[: self.config.max_commands_per_response]

        for command_index, command in enumerate(command_batch, 1):
            result = self.shell.run(command)
            observation = self.shell.format_observation(result)
            observations.append(
                f"OBSERVATION {command_index}/{len(command_batch)}:\n{observation}"
            )
            self.logs.append(
                f"\nOBSERVATION {command_index}/{len(command_batch)}:\n" + observation
            )

        if final is not None:
            # _queue_refinement appends both the assistant draft and the
            # corrective user message; we must NOT pre-append the assistant
            # here or the refinement turn would duplicate it.
            if self._queue_refinement(response_text):
                return False
            self.logs.append("\nFINAL_SUMMARY:\n" + final)
            self.success = True
            return True

        if observations:
            self.messages.append({"role": "assistant", "content": response_text})
            self.messages.append(
                {"role": "user", "content": "\n\n".join(observations)}
            )
        return False


# ============================================================================
# Layer 6: Public entry point
# ============================================================================
#
# VALIDATOR CONTRACT \u2014 DO NOT CHANGE THE SIGNATURE OR RETURN SHAPE.
# The validator imports this module and calls solve(...) directly.
# Everything else in this file is implementation detail; this function
# MUST remain a free function (not a method) and MUST keep this exact
# parameter list.

# MINER-EDITABLE: validator entry point. Single-shot: builds the Agent
# graph once and runs it. The body must stay portable: no third-party
# imports, no hidden secrets, no LLM calls outside the validator-supplied
# api_base/api_key.
def solve(
    repo_path: str,
    issue: str,
    model: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    max_steps: int = 30,
    command_timeout: int = 15,
    max_tokens: int = 8192,
) -> Dict[str, Any]:
    """Main portable interface for validators.

    Constructs an `AgentConfig` from environment defaults overridden by
    explicit arguments, then runs a single `Agent` attempt. Returns the
    contracted dict shape: {patch, logs, steps, cost, success}.
    """
    config = AgentConfig.from_env().with_overrides(
        model=model,
        api_base=api_base,
        api_key=api_key,
        max_steps=max_steps,
        command_timeout=command_timeout,
        max_tokens=max_tokens,
    )
    repo_obj = _resolve_repo_path(repo_path)
    try:
        shell = ShellRunner(repo_obj, config)
        repo_ctx = RepoContext(repo_obj, shell, config)
        llm = LLMClient(config)
        issue_analyzer = IssueAnalyzer(issue)
        context_builder = ContextBuilder(repo_ctx, issue_analyzer, config)
        checker = FinalChecker(llm, config)
        refinement = RefinementOrchestrator(config, issue_analyzer, checker)
        agent = Agent(
            config, repo_ctx, llm, shell, issue_analyzer, context_builder, refinement
        )
        return agent.run().to_dict()
    except Exception:
        tb = "FATAL_ERROR:\n" + traceback.format_exc()
        patch = ""
        try:
            shell = ShellRunner(repo_obj, config)
            repo_ctx = RepoContext(repo_obj, shell, config)
            patch = repo_ctx.get_patch()
        except Exception:
            pass
        return AgentResult(
            patch=patch,
            logs=_truncate(tb, config.max_total_log_chars),
            steps=0,
            cost=0.0,
            success=False,
        ).to_dict()


# ============================================================================
# CLI for local testing
# ============================================================================

# LOCAL TESTING ONLY: The validator imports solve() directly. You may
# adjust the CLI to make local experiments easier, but do not rely on
# CLI-only behaviour for validation.
def _parse_args(argv: List[str]) -> Dict[str, Any]:
    import argparse

    cfg_defaults = AgentConfig.from_env()

    parser = argparse.ArgumentParser(
        description="Run portable single-file coding agent."
    )
    parser.add_argument("--repo", required=True, help="Path to repo/task directory.")
    parser.add_argument("--issue", required=False, help="Issue text.")
    parser.add_argument("--issue-file", required=False, help="File containing issue text.")
    parser.add_argument("--model", default=cfg_defaults.model, help="Model name.")
    parser.add_argument(
        "--api-base", default=cfg_defaults.api_base, help="OpenAI-compatible API base."
    )
    parser.add_argument("--api-key", default=cfg_defaults.api_key, help="API key.")
    parser.add_argument("--max-steps", type=int, default=cfg_defaults.max_steps)
    parser.add_argument(
        "--command-timeout", type=int, default=cfg_defaults.command_timeout
    )
    parser.add_argument("--max-tokens", type=int, default=cfg_defaults.max_tokens)
    parser.add_argument(
        "--json-out", default="", help="Optional path to write result JSON."
    )
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
