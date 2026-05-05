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

# MINER-EDITABLE: You may tune local budgets like step count, command timeout,
# observation size, and max_tokens. Do not set sampling parameters; the
# validator proxy owns temperature/top-p/etc. and overwrites them server-side.
DEFAULT_MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", "40"))
DEFAULT_COMMAND_TIMEOUT = int(os.environ.get("AGENT_COMMAND_TIMEOUT", "30"))

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
DEFAULT_MAX_COMMANDS_PER_TURN = int(os.environ.get("AGENT_MAX_COMMANDS_PER_TURN", "8"))
DEFAULT_HTTP_RETRY_ATTEMPTS = int(os.environ.get("AGENT_HTTP_RETRY_ATTEMPTS", "3"))

MAX_OBSERVATION_CHARS = int(os.environ.get("AGENT_MAX_OBSERVATION_CHARS", "12000"))
MAX_TOTAL_LOG_CHARS = int(os.environ.get("AGENT_MAX_TOTAL_LOG_CHARS", "200000"))

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

    last_error: Optional[Exception] = None
    data: Optional[Dict[str, Any]] = None
    for attempt in range(DEFAULT_HTTP_RETRY_ATTEMPTS):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                data = json.loads(raw)
            last_error = None
            break
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            if e.code in (500, 502, 503, 504) and attempt < DEFAULT_HTTP_RETRY_ATTEMPTS - 1:
                time.sleep(min(8.0, 1.5 ** attempt))
                last_error = e
                continue
            if e.code == 429 and "budget_exceeded" not in err_body and attempt < DEFAULT_HTTP_RETRY_ATTEMPTS - 1:
                time.sleep(min(8.0, 2 ** attempt))
                last_error = e
                continue
            raise RuntimeError(f"HTTP {e.code} from model endpoint: {err_body}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < DEFAULT_HTTP_RETRY_ATTEMPTS - 1:
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
            env={
                **os.environ,
                "PYTHONUNBUFFERED": "1",
                "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                "GIT_PAGER": "cat",
                "PAGER": "cat",
            },
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


def extract_command(model_text: str) -> Optional[str]:
    match = ACTION_RE.search(model_text)
    if not match:
        return None
    return match.group(1).strip()


def extract_commands(model_text: str) -> List[str]:
    return [c.strip() for c in ACTION_RE.findall(model_text) if c.strip()]


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

    return diff_output


def _should_skip_patch_path(relative_path: str) -> bool:
    path = Path(relative_path)
    if path.suffix == ".pyc":
        return True
    return any(part in {"__pycache__", ".pytest_cache", "node_modules", ".git"} for part in path.parts)


def get_repo_summary(repo: Path) -> str:
    commands = [
        "pwd",
        "find . -maxdepth 3 -type f | sed 's#^./##' | sort | head -200",
        "git status --short || true",
    ]

    parts = []
    for cmd in commands:
        res = run_command(cmd, repo, timeout=10)
        parts.append(format_observation(res))

    return "\n\n".join(parts)


# -----------------------------
# Prompting
# -----------------------------

# MINER-EDITABLE: This prompt is the main behavior policy for the inner coding
# agent. Prompt improvements are encouraged as long as they respect the
# validator-owned boundaries above.
SYSTEM_PROMPT = """You are a coding agent running inside a repository.

You must fix the issue by editing files in the repo.

You interact by issuing bash commands. The environment runs your commands and returns stdout/stderr/exit_code for each one.

Use this exact format. You MAY include up to 8 <command> blocks per response; they will be executed in order and you will receive observations for all of them in the next turn:

<command>
your bash command here
</command>

Each command runs in a FRESH bash shell at the repository root. Shell state (cwd, env vars, set variables) does NOT persist between commands. If you need a working directory, chain it: `cd subdir && pytest -x`. To make a multi-line edit, use a single heredoc command like:

<command>
cat > path/to/file.py <<'EOF'
new contents
EOF
</command>

When you are finished, respond with:

<final>
short summary of what you changed
</final>

Strategy:
- Batch read commands early (e.g. cat several files at once) to minimize turns.
- After reading enough, plan your edits, then write them with heredocs or sed.
- Run tests if a test runner exists. Use exit codes to verify.
- If you have a working patch, output <final> immediately - do not waste budget on extras.

Rules:
- Work directly in the repository.
- Inspect files before editing them.
- Do not use sudo. Do not delete the repository. Do not access secrets.
- Do not make network calls except through the validator-provided inference proxy.
- Do not modify hidden tests or evaluator files.
- Do not stop after only explaining; actually edit the code.
- You may use python, sed, cat, grep, find, pytest, npm, etc.
"""


def build_initial_user_prompt(issue: str, repo_summary: str, prelocalized: str = "") -> str:
    parts = [
        "We need to fix this issue:",
        "",
        issue,
        "",
        "Repository summary:",
        "",
        repo_summary,
    ]
    if prelocalized.strip():
        parts.extend(["", "Files referenced by the issue (pre-loaded):", "", prelocalized])
    parts.extend([
        "",
        "Start by reading any unfamiliar files in batched <command> blocks. Then plan, edit, and verify.",
    ])
    return "\n".join(parts)


_FILE_HINT_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_./-]*\.(?:py|js|ts|jsx|tsx|go|rs|java|kt|c|h|cpp|hpp|rb|sh|yml|yaml|json|toml|md|html|css))\b")


def prelocalize_files_from_issue(repo: Path, issue: str, max_files: int = 8, max_bytes_per_file: int = 4000) -> str:
    candidates: List[str] = []
    seen: set[str] = set()
    for match in _FILE_HINT_RE.findall(issue):
        if match in seen:
            continue
        seen.add(match)
        candidates.append(match)
        if len(candidates) >= 32:
            break

    chosen: List[Tuple[str, str]] = []
    for hint in candidates:
        if len(chosen) >= max_files:
            break
        target = repo / hint
        try:
            if target.is_file():
                content = target.read_text(encoding="utf-8", errors="replace")[:max_bytes_per_file]
                chosen.append((hint, content))
                continue
        except Exception:
            pass
        try:
            proc = subprocess.run(
                ["git", "ls-files", "--", f"*/{hint}", hint],
                cwd=str(repo),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5,
            )
            for path_str in (proc.stdout or "").splitlines():
                path_str = path_str.strip()
                if not path_str:
                    continue
                p = repo / path_str
                if p.is_file():
                    try:
                        content = p.read_text(encoding="utf-8", errors="replace")[:max_bytes_per_file]
                        chosen.append((path_str, content))
                        break
                    except Exception:
                        continue
        except Exception:
            continue

    if not chosen:
        return ""
    blocks = []
    for path_str, content in chosen:
        blocks.append(f"--- BEGIN {path_str} ---\n{content}\n--- END {path_str} ---")
    return "\n\n".join(blocks)


def _trim_messages(messages: List[Dict[str, str]], keep_recent_pairs: int = 3, summary_chars: int = 240) -> List[Dict[str, str]]:
    """Keep system + first user prompt + last keep_recent_pairs (assistant,user) pairs;
    older observations are summarized to a short header to control prompt growth."""
    if len(messages) <= 2 + keep_recent_pairs * 2:
        return messages
    head = messages[:2]
    tail = messages[-keep_recent_pairs * 2 :]
    middle = messages[2 : -keep_recent_pairs * 2]

    summarized: List[Dict[str, str]] = []
    for msg in middle:
        content = msg["content"]
        if msg["role"] == "user" and content.startswith("COMMAND:"):
            first_line = content.split("\n", 1)[0]
            exit_match = re.search(r"EXIT_CODE:\n(-?\d+)", content)
            ec = exit_match.group(1) if exit_match else "?"
            summarized.append({"role": "user", "content": f"<earlier observation: {first_line[:80]} exit={ec}>"})
        elif msg["role"] == "assistant":
            cmds = ACTION_RE.findall(content)
            if cmds:
                summary = "; ".join(c.strip().split("\n", 1)[0][:80] for c in cmds[:3])
                summarized.append({"role": "assistant", "content": f"<earlier turn: {len(cmds)} cmds: {summary}>"})
            else:
                summarized.append({"role": "assistant", "content": content[:summary_chars]})
        else:
            summarized.append({"role": msg["role"], "content": content[:summary_chars]})

    return head + summarized + tail


def build_no_command_repair_prompt() -> str:
    return """Your previous response did not contain a valid <command>...</command> block or <final>...</final> block.

Continue by issuing exactly one bash command in this format:

<command>
your command here
</command>
"""


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

    try:
        repo = _repo_path(repo_path)
        model_name, api_base, api_key = _resolve_inference_config(model, api_base, api_key)
        ensure_git_repo(repo)
        repo_summary = get_repo_summary(repo)
        prelocalized = prelocalize_files_from_issue(repo, issue)

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_initial_user_prompt(issue, repo_summary, prelocalized)},
        ]

        for step in range(1, max_steps + 1):
            logs.append(f"\n\n===== STEP {step} =====\n")

            try:
                response_text, cost, _raw = chat_completion(
                    messages=_trim_messages(messages),
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

            final = extract_final(response_text)
            if final is not None:
                logs.append("\nFINAL_SUMMARY:\n" + final)
                success = True
                break

            commands = extract_commands(response_text)

            if not commands:
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content": build_no_command_repair_prompt()})
                continue

            commands = commands[:DEFAULT_MAX_COMMANDS_PER_TURN]
            observations: List[str] = []
            last_result: Optional[CommandResult] = None
            for cmd in commands:
                result = run_command(cmd, repo, timeout=command_timeout)
                observations.append(format_observation(result))
                last_result = result
            combined_observation = "\n---\n".join(observations)

            logs.append("\nOBSERVATION:\n" + combined_observation)

            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": combined_observation})

            if step >= 4 and last_result is not None:
                patch = get_patch(repo)
                if patch.strip() and _looks_like_successful_test_output(observations[-1]):
                    logs.append("\nAUTO_STOP:\nPatch exists and latest command looked like successful tests.")
                    success = True
                    break

        patch = get_patch(repo)
        return AgentResult(
            patch=patch,
            logs=_safe_join_logs(logs),
            steps=step,
            cost=total_cost,
            success=bool(patch.strip()),
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


def _looks_like_successful_test_output(observation: str) -> bool:
    lower = observation.lower()
    exit_code = _extract_observation_exit_code(lower)
    if exit_code is None or exit_code != 0:
        return False

    stdout_body = _extract_observation_section(lower, "stdout")
    stderr_body = _extract_observation_section(lower, "stderr")
    body = stdout_body + "\n" + stderr_body

    nonzero_failed = re.search(r"\b([1-9]\d*)\s+failed\b", body)
    nonzero_errors = re.search(r"\b([1-9]\d*)\s+errors?\b", body)
    if nonzero_failed or nonzero_errors:
        return False
    if "traceback (most recent call last)" in body:
        return False

    if re.search(r"\b\d+\s+passed\b", body):
        return True
    last_lines = [ln for ln in (stdout_body.splitlines() or []) if ln.strip()]
    if last_lines and last_lines[-1].strip() in {"ok", "success"}:
        return True
    return False


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
