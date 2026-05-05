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
DEFAULT_MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", "30"))
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

MAX_OBSERVATION_CHARS = int(os.environ.get("AGENT_MAX_OBSERVATION_CHARS", "12000"))
MAX_TOTAL_LOG_CHARS = int(os.environ.get("AGENT_MAX_TOTAL_LOG_CHARS", "200000"))

# Keep at most this many messages in the conversation window (system + user pairs).
# Older middle messages are trimmed to avoid context overflow on long runs.
_MAX_CONTEXT_MESSAGES = 22

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
    timeout: int = 300,
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
    return f"""COMMAND:
{result.command}

EXIT_CODE:
{result.exit_code}

DURATION_SECONDS:
{result.duration_sec:.3f}

STDOUT:
{result.stdout}

STDERR:
{result.stderr}
"""


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
        (
            "find . -maxdepth 3 -type f \\( -name '*.py' -o -name '*.js' -o -name '*.ts' "
            "-o -name '*.rb' -o -name '*.go' -o -name '*.java' -o -name '*.c' -o -name '*.cpp' "
            "-o -name '*.rs' -o -name 'Makefile' -o -name 'pytest.ini' "
            "-o -name 'package.json' -o -name 'setup.py' -o -name 'pyproject.toml' \\) "
            "| grep -v '__pycache__\\|node_modules\\|\\.git\\|\\.pytest_cache' "
            "| sed 's#^./##' | sort | head -60"
        ),
        "find . -maxdepth 1 -type f | sed 's#^./##' | sort",
        "git log --oneline -8 2>/dev/null || true",
        "git status --short || true",
        "cat pytest.ini setup.cfg pyproject.toml 2>/dev/null | head -40 || true",
    ]

    parts = []
    for cmd in commands:
        res = run_command(cmd, repo, timeout=15)
        parts.append(format_observation(res))

    return "\n\n".join(parts)


# -----------------------------
# Context management
# -----------------------------

def _trim_messages(
    messages: List[Dict[str, str]],
    max_messages: int = _MAX_CONTEXT_MESSAGES,
) -> List[Dict[str, str]]:
    """
    Trim conversation history to prevent context overflow.

    Always preserves:
    - messages[0]: system prompt
    - messages[1]: initial user task

    Trims the oldest middle exchanges when the conversation grows too long.
    """
    if len(messages) <= max_messages:
        return messages
    # Keep system + initial task message, then take the most recent remainder
    anchor = messages[:2]
    keep_count = max_messages - 2
    recent = messages[-keep_count:]
    return anchor + recent


# -----------------------------
# Prompting
# -----------------------------

# MINER-EDITABLE: This prompt is the main behavior policy for the inner coding
# agent. Prompt improvements are encouraged as long as they respect the
# validator-owned boundaries above.
SYSTEM_PROMPT = """You are an expert software engineer fixing GitHub issues with precise, minimal code changes.

WORKFLOW:
1. ANALYZE: Read the issue and extract key identifiers (function names, class names, error messages, file hints)
2. LOCATE: Use grep to find relevant code; read target files COMPLETELY before editing
3. FIX: Make the smallest correct change that addresses the issue
4. VERIFY: Run only the specific relevant tests to confirm correctness
5. FINALIZE: Submit immediately with <final>brief summary</final>

COMMAND FORMAT - one command per response:
<command>
bash command here
</command>

When the fix is verified:
<final>brief description of what was changed and why</final>

EDITING RULES (critical for quality):
- Read any file COMPLETELY before editing it: cat -n filename
- Change ONLY the lines the issue requires; preserve all surrounding code
- Match existing code style exactly: indentation, quotes, spacing, naming
- Do NOT add comments, docstrings, or error handling unless the issue requires it
- Do NOT reorder imports, rename variables, or reformat unrelated code
- Do NOT fix unrelated issues even if you notice them
- Use sed -i for targeted single-line replacements
- Use python3 -c with open() for multi-line or complex edits
- Process files in alphabetical order; edit top-to-bottom within each file

EXPLORATION:
- grep -rn "pattern" --include="*.py" . : find relevant code
- grep -rn "pattern" . : search all files
- cat -n filename : read file with line numbers
- find . -name "*.py" -type f | sort : list source files
- git diff : review current changes

TESTING:
- Run ONLY the specific relevant test: python -m pytest tests/test_X.py -x -q
- Do NOT run the full test suite unless needed to understand failures
- Do NOT modify test files
- Stop immediately once tests pass

EFFICIENCY:
- Extract keywords from the issue first, then grep for them
- Read files fully once rather than partially multiple times
- Make one focused fix, verify it, then finalize
- Do not re-explore after making a successful fix
"""


def build_initial_user_prompt(issue: str, repo_summary: str) -> str:
    return f"""Fix this GitHub issue with the minimal, correct code change.

ISSUE:
{issue}

REPOSITORY OVERVIEW:
{repo_summary}

START HERE:
1. Identify the key function/class/file names mentioned or implied in the issue
2. Run: grep -rn "key_term" --include="*.py" . (or appropriate extension)
3. Read the relevant file fully: cat -n filename
4. Understand exactly which lines need to change
5. Make the precise minimal fix
6. Run the specific relevant test to confirm
7. Submit with <final>what changed</final>

Focus: every unnecessary line change reduces quality. Make the minimal correct fix only.
"""


def build_no_command_repair_prompt() -> str:
    return """Your previous response did not contain a valid <command>...</command> block or <final>...</final> block.

Continue by issuing exactly one bash command:

<command>
your command here
</command>

If you have already identified the fix location, make the edit now.
If not, search for the relevant code:
<command>
grep -rn "key_term" --include="*.py" .
</command>
"""


def build_stuck_recovery_prompt(step: int, max_steps: int) -> str:
    remaining = max_steps - step
    return f"""You have {remaining} steps remaining. Focus now.

If you have not yet found the relevant file: grep for the key terms from the issue.
If you found the file but have not edited it: make the edit now using sed or python3.
If you edited the code: run the relevant test to verify.
If tests pass: submit with <final>summary</final>.

Issue one focused command:
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
    consecutive_no_command = 0
    steps_since_last_change = 0

    try:
        repo = _repo_path(repo_path)
        model_name, api_base, api_key = _resolve_inference_config(model, api_base, api_key)
        ensure_git_repo(repo)
        repo_summary = get_repo_summary(repo)

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_initial_user_prompt(issue, repo_summary)},
        ]

        last_patch_hash = ""

        for step in range(1, max_steps + 1):
            logs.append(f"\n\n===== STEP {step} =====\n")

            # Trim messages to avoid context overflow
            messages = _trim_messages(messages, max_messages=_MAX_CONTEXT_MESSAGES)

            try:
                response_text, cost, _raw = chat_completion(
                    messages=messages,
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

            command = extract_command(response_text)

            if command is None:
                consecutive_no_command += 1
                if consecutive_no_command >= 2:
                    logs.append("\nSTOPPED: Model failed to produce valid commands twice.")
                    break
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content": build_no_command_repair_prompt()})
                continue

            consecutive_no_command = 0
            result = run_command(command, repo, timeout=command_timeout)
            observation = format_observation(result)

            logs.append("\nOBSERVATION:\n" + observation)

            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": observation})

            # Track whether the patch is changing
            current_patch = get_patch(repo)
            current_patch_hash = str(hash(current_patch))
            if current_patch_hash != last_patch_hash:
                last_patch_hash = current_patch_hash
                steps_since_last_change = 0
            else:
                steps_since_last_change += 1

            # Auto-stop: patch exists and recent command looks like passing tests
            if step >= 3 and current_patch.strip():
                if _looks_like_successful_test_output(observation):
                    logs.append("\nAUTO_STOP:\nPatch exists and latest command showed passing tests.")
                    success = True
                    break

            # Inject a recovery nudge when the agent seems stuck (no patch change for 5+ steps)
            if steps_since_last_change >= 5 and step < max_steps - 2:
                messages.append({
                    "role": "user",
                    "content": build_stuck_recovery_prompt(step, max_steps),
                })
                steps_since_last_change = 0

        patch = get_patch(repo)
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


def _looks_like_successful_test_output(observation: str) -> bool:
    lower = observation.lower()

    # Strong failure signals
    bad_markers = [
        " failed",
        " failures",
        "traceback",
        "assertionerror",
        "syntaxerror",
        "exit_code:\n1",
        "exit_code:\n2",
        "exit_code:\n124",
        "failed:",
        "failures:",
        "error:",
        "importerror",
        "modulenotfounderror",
    ]

    # Strong success signals (test runners)
    good_markers = [
        " passed",
        "all tests passed",
        "tests passed",
        "passed all",
        "exit_code:\n0",
        "ok\n",
        "test session starts",
        " passed,",
    ]

    has_good = any(marker in lower for marker in good_markers)
    has_bad = any(marker in lower for marker in bad_markers)

    # Exit code 0 + "passed" is the strongest positive signal
    if "exit_code:\n0" in lower and "passed" in lower:
        return True

    return has_good and not has_bad


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
