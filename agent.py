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

DEFAULT_MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", "40"))
DEFAULT_COMMAND_TIMEOUT = int(os.environ.get("AGENT_COMMAND_TIMEOUT", "30"))

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

MAX_OBSERVATION_CHARS = int(os.environ.get("AGENT_MAX_OBSERVATION_CHARS", "12000"))
MAX_TOTAL_LOG_CHARS = int(os.environ.get("AGENT_MAX_TOTAL_LOG_CHARS", "200000"))

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
# Utility (validator-owned - do not edit)
# -----------------------------

def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + f"\n...[truncated {len(text)-max_chars} chars]...\n" + text[-half:]


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
        raise ValueError("model is required")
    if not base:
        raise ValueError("api_base is required")
    if not key:
        raise ValueError("api_key is required")

    return model_name, _normalize_api_base(base), key


def _is_dangerous_command(command: str) -> Optional[str]:
    lowered = command.strip().lower()
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
# OpenAI-compatible client (validator-owned)
# -----------------------------

def chat_completion(
    messages: List[Dict[str, str]],
    model: str,
    api_base: Optional[str],
    api_key: Optional[str],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: int = 120,
) -> Tuple[str, Optional[float], Dict[str, Any]]:
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
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read())
            content = data["choices"][0]["message"]["content"]
            cost = None
            return content, cost, data
    except Exception as e:
        return f"LLM call failed: {e}", None, {}


# =============================================================================
# MINER-EDITABLE SECTION - ONLY EDIT BELOW THIS LINE
# =============================================================================

# MINER-EDITABLE: This prompt is the main behavior policy for the inner coding
# agent. Prompt improvements are encouraged.
SYSTEM_PROMPT = """You are an expert software engineer specializing in SWE-bench tasks. Your ONLY goal is to produce the smallest possible unified git diff that maximizes positional line-level exact matching against the hidden reference solution.

You interact only by issuing bash commands. The environment will run your command and return stdout/stderr.

STRICT RULES - follow these exactly:
1. BREADTH-FIRST: Touch as many target files as possible with small, precise edits. Don't spend too long on one file.
2. MINIMAL DIFFS: Match indentation, quotes, line endings, trailing commas EXACTLY. Character-identical to reference.
3. NO TESTS/LINTERS/FORMATTING: Don't run or add tests, don't format code, don't add comments.
4. TWO-PHASE: First understand task + context, then make surgical edits.
5. STOP EARLY: As soon as all acceptance criteria are addressed, output <final>.
6. READ FIRST: Read every mentioned file once (alphabetical order) before editing.

Use this exact format for commands:

<command>
your bash command here
</command>

When you are finished, respond with:

<final>
short summary of what you changed
</final>

NEVER: sudo, delete repo, modify hidden tests, make network calls except validator proxy, run linters/formatters.
ALWAYS: Work in the repo, make small targeted edits, inspect files before editing.
"""


# =============================================================================
# SOLVE FUNCTION - THIS IS THE ONLY PART YOU ARE ALLOWED TO EDIT
# =============================================================================
def solve(
    repo_path: str,
    issue: str,
    model: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    start_time = time.time()
    logs: List[str] = [f"Starting solve with model: {model or DEFAULT_MODEL}"]

    try:
        repo = _repo_path(repo_path)
        os.chdir(repo)

        # Gather context
        ls_out = subprocess.getoutput("ls -la")
        git_status = subprocess.getoutput("git status --porcelain")

        context = f"""Task:
{issue}

Repository contents:
{ls_out}

Git status:
{git_status}

Produce the smallest possible unified diff that solves this task."""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ]

        logs.append("Calling validator model...")

        response, cost, raw = chat_completion(
            messages=messages,
            model=model,
            api_base=api_base,
            api_key=api_key,
        )

        logs.append(f"Model response received ({len(response)} chars)")

        patch = response.strip()

        logs.append(f"Completed in {int(time.time() - start_time)} seconds")

        return AgentResult(
            patch=patch,
            logs=_safe_join_logs(logs),
            steps=1,
            cost=cost,
            success=True,
        ).to_dict()

    except Exception as e:
        logs.append(f"Error: {traceback.format_exc()}")
        return AgentResult(
            patch="",
            logs=_safe_join_logs(logs),
            steps=0,
            cost=None,
            success=False,
        ).to_dict()


if __name__ == "__main__":
    print("This file is meant to be imported by the validator.")
