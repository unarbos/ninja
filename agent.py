#!/usr/bin/env python3
# V2 Improved Coding Agent (Merged Upgrade)

from __future__ import annotations
import json, os, re, subprocess, time, traceback, urllib.request, urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

# -----------------------------
# Config
# -----------------------------
DEFAULT_MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", "30"))
DEFAULT_COMMAND_TIMEOUT = int(os.environ.get("AGENT_COMMAND_TIMEOUT", "15"))
DEFAULT_MAX_TOKENS = int(os.environ.get("AGENT_MAX_TOKENS", "2048"))

MAX_OBSERVATION_CHARS = 9000
MAX_TOTAL_LOG_CHARS = 180000
MAX_CONVERSATION_CHARS = 60000
MAX_NO_COMMAND_REPAIRS = 3

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

    def to_dict(self):
        return self.__dict__

# -----------------------------
# Utility
# -----------------------------
def _truncate(text: str, max_chars: int):
    return text if len(text) <= max_chars else text[:max_chars]

def _safe_join_logs(logs):
    return _truncate("\n".join(logs), MAX_TOTAL_LOG_CHARS)

def _repo_path(path):
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    return p

# -----------------------------
# Safety
# -----------------------------
def _is_dangerous_command(command: str):
    if ".." in command:
        return "path traversal"
    if re.search(r"\brm\s+-rf\b", command):
        return "destructive delete"
    if re.search(r"\bcurl\b|\bwget\b", command):
        return "network access"
    return None

# -----------------------------
# OpenAI-compatible client
# -----------------------------
def chat_completion(messages, model, api_base, api_key, max_tokens):
    url = api_base.rstrip("/") + "/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }).encode()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    )

    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read().decode())

    return data["choices"][0]["message"]["content"], 0.0, data

# -----------------------------
# Command execution
# -----------------------------
def run_command(cmd, cwd, timeout):
    if not cmd.strip():
        return CommandResult(cmd, 0, "", "empty command", 0)

    danger = _is_dangerous_command(cmd)
    if danger:
        return CommandResult(cmd, 126, "", f"blocked: {danger}", 0, blocked=True)

    start = time.time()
    try:
        p = subprocess.run(
            cmd,
            shell=True,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            executable="/bin/bash",
        )
        return CommandResult(
            cmd,
            p.returncode,
            _truncate(p.stdout, MAX_OBSERVATION_CHARS),
            _truncate(p.stderr, MAX_OBSERVATION_CHARS),
            time.time() - start,
        )
    except subprocess.TimeoutExpired as e:
        return CommandResult(cmd, 124, "", "timeout", time.time() - start, timed_out=True)

# -----------------------------
# Parsing
# -----------------------------
ACTION_RE = re.compile(r"<command>(.*?)</command>", re.S)
FINAL_RE = re.compile(r"<final>(.*?)</final>", re.S)

def extract_commands(text):
    return [c.strip() for c in ACTION_RE.findall(text)]

def extract_final(text):
    m = FINAL_RE.search(text)
    return m.group(1).strip() if m else None

# -----------------------------
# Git helpers
# -----------------------------
def ensure_git(repo):
    if not (repo / ".git").exists():
        subprocess.run("git init && git add . && git commit -m init",
                       cwd=repo, shell=True)

def get_patch(repo):
    p = subprocess.run(["git", "diff"], cwd=repo,
                       stdout=subprocess.PIPE, text=True)
    return p.stdout

# -----------------------------
# Improved logic
# -----------------------------
def _looks_like_success(observation, command):
    lower = observation.lower()
    if "error" in lower or "failed" in lower:
        return False
    return True

def _patch_is_meaningful(patch, issue):
    if len(patch) < 40:
        return False
    terms = re.findall(r"[a-zA-Z]{4,}", issue.lower())
    return any(t in patch.lower() for t in terms)

# -----------------------------
# SYSTEM PROMPT (UPGRADED)
# -----------------------------
SYSTEM_PROMPT = """You are an expert coding agent.

Fix the issue with minimal precise edits.

Rules:
- Edit immediately (no long exploration)
- Use Python inline scripts for edits
- Make small safe changes
- Run ONE quick check
- Then finalize

Format:

<command>
bash command
</command>

or

<final>
summary
</final>
"""

# -----------------------------
# MAIN
# -----------------------------
def solve(repo_path, issue, model=None, api_base=None, api_key=None,
          max_steps=DEFAULT_MAX_STEPS,
          command_timeout=DEFAULT_COMMAND_TIMEOUT,
          max_tokens=DEFAULT_MAX_TOKENS):

    logs = []
    success = False
    total_cost = 0.0

    try:
        repo = _repo_path(repo_path)
        ensure_git(repo)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": issue},
        ]

        for step in range(1, max_steps + 1):
            logs.append(f"\nSTEP {step}")

            response, cost, _ = chat_completion(
                messages, model, api_base, api_key, max_tokens
            )
            logs.append(response)

            cmds = extract_commands(response)
            final = extract_final(response)

            if final:
                success = True
                break

            if not cmds:
                messages.append({"role": "user", "content": "Provide a command."})
                continue

            for cmd in cmds:
                result = run_command(cmd, repo, command_timeout)
                obs = f"{result.stdout}\n{result.stderr}"
                logs.append(obs)

                patch = get_patch(repo)
                if _patch_is_meaningful(patch, issue) and _looks_like_success(obs, cmd):
                    success = True
                    break

                messages.append({"role": "user", "content": obs})

            if success:
                break

        patch = get_patch(repo)

        return AgentResult(
            patch=patch,
            logs=_safe_join_logs(logs),
            steps=step,
            cost=total_cost,
            success=success and bool(patch.strip())
        ).to_dict()

    except Exception:
        logs.append(traceback.format_exc())
        return AgentResult("", _safe_join_logs(logs), 0, None, False).to_dict()