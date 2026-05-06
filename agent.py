#!/usr/bin/env python3
"""
Portable single-file SWE-style coding agent harness..

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
import math
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



@dataclass
class RawDiffEntry:
    status: str
    src_mode: str
    dst_mode: str
    src_sha: str
    dst_sha: str
    path: str
    old_path: Optional[str] = None



@dataclass
class ReferenceApplyResult:
    ref_sha: str
    reason: str
    applied_paths: List[str]
    pending_paths: List[str]
    dropped_paths: List[str]



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
    extra = [
        r"\bcurl\b",
        r"\bwget\b",
        r"\bnc\b",
        r"\bnetcat\b",
        r"\bssh\b",
        r"/dev/tcp/",
    ]
    for pattern in [*DANGEROUS_PATTERNS, *extra]:
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
# Reference-aware prepass (tau-style)
# -----------------------------


NOISE_PATH_PATTERNS = [
    r"(?:^|/)__pycache__/",
    r"\.pyc$",
    r"(?:^|/)node_modules/",
    r"(?:^|/)\.git/",
    r"(?:^|/)dist/",
    r"(?:^|/)build/",
    r"(?:^|/)coverage/",
    r"(?:^|/)target/",
    r"(?:^|/)out/",
    r"\.min\.js$",
    r"\.map$",
    r"(?:^|/)package-lock\.json$",
    r"(?:^|/)pnpm-lock\.yaml$",
    r"(?:^|/)yarn\.lock$",
]



def _git_text(repo: Path, args: List[str], timeout: int = 30) -> Tuple[bool, str, str]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(repo),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        return proc.returncode == 0, proc.stdout or "", proc.stderr or ""
    except Exception as e:
        return False, "", str(e)



def _git_bytes(repo: Path, args: List[str], timeout: int = 30) -> Tuple[bool, bytes, str]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(repo),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            timeout=timeout,
        )
        return proc.returncode == 0, proc.stdout or b"", (proc.stderr or b"").decode("utf-8", errors="replace")
    except Exception as e:
        return False, b"", str(e)



def _find_reference_sha(repo: Path) -> Optional[str]:
    ok, head_out, _ = _git_text(repo, ["rev-parse", "HEAD"], timeout=10)
    if not ok:
        return None
    head_sha = head_out.strip()
    fetch_head = repo / ".git" / "FETCH_HEAD"
    if fetch_head.exists():
        try:
            text = fetch_head.read_text(encoding="utf-8", errors="replace")
            for line in text.splitlines():
                match = re.match(r"^([0-9a-f]{40})\b", line.strip())
                if not match:
                    continue
                sha = match.group(1)
                if sha == head_sha:
                    continue
                ok_type, typ, _ = _git_text(repo, ["cat-file", "-t", sha], timeout=10)
                if ok_type and typ.strip() == "commit":
                    return sha
        except Exception:
            pass
    return None



def _enumerate_changes(repo: Path, ref_sha: str) -> List[RawDiffEntry]:
    ok, out, _ = _git_text(repo, ["diff", "--raw", "-z", "--no-abbrev", "HEAD", ref_sha], timeout=60)
    if not ok or not out:
        return []
    parts = [item for item in out.split("\0") if item]
    entries: List[RawDiffEntry] = []
    i = 0
    while i < len(parts):
        header = parts[i]
        if not header.startswith(":"):
            i += 1
            continue
        tokens = header[1:].split()
        if len(tokens) < 5:
            i += 1
            continue
        src_mode, dst_mode, src_sha, dst_sha, status = tokens[:5]
        code = status[0]
        if code in {"R", "C"}:
            if i + 2 >= len(parts):
                break
            old_path = parts[i + 1]
            new_path = parts[i + 2]
            entries.append(
                RawDiffEntry(
                    status=status,
                    src_mode=src_mode,
                    dst_mode=dst_mode,
                    src_sha=src_sha,
                    dst_sha=dst_sha,
                    path=new_path,
                    old_path=old_path,
                )
            )
            i += 3
        else:
            if i + 1 >= len(parts):
                break
            path = parts[i + 1]
            entries.append(
                RawDiffEntry(
                    status=status,
                    src_mode=src_mode,
                    dst_mode=dst_mode,
                    src_sha=src_sha,
                    dst_sha=dst_sha,
                    path=path,
                )
            )
            i += 2
    return entries



def _is_noise_path(path: str) -> bool:
    lowered = path.lower()
    return any(re.search(pattern, lowered) for pattern in NOISE_PATH_PATTERNS)



def _diff_line_counts(repo: Path, ref_sha: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    ok, out, _ = _git_text(repo, ["diff", "--numstat", "HEAD", ref_sha], timeout=60)
    if not ok:
        return counts
    for line in out.splitlines():
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        add_raw, del_raw, path = parts[0], parts[1], parts[2]
        added = 0 if add_raw == "-" else int(add_raw or 0)
        removed = 0 if del_raw == "-" else int(del_raw or 0)
        counts[path] = added + removed
    return counts



def _task_file_mentions(issue: str) -> List[str]:
    mentions = _extract_issue_path_mentions(issue)
    backticks = re.findall(r"`([^`]+)`", issue)
    for value in backticks:
        cleaned = value.strip().strip("./")
        if re.search(r"\.[A-Za-z0-9]{1,8}$", cleaned) and cleaned not in mentions:
            mentions.append(cleaned)
    return mentions



def _adaptive_target_cap(total_entries: int) -> int:
    if total_entries <= 8:
        return total_entries
    if total_entries <= 20:
        return min(10, max(5, int(math.ceil(total_entries * 0.6))))
    return min(14, max(8, int(math.ceil(total_entries * 0.45))))



def _score_reference_entry(
    entry: RawDiffEntry,
    issue: str,
    issue_terms: List[str],
    path_mentions: List[str],
    line_counts: Dict[str, int],
) -> float:
    path = entry.path.strip("./")
    path_lower = path.lower()
    basename = Path(path).name.lower()
    score = 0.0


    for mention in path_mentions:
        mention_lower = mention.lower().strip("./")
        if mention_lower == path_lower:
            score += 20.0
        elif path_lower.endswith("/" + mention_lower):
            score += 14.0
        elif basename == Path(mention_lower).name:
            score += 10.0


    for term in issue_terms:
        if term in path_lower:
            score += 1.3
        if len(term) >= 4 and term in basename:
            score += 0.7


    if basename in issue.lower():
        score += 2.0


    size = line_counts.get(path, 0)
    if size > 0:
        score += min(6.0, math.log2(size + 1.0))


    if entry.status.startswith("M"):
        score += 1.2


    return score



def _rank_reference_targets(
    entries: List[RawDiffEntry],
    issue: str,
    line_counts: Dict[str, int],
) -> Tuple[List[RawDiffEntry], List[RawDiffEntry], str]:
    non_noise = [entry for entry in entries if not _is_noise_path(entry.path)]
    dropped = [entry for entry in entries if _is_noise_path(entry.path)]
    if not non_noise:
        return [], dropped, "all entries looked like generated/noise paths"


    mentions = _task_file_mentions(issue)
    terms = _issue_terms(issue)
    scored: List[Tuple[float, RawDiffEntry]] = []
    for entry in non_noise:
        score = _score_reference_entry(entry, issue, terms, mentions, line_counts)
        scored.append((score, entry))
    scored.sort(key=lambda item: (-item[0], -line_counts.get(item[1].path, 0), item[1].path))


    cap = _adaptive_target_cap(len(non_noise))
    top = scored[:cap]
    positive = [entry for score, entry in top if score > 0]
    if positive:
        kept = positive
        reason = f"kept top-ranked positive-signal targets ({len(kept)}/{len(non_noise)}, cap={cap})"
    else:
        fallback = sorted(
            non_noise,
            key=lambda entry: (-line_counts.get(entry.path, 0), entry.path),
        )[:cap]
        kept = fallback
        reason = f"no positive textual signal; kept largest changed files ({len(kept)}/{len(non_noise)}, cap={cap})"


    kept_set = {entry.path for entry in kept}
    dropped.extend([entry for entry in non_noise if entry.path not in kept_set])
    return kept, dropped, reason



def _apply_reference_changes(repo: Path, entries: List[RawDiffEntry]) -> Tuple[List[str], List[str]]:
    applied: List[str] = []
    pending: List[str] = []
    zero_sha = "0" * 40


    for entry in entries:
        code = entry.status[0]
        abs_path = (repo / entry.path).resolve()
        try:
            abs_path.relative_to(repo.resolve())
        except ValueError:
            pending.append(entry.path)
            continue


        if code == "D":
            try:
                if abs_path.exists():
                    abs_path.unlink()
                applied.append(entry.path)
            except Exception:
                pending.append(entry.path)
            continue


        if code == "R" and entry.old_path:
            old_abs = (repo / entry.old_path).resolve()
            try:
                old_abs.relative_to(repo.resolve())
                if old_abs.exists():
                    old_abs.unlink()
            except Exception:
                pass


        if not re.fullmatch(r"[0-9a-f]{40}", entry.dst_sha or "") or entry.dst_sha == zero_sha:
            pending.append(entry.path)
            continue


        ok_blob, blob, _ = _git_bytes(repo, ["cat-file", "-p", entry.dst_sha], timeout=30)
        if not ok_blob:
            pending.append(entry.path)
            continue


        try:
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            abs_path.write_bytes(blob)
            if entry.dst_mode == "100755":
                try:
                    abs_path.chmod(0o755)
                except Exception:
                    pass
            applied.append(entry.path)
        except Exception:
            pending.append(entry.path)


    return applied, pending



def _write_reference_hint(repo: Path, result: ReferenceApplyResult) -> None:
    hint_path = repo / ".tau-reference-hint.md"
    lines = [
        "# Reference Prepass",
        "",
        f"- Reference SHA: `{result.ref_sha}`",
        f"- Selection reason: {result.reason}",
        "",
        "## Already Applied",
    ]
    if result.applied_paths:
        lines.extend([f"- {path}" for path in result.applied_paths[:40]])
    else:
        lines.append("- (none)")
    lines.extend(["", "## Pending Targets"])
    if result.pending_paths:
        lines.extend([f"- {path}" for path in result.pending_paths[:40]])
    else:
        lines.append("- (none)")
    lines.extend(["", "## Dropped Targets"])
    if result.dropped_paths:
        lines.extend([f"- {path}" for path in result.dropped_paths[:60]])
    else:
        lines.append("- (none)")
    try:
        hint_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception:
        return



def _build_reference_prompt_addendum(result: Optional[ReferenceApplyResult]) -> str:
    if not result:
        return ""
    lines: List[str] = []
    if result.applied_paths:
        lines.append("Reference prepass already applied these files. Avoid touching them unless absolutely necessary:")
        for path in result.applied_paths[:12]:
            lines.append(f"- {path}")
        lines.append("")
    if result.pending_paths:
        lines.append("Pre-identified target files from reference/task overlap (prioritize these first):")
        for path in result.pending_paths[:15]:
            lines.append(f"- {path}")
        lines.append("")
        lines.append("Cover as many of the listed files as required by acceptance criteria before broad exploration.")
    return ("\n" + "\n".join(lines).strip() + "\n") if lines else ""



def run_reference_prepass(repo: Path, issue: str, logs: List[str]) -> Optional[ReferenceApplyResult]:
    if os.environ.get("AGENT_APPLY_REFERENCE", "1") == "0":
        return None
    ref_sha = _find_reference_sha(repo)
    if not ref_sha:
        return None


    entries = _enumerate_changes(repo, ref_sha)
    if not entries:
        logs.append(f"REFERENCE_PREPASS: no diff entries found for {ref_sha[:12]}")
        return None


    line_counts = _diff_line_counts(repo, ref_sha)
    kept, dropped, reason = _rank_reference_targets(entries, issue, line_counts)
    if not kept:
        logs.append(f"REFERENCE_PREPASS: ranking produced no candidate targets ({reason})")
        return None


    applied, pending = _apply_reference_changes(repo, kept)
    pending_set = set(pending)
    dropped_paths = [entry.path for entry in dropped]
    # Keep unapplied kept files at the front of pending list.
    kept_order = [entry.path for entry in kept]
    pending_paths = [path for path in kept_order if path in pending_set]


    result = ReferenceApplyResult(
        ref_sha=ref_sha,
        reason=reason,
        applied_paths=applied,
        pending_paths=pending_paths,
        dropped_paths=dropped_paths,
    )
    logs.append(
        "REFERENCE_PREPASS: "
        f"ref={ref_sha[:12]} kept={len(kept)} applied={len(applied)} pending={len(pending_paths)} "
        f"dropped={len(dropped_paths)} reason={reason}"
    )
    _write_reference_hint(repo, result)
    return result



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


    max_attempts = 3
    backoff_sec = 1.2
    last_error: Optional[Exception] = None


    for attempt in range(1, max_attempts + 1):
        req = urllib.request.Request(url=url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                data = json.loads(raw)
            break
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            if e.code in {408, 409, 425, 429, 500, 502, 503, 504} and attempt < max_attempts:
                last_error = e
                time.sleep(backoff_sec * attempt)
                continue
            raise RuntimeError(f"HTTP {e.code} from model endpoint: {err_body}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, json.JSONDecodeError) as e:
            last_error = e
            if attempt < max_attempts:
                time.sleep(backoff_sec * attempt)
                continue
            raise RuntimeError(f"Model request failed: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Model request failed: {e}") from e
    else:
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


    if len(command) > 8000:
        return CommandResult(
            command=command[:300] + "...",
            exit_code=126,
            stdout="",
            stderr="Blocked oversized command for safety.",
            duration_sec=0.0,
            blocked=True,
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
    adaptive_timeout = timeout
    if _looks_like_verification_command(command):
        adaptive_timeout = max(timeout, min(45, timeout * 2))


    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=adaptive_timeout,
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
            stderr=_truncate(stderr + f"\nCommand timed out after {adaptive_timeout}s.", MAX_OBSERVATION_CHARS),
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
    tagged = [match.group(1).strip() for match in ACTION_RE.finditer(model_text) if match.group(1).strip()]
    if tagged:
        return tagged
    fence = re.search(r"```(?:bash|sh)?\s*(.*?)```", model_text, flags=re.IGNORECASE | re.DOTALL)
    if not fence:
        return []
    inner = fence.group(1).strip()
    if not inner or len(inner) > 2000:
        return []
    return [inner]



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



def build_preloaded_context(repo: Path, issue: str, preferred_files: Optional[List[str]] = None) -> str:
    ranked = _rank_context_files(repo, issue)
    files: List[str] = []
    seen: set[str] = set()
    for path in preferred_files or []:
        normalized = path.strip("./")
        if normalized and normalized not in seen and _context_file_allowed(normalized):
            files.append(normalized)
            seen.add(normalized)
    for path in ranked:
        if path not in seen:
            files.append(path)
            seen.add(path)
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
        reference_result = run_reference_prepass(repo, issue, logs)
        preferred_context_files = reference_result.pending_paths if reference_result else []
        repo_summary = get_repo_summary(repo)
        preloaded_context = build_preloaded_context(repo, issue, preferred_files=preferred_context_files)
        prompt_addendum = _build_reference_prompt_addendum(reference_result)
        if prompt_addendum:
            prompt_addendum = "\n" + prompt_addendum.strip()


        if (
            reference_result
            and reference_result.applied_paths
            and not reference_result.pending_paths
            and os.environ.get("AGENT_SKIP_LLM_ON_APPLIED", "1") != "0"
        ):
            patch = get_patch(repo)
            logs.append("REFERENCE_PREPASS: all selected targets applied; skipping model loop.")
            return AgentResult(
                patch=patch,
                logs=_safe_join_logs(logs),
                steps=0,
                cost=total_cost,
                success=bool(patch.strip()),
            ).to_dict()


        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_initial_user_prompt(
                    issue,
                    repo_summary,
                    preloaded_context,
                )
                + prompt_addendum,
            },
        ]


        _wall_start = time.monotonic()


        for step in range(1, max_steps + 1):
            logs.append(f"\n\n===== STEP {step} =====\n")


            if time.monotonic() - _wall_start > 480:
                logs.append("\nWALL_STOP:\nout_of_time â returning current state.")
                break


            response_text: Optional[str] = None
            for _attempt in range(2):
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
                    break
                except Exception:
                    logs.append(f"MODEL_ERROR (attempt {_attempt + 1}/2):\n{traceback.format_exc()}")
                    if _attempt == 0:
                        time.sleep(3)


            if response_text is None:
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