from __future__ import annotations
import re
import time
from dataclasses import dataclass, field
from typing import List

from .environment import compress_message_content, execute_command, truncate_text
from .model import ChatModel, ModelQueryError
from .prompts import (
    COMPLETION_SENTINEL,
    SYSTEM_PROMPT,
    build_task_prompt,
    format_help_message,
    render_observation,
)
from .repo_diff import collect_repo_patch

_ACTION_BLOCK_RE = re.compile(r"```(?:bash|sh)?\s*\n(.*?)\n?```", re.DOTALL)
_MAX_FORMAT_RETRIES = 3
_RECENT_MESSAGES_FULL = 6
_COMPRESS_TRIGGER_CHARS = 800
_COMPRESSED_MESSAGE_CHARS = 1600
_COMPRESSED_FLOOR_CHARS = 500
_MIN_REST_MESSAGES = 4
_KEYWORD_FILE_RE = re.compile(
    r"`?([\w./-]+\.(?:py|ts|tsx|js|jsx|go|rs|java|cs|cpp|hpp|c|h|php|rb))\b`?",
    re.I,
)
_KEYWORD_SYMBOL_RE = re.compile(
    r"`([A-Za-z_][\w.]*)`"
    r"|\b([A-Z][a-zA-Z0-9]{2,})\b"
    r"|\b([a-z][a-z0-9]*(?:_[a-z][a-z0-9_]+)+)\b",
)
_KEYWORD_SKIP = frozenset({
    "the", "and", "for", "with", "from", "this", "that", "task", "issue", "file",
    "files", "test", "tests", "class", "function", "method", "implement", "create",
    "add", "fix", "update", "change", "remove", "delete", "ensure", "make", "use",
})


@dataclass
class AgentRunConfig:
    repo_dir: str
    model_name: str
    base_url: str
    auth_token: str
    max_steps: int = 50
    command_timeout: int = 15
    max_tokens: int = 8192
    max_observation_chars: int = 16000
    max_log_chars: int = 260000
    max_message_chars: int = 120000
    issue_text: str = ""
    wall_clock_limit: float = 0.0


@dataclass
class AgentOutcome:
    success: bool
    patch: str
    logs: str
    steps: int
    cost: float | None
    message: str
    exit_status: str = "Submitted"
    transcript: list = field(default_factory=list)


def extract_task_keywords(task_text: str, limit: int = 8) -> List[str]:
    """Symbol-like terms from the issue for message compression."""
    seen: List[str] = []
    for match in _KEYWORD_FILE_RE.finditer(task_text or ""):
        path = match.group(1).strip().lstrip("./")
        base = path.rsplit("/", 1)[-1]
        for term in (path, base, base.rsplit(".", 1)[0] if "." in base else base):
            low = term.lower()
            if low in _KEYWORD_SKIP or len(low) < 3:
                continue
            if low not in seen:
                seen.append(low)
            if len(seen) >= limit:
                return seen
    for match in _KEYWORD_SYMBOL_RE.finditer(task_text or ""):
        term = next(g for g in match.groups() if g)
        low = term.lower()
        if low in _KEYWORD_SKIP or len(low) < 3:
            continue
        if low not in seen:
            seen.append(low)
        if len(seen) >= limit:
            break
    return seen


def run_agent_loop(*, config: AgentRunConfig, task: str) -> AgentOutcome:
    model = ChatModel(
        model_name=config.model_name,
        base_url=config.base_url,
        auth_token=config.auth_token,
        max_completion_tokens=config.max_tokens,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task if "<task>" in task else build_task_prompt(task_text=task)},
    ]
    started = time.monotonic()
    log_lines: list = []
    exit_status = "LimitsExceeded"
    message = f"step limit of {config.max_steps} reached"
    format_retries = 0
    task_keywords = extract_task_keywords(config.issue_text)

    for step in range(1, max(1, config.max_steps) + 1):
        if 0 < config.wall_clock_limit <= time.monotonic() - started:
            exit_status = "TimeExceeded"
            message = f"wall clock limit of {config.wall_clock_limit:.0f}s reached"
            break
        messages[:] = _cap_messages(messages, config.max_message_chars, task_keywords)
        try:
            reply = model.query(messages)
        except ModelQueryError as exc:
            exit_status = "ModelError"
            message = str(exc)
            log_lines.append(f"[step {step}] model error: {exc}")
            break
        messages.append({"role": "assistant", "content": reply})
        log_lines.append(f"[step {step}] assistant:\n{reply}")

        actions = _ACTION_BLOCK_RE.findall(reply)
        commands = [action.strip() for action in actions if action.strip()]
        if len(commands) != 1:
            format_retries += 1
            if format_retries > _MAX_FORMAT_RETRIES:
                exit_status = "FormatError"
                message = "model kept replying without exactly one bash code block"
                break
            messages.append({"role": "user", "content": format_help_message()})
            log_lines.append(f"[step {step}] format retry {format_retries}")
            continue
        format_retries = 0
        command = commands[0]

        result = execute_command(command, cwd=config.repo_dir, timeout=config.command_timeout)
        output_text = result.get("output") or ""
        log_lines.append(f"[step {step}] $ {command}\n{truncate_text(output_text, 2000)}")
        if _is_submission(output_text, result.get("returncode")):
            exit_status = "Submitted"
            message = f"submitted after {step} step(s)"
            break
        observation = render_observation(
            returncode=int(result.get("returncode") or 0),
            output_text=truncate_text(output_text, config.max_observation_chars),
            remaining_steps=config.max_steps - step,
        )
        messages.append({"role": "user", "content": observation})

    patch = collect_repo_patch(config.repo_dir)
    logs = truncate_text("\n".join(log_lines), config.max_log_chars)
    return AgentOutcome(
        success=bool(patch.strip()),
        patch=patch,
        logs=logs,
        steps=model.calls,
        cost=None,
        message=message,
        exit_status=exit_status,
        transcript=messages,
    )


def _message_chars(messages: list) -> int:
    return sum(len(str(m.get("content") or "")) for m in messages)


def _cap_messages(messages: list, max_chars: int, keywords: list[str]) -> list:
    """Keep system + task; compress old turns in two passes; drop pairs last."""
    if max_chars <= 0 or _message_chars(messages) <= max_chars:
        return messages
    if len(messages) <= 2:
        return messages

    pinned = [{**m} for m in messages[:2]]
    rest = [{**m, "content": str(m.get("content") or "")} for m in messages[2:]]

    recent_start = max(0, len(rest) - _RECENT_MESSAGES_FULL)
    compressed_pass: dict[int, int] = {}

    while rest and _message_chars(pinned + rest) > max_chars:
        compress_idx = None
        best_len = 0
        for idx in range(recent_start):
            content = rest[idx]["content"]
            clen = len(content)
            if clen <= _COMPRESSED_FLOOR_CHARS:
                continue
            passes = compressed_pass.get(idx, 0)
            if passes == 0 and clen <= _COMPRESS_TRIGGER_CHARS:
                continue
            limit = _COMPRESSED_MESSAGE_CHARS if passes == 0 else _COMPRESSED_FLOOR_CHARS
            if clen > limit and clen > best_len:
                best_len = clen
                compress_idx = idx

        if compress_idx is not None:
            passes = compressed_pass.get(compress_idx, 0)
            limit = _COMPRESSED_MESSAGE_CHARS if passes == 0 else _COMPRESSED_FLOOR_CHARS
            content = rest[compress_idx]["content"]
            shrunk = compress_message_content(content, keywords=keywords, limit=limit)
            if shrunk != content:
                rest[compress_idx] = {**rest[compress_idx], "content": shrunk}
                compressed_pass[compress_idx] = passes + 1
                continue

        if len(rest) <= _MIN_REST_MESSAGES:
            break

        if (
            len(rest) >= 2
            and rest[0].get("role") == "assistant"
            and rest[1].get("role") == "user"
        ):
            rest = rest[2:]
        else:
            rest = rest[1:]
    return pinned + rest


def _is_submission(output_text: str, returncode) -> bool:
    lines = output_text.lstrip().splitlines()
    return bool(lines) and lines[0].strip() == COMPLETION_SENTINEL and not returncode
