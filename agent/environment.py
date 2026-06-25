import os
import re
import subprocess

_QUIET_TOOL_DEFAULTS = {
    "PAGER": "cat",
    "MANPAGER": "cat",
    "LESS": "-R",
    "PIP_PROGRESS_BAR": "off",
    "TQDM_DISABLE": "1",
    "NO_COLOR": "1",
    "GIT_PAGER": "cat",
    "PYTHONDONTWRITEBYTECODE": "1",
}

_ERROR_LINE_RE = re.compile(
    r"Traceback \(most recent call last\)|"
    r"\b(Error|Exception|FAILED|Failure|AssertionError|SyntaxError|"
    r"TypeError|ValueError|NameError|ImportError|ModuleNotFoundError|"
    r"panic:|fatal error|ENOENT|No such file)\b",
    re.I,
)
_CONTEXT_LINES = 2
_MAX_PICKED_LINES = 120


def execute_command(command: str, *, cwd: str, timeout: int) -> dict:
    env = os.environ.copy()
    env.update(_QUIET_TOOL_DEFAULTS)
    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(1, int(timeout)),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return {"output": completed.stdout or "", "returncode": completed.returncode}
    except subprocess.TimeoutExpired as exc:
        partial = exc.output or ""
        if isinstance(partial, bytes):
            partial = partial.decode("utf-8", errors="replace")
        return {
            "output": f"{partial}\n[command timed out after {timeout} seconds]",
            "returncode": 124,
        }
    except (OSError, ValueError) as exc:
        return {"output": f"[command could not be executed: {exc}]", "returncode": -1}


def truncate_text(text: str, limit: int) -> str:
    """Head/tail elision so long outputs keep their start and end visible."""
    if limit <= 0 or len(text) <= limit:
        return text
    elision = "\n[... truncated ...]\n"
    budget = limit - len(elision)
    if budget < 2:
        return text[:limit]
    half = max(1, budget // 2)
    return f"{text[:half]}{elision}{text[-half:]}"


def _keyword_patterns(keywords: list[str]) -> list[re.Pattern[str]]:
    patterns: list[re.Pattern[str]] = []
    for keyword in keywords:
        term = keyword.strip()
        if len(term) < 3:
            continue
        patterns.append(re.compile(re.escape(term), re.I))
    return patterns


def _lines_with_context(lines: list[str], indices: set[int]) -> list[str]:
    picked: set[int] = set()
    for i in indices:
        for j in range(max(0, i - _CONTEXT_LINES), min(len(lines), i + _CONTEXT_LINES + 1)):
            picked.add(j)
    ordered = sorted(picked)
    if len(ordered) > _MAX_PICKED_LINES:
        half = _MAX_PICKED_LINES // 2
        ordered = ordered[:half] + ordered[-half:]
    return [lines[i] for i in ordered]


def compress_message_content(
    text: str,
    *,
    keywords: list[str] | None = None,
    limit: int,
) -> str:
    """Shrink one chat message by keeping error/task-keyword lines; else truncate."""
    if limit <= 0 or len(text) <= limit:
        return text
    lines = text.splitlines()
    if not lines:
        return text

    hit_indices: set[int] = set()
    for i, line in enumerate(lines):
        if _ERROR_LINE_RE.search(line):
            hit_indices.add(i)
    for pattern in _keyword_patterns(keywords or []):
        for i, line in enumerate(lines):
            if pattern.search(line):
                hit_indices.add(i)

    if not hit_indices:
        return truncate_text(text, limit)

    picked_lines = _lines_with_context(lines, hit_indices)
    compressed = "\n".join(picked_lines)
    if len(picked_lines) < len(lines):
        header = (
            f"[message compressed: {len(picked_lines)} of {len(lines)} lines "
            f"with errors or task keywords]\n"
        )
        compressed = header + compressed

    if len(compressed) <= limit:
        return compressed
    return truncate_text(compressed, limit)
