#!/usr/bin/env python3
"""Fail external PRs that touch files outside the miner harness."""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

GITHUB_API = "https://api.github.com"
MARKER = "<!-- ninja-pr-scope-guard -->"
DEFAULT_TRUSTED_AUTHORS = ("unarbos",)
DEFAULT_EXTERNAL_ALLOWED_FILES = ("agent.py",)
REQUIRED_SOLVE_ARGS = ("repo_path", "issue", "model", "api_base", "api_key")
MINER_HOTKEY_TITLE_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{32,64}(?:$|[\s:#-])")
ALLOWED_ENV_NAMES = {
    "AGENT_MAX_STEPS",
    "AGENT_COMMAND_TIMEOUT",
    "AGENT_MODEL",
    "NINJA_MODEL",
    "AGENT_API_BASE",
    "NINJA_INFERENCE_BASE_URL",
    "OPENAI_BASE_URL",
    "AGENT_API_KEY",
    "NINJA_INFERENCE_API_KEY",
    "OPENAI_API_KEY",
    "AGENT_MAX_TOKENS",
    "AGENT_MAX_OBSERVATION_CHARS",
    "AGENT_MAX_TOTAL_LOG_CHARS",
}
FORBIDDEN_SAMPLING_NAMES = {
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "top_a",
    "frequency_penalty",
    "presence_penalty",
    "repetition_penalty",
    "seed",
    "logit_bias",
    "logprobs",
    "top_logprobs",
}
PROTECTED_EDIT_MARKERS = (
    "def solve(",
    "repo_path: str,",
    "issue: str,",
    "model: Optional[str] = None,",
    "api_base: Optional[str] = None,",
    "api_key: Optional[str] = None,",
    "def _resolve_inference_config(",
    "DEFAULT_MODEL =",
    "DEFAULT_API_BASE =",
    "DEFAULT_API_KEY =",
    "DEFAULT_TEMPERATURE =",
)
PROTECTED_HUNK_SYMBOLS = ("_resolve_inference_config",)
FORBIDDEN_ADDED_SUBSTRINGS = (
    "openrouter_api_key",
    "anthropic_api_key",
    "gemini_api_key",
    "groq_api_key",
    "together_api_key",
    "fireworks_api_key",
    "mistral_api_key",
    "deepinfra_api_key",
    "github_token",
    "api.openai.com",
    "openrouter.ai",
    "anthropic.com",
    "generativelanguage.googleapis.com",
    "api.groq.com",
    "api.together.xyz",
    "api.fireworks.ai",
    "api.mistral.ai",
    "api.deepseek.com",
    "deepinfra.com",
    "cohere.ai",
    "/proc/self/environ",
    "/proc/environ",
    ".ssh",
    "id_rsa",
    ".netrc",
    "wallet",
)


def main() -> int:
    try:
        event = _load_event()
        repo = _required_env("GITHUB_REPOSITORY")
        token = _required_env("GITHUB_TOKEN")

        pr = event["pull_request"]
        pr_number = int(pr["number"])
        title = str(pr.get("title") or "")
        author = str((pr.get("user") or {}).get("login") or "")
        trusted_authors = _csv_env("TRUSTED_PR_AUTHORS", DEFAULT_TRUSTED_AUTHORS)
        allowed_files = _csv_env("EXTERNAL_PR_ALLOWED_FILES", DEFAULT_EXTERNAL_ALLOWED_FILES)

        files = _fetch_pr_files(token, repo, pr_number)
        changed_files = [str(item.get("filename") or "") for item in files]
        scope_violations = _scope_violations(changed_files, author, trusted_authors, allowed_files)

        if author in trusted_authors:
            body = _render_comment("pass", author, changed_files, allowed_files, [], [])
            _update_existing_comment(token, repo, pr_number, body)
            _write_step_summary(body)
            print(f"Trusted PR author {author}; external file-scope guard bypassed.")
            return 0

        contract_violations = [
            *_title_violations(title),
            *_agent_contract_violations(token, files),
        ]
        if scope_violations or contract_violations:
            body = _render_comment(
                "fail",
                author,
                changed_files,
                allowed_files,
                scope_violations,
                contract_violations,
            )
            _upsert_comment(token, repo, pr_number, body)
            _write_step_summary(body)
            if scope_violations:
                print("External PR changed files outside the allowed surface:")
                for filename in scope_violations:
                    print(f"- {filename}")
            if contract_violations:
                print("External PR violated the agent.py contract:")
                for reason in contract_violations:
                    print(f"- {reason}")
            return 1

        body = _render_comment("pass", author, changed_files, allowed_files, [], [])
        _update_existing_comment(token, repo, pr_number, body)
        _write_step_summary(body)
        print("External PR file scope and agent.py contract are valid.")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"scope guard error: {exc}", file=sys.stderr)
        return 1


def _load_event() -> dict[str, Any]:
    path = Path(_required_env("GITHUB_EVENT_PATH"))
    return json.loads(path.read_text(encoding="utf-8"))


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def _csv_env(name: str, default: tuple[str, ...]) -> set[str]:
    raw = os.environ.get(name)
    values = default if not raw else tuple(raw.split(","))
    parsed = {value.strip() for value in values if value.strip()}
    if not parsed:
        raise RuntimeError(f"{name} must contain at least one value")
    return parsed


def _title_violations(title: str) -> list[str]:
    if MINER_HOTKEY_TITLE_RE.match(title):
        return []
    return ["PR title must start with the committing miner hotkey, for example `<miner-hotkey> improve solver`."]


def _scope_violations(
    changed_files: list[str],
    author: str,
    trusted_authors: set[str],
    allowed_files: set[str],
) -> list[str]:
    if author in trusted_authors:
        return []
    return [filename for filename in changed_files if filename not in allowed_files]


def _agent_contract_violations(token: str, files: list[dict[str, Any]]) -> list[str]:
    agent_file = next((item for item in files if item.get("filename") == "agent.py"), None)
    if not agent_file:
        return ["PR must modify agent.py."]
    if agent_file.get("status") in {"removed", "renamed"}:
        return ["agent.py must not be removed or renamed."]

    violations = _agent_patch_violations(str(agent_file.get("patch") or ""))
    raw_url = str(agent_file.get("raw_url") or "")
    if raw_url:
        agent_source = _fetch_url_text(token, raw_url)
        violations.extend(_agent_source_violations(agent_source))
    else:
        violations.append("Unable to fetch agent.py for static contract checks.")
    return _dedupe(violations)


def _agent_patch_violations(patch: str) -> list[str]:
    violations: list[str] = []
    current_hunk = ""
    for raw_line in patch.splitlines():
        if raw_line.startswith("@@"):
            current_hunk = raw_line
            continue
        if not raw_line.startswith(("+", "-")) or raw_line.startswith(("+++", "---")):
            continue

        text = raw_line[1:].strip()
        if not text:
            continue
        if any(symbol in current_hunk for symbol in PROTECTED_HUNK_SYMBOLS):
            violations.append(f"agent.py must not edit validator-owned function near `{current_hunk}`.")
        if any(marker in text for marker in PROTECTED_EDIT_MARKERS):
            violations.append(f"agent.py must not edit validator-owned contract line `{text[:100]}`.")

        if not raw_line.startswith("+"):
            continue

        lowered = text.lower()
        for sampling_name in FORBIDDEN_SAMPLING_NAMES:
            if sampling_name in lowered:
                violations.append(f"agent.py must not add miner-controlled sampling parameter `{sampling_name}`.")

        for forbidden in FORBIDDEN_ADDED_SUBSTRINGS:
            if forbidden in lowered:
                violations.append(f"agent.py adds forbidden secret/provider reference `{forbidden}`.")

        if "os.environ" in text or "getenv(" in text:
            env_names = set(re.findall(r"""["']([A-Z][A-Z0-9_]{2,})["']""", text))
            disallowed = sorted(name for name in env_names if name not in ALLOWED_ENV_NAMES)
            if disallowed:
                violations.append(
                    "agent.py reads non-allowlisted environment variable(s): "
                    + ", ".join(disallowed[:8])
                )
    return violations


def _agent_source_violations(source: str) -> list[str]:
    try:
        tree = __import__("ast").parse(source, filename="agent.py")
    except SyntaxError as exc:
        return [f"agent.py must remain valid Python: {exc.msg} at line {exc.lineno}."]

    violations: list[str] = []
    solve = next(
        (node for node in tree.body if node.__class__.__name__ == "FunctionDef" and node.name == "solve"),
        None,
    )
    if solve is None:
        violations.append("agent.py must define solve(...).")
    else:
        args = [arg.arg for arg in [*solve.args.posonlyargs, *solve.args.args]]
        if tuple(args[: len(REQUIRED_SOLVE_ARGS)]) != REQUIRED_SOLVE_ARGS:
            violations.append(
                "solve() must keep leading arguments: " + ", ".join(REQUIRED_SOLVE_ARGS) + "."
            )
        sampling_args = sorted(name for name in args if name in FORBIDDEN_SAMPLING_NAMES)
        if sampling_args:
            violations.append("solve() must not expose sampling parameter(s): " + ", ".join(sampling_args) + ".")

    stdlib = set(getattr(sys, "stdlib_module_names", ()))
    stdlib.update({"__future__"})
    for node in __import__("ast").walk(tree):
        if node.__class__.__name__ == "FunctionDef":
            args = [arg.arg for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]]
            sampling_args = sorted(name for name in args if name in FORBIDDEN_SAMPLING_NAMES)
            if sampling_args:
                violations.append(
                    f"{node.name}() must not expose sampling parameter(s): "
                    + ", ".join(sampling_args)
                    + "."
                )
        if node.__class__.__name__ == "Dict":
            for key in node.keys:
                if getattr(key, "value", None) in FORBIDDEN_SAMPLING_NAMES:
                    violations.append(
                        f"agent.py must not set sampling request field `{key.value}`; validator proxy owns sampling."
                    )
        roots: list[str] = []
        if node.__class__.__name__ == "Import":
            roots = [str(alias.name).split(".", 1)[0] for alias in node.names]
        elif node.__class__.__name__ == "ImportFrom":
            roots = [str(node.module or "").split(".", 1)[0]]
        for root in roots:
            if root and root not in stdlib:
                violations.append(f"agent.py imports non-stdlib module `{root}`.")

    return violations


def _fetch_pr_files(token: str, repo: str, pr_number: int) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    page = 1
    while True:
        batch = _github_json(token, repo, f"/pulls/{pr_number}/files?per_page=100&page={page}")
        if not batch:
            return files
        files.extend(batch)
        if len(batch) < 100:
            return files
        page += 1


def _github_json(token: str, repo: str, path: str, method: str = "GET", payload: Any | None = None) -> Any:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    data = _github_request(token, f"/repos/{repo}{path}", method, body)
    return json.loads(data.decode("utf-8"))


def _github_request(token: str, path: str, method: str, body: bytes | None) -> bytes:
    req = urllib.request.Request(
        url=f"{GITHUB_API}{path}",
        data=body,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "ninja-pr-scope-guard",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub API {method} {path} failed with HTTP {exc.code}: {error_body}") from exc


def _fetch_url_text(token: str, url: str) -> str:
    req = urllib.request.Request(
        url=url,
        headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": "ninja-pr-scope-guard",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Fetch {url} failed with HTTP {exc.code}: {error_body}") from exc


def _render_comment(
    verdict: str,
    author: str,
    changed_files: list[str],
    allowed_files: set[str],
    scope_violations: list[str],
    contract_violations: list[str],
) -> str:
    title_verdict = verdict.upper()
    lines = [
        MARKER,
        "## Ninja PR Scope Guard",
        "",
        f"Verdict: **{title_verdict}**",
        f"Author: `{author}`",
        "External contributor file allowlist: "
        + ", ".join(f"`{filename}`" for filename in sorted(allowed_files)),
        "",
    ]
    if scope_violations or contract_violations:
        lines.append("External PRs may only change allowed miner-owned parts of agent.py.")
        if scope_violations:
            lines.extend(["", "### Files Outside Scope"])
            lines.extend(f"- `{filename}`" for filename in scope_violations)
        if contract_violations:
            lines.extend(["", "### Agent.py Contract Violations"])
            lines.extend(f"- {reason}" for reason in contract_violations)
    else:
        lines.append("This PR satisfies the external contributor file-scope and agent.py contract rules.")

    lines.extend(["", "### Changed Files"])
    lines.extend(f"- `{filename}`" for filename in changed_files or ["No files returned by GitHub."])
    return "\n".join(lines) + "\n"


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _upsert_comment(token: str, repo: str, pr_number: int, body: str) -> None:
    if _update_existing_comment(token, repo, pr_number, body):
        return
    _github_json(token, repo, f"/issues/{pr_number}/comments", method="POST", payload={"body": body})


def _update_existing_comment(token: str, repo: str, pr_number: int, body: str) -> bool:
    comments = _github_json(token, repo, f"/issues/{pr_number}/comments?per_page=100")
    for comment in comments:
        if MARKER in str(comment.get("body", "")):
            _github_json(token, repo, f"/issues/comments/{comment['id']}", method="PATCH", payload={"body": body})
            return True
    return False


def _write_step_summary(body: str) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        Path(path).write_text(body, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
