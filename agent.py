#!/usr/bin/env python3
"""
Multi-file SWE coding agent for the tau subnet.

Contract (unchanged from the public single-file base agent):
    The validator imports this file and calls:

        solve(
            repo_path="/tmp/task_repo",
            issue="Fix the bug...",
            model="validator-managed-model",
            api_base="http://validator-proxy/v1",
            api_key="per-run-proxy-token"
        )

    It returns a dict with patch, logs, steps, cost, and success.

Layout:
    agent.py             validator-owned contract + thin solve() wiring
    agent/prompts.py     system/instance templates for complete, verified fixes
    agent/model.py       stdlib OpenAI-compatible chat client with retries
    agent/environment.py fresh-subshell bash executor
    agent/agent_loop.py  the query -> act -> observe step loop
    agent/repo_diff.py   harness-compatible patch collection

All inference uses only the validator-provided api_base/api_key; there are no
third-party dependencies and no sampling overrides (the validator proxy owns
sampling).
"""

from __future__ import annotations

import os
import subprocess
import time
import traceback
from typing import Any, Dict, Optional, Tuple

from agent.agent_loop import AgentRunConfig, run_agent_loop
from agent.prompts import build_task_prompt
from agent.repo_diff import collect_repo_patch

# -----------------------------
# Config
# -----------------------------

DEFAULT_MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", "50"))
# Allow a single command enough time to run a small reproduction or assertion
# that demonstrates the fix is correct. Still far under the per-round wall
# budget so the loop finishes and reports its own patch.
DEFAULT_COMMAND_TIMEOUT = int(os.environ.get("AGENT_COMMAND_TIMEOUT", "40"))

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

MAX_OBSERVATION_CHARS = int(os.environ.get("AGENT_MAX_OBSERVATION_CHARS", "16000"))
MAX_TOTAL_LOG_CHARS = int(os.environ.get("AGENT_MAX_TOTAL_LOG_CHARS", "260000"))

# Stay under the validator's per-round budget so the loop can finish gracefully
# and report its own patch instead of relying on the kill path. The validator
# now exports its real per-round budget as TAU_AGENT_TIMEOUT_SECONDS; honor it
# (leaving a margin for diff collection) so a looser budget actually lets the
# agent keep working. Falls back to the conservative 280s when unset.
def _wall_clock_limit_seconds() -> float:
    budget = os.environ.get("TAU_AGENT_TIMEOUT_SECONDS")
    if budget:
        try:
            return max(60.0, float(int(budget)) - 20.0)
        except ValueError:
            pass
    return 280.0


WALL_CLOCK_LIMIT_SECONDS = _wall_clock_limit_seconds()

# Headroom kept before the wall limit so a repair pass leaves time for the
# final diff collection instead of being killed mid-write.
WALL_CLOCK_RESERVE_SECONDS = 10.0


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


def build_initial_user_prompt(issue: str, repo_summary: str, preloaded_context: str = "") -> str:
    return build_task_prompt(task_text=issue, repo_summary=repo_summary, preloaded_context=preloaded_context)


# Minimum wall-clock headroom (seconds) needed to attempt a repair pass; below
# this we keep the first patch rather than start work we cannot finish.
VERIFY_REPAIR_MIN_BUDGET_SECONDS = 45.0
VERIFY_REPAIR_MAX_STEPS = 14


def _changed_py_files(patch_text: str) -> list:
    """Python files touched by the patch (parsed from its `+++ b/` headers)."""
    paths = []
    for line in patch_text.splitlines():
        if line.startswith("+++ b/"):
            path = line[len("+++ b/"):].strip()
            if path.endswith(".py") and path not in paths:
                paths.append(path)
    return paths


def _py_syntax_errors(repo_dir: str, patch_text: str) -> list:
    """Changed .py files whose current on-disk content does not parse."""
    broken = []
    for rel in _changed_py_files(patch_text):
        full = os.path.join(repo_dir, rel)
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as handle:
                source = handle.read()
        except OSError:
            continue
        try:
            compile(source, rel, "exec")
        except SyntaxError as exc:
            broken.append(f"{rel}: line {exc.lineno}: {exc.msg}")
        except (ValueError, TypeError):
            broken.append(f"{rel}: could not be parsed")
    return broken


def _repair_reason(repo_dir: str, patch_text: str) -> Optional[str]:
    """Deterministic signal that the emitted patch is empty or broken, else None."""
    if not (patch_text or "").strip():
        return "the current change set is empty; no fix was produced yet"
    broken = _py_syntax_errors(repo_dir, patch_text)
    if broken:
        return "the edited files contain syntax errors that must be fixed:\n- " + "\n- ".join(broken[:8])
    return None


def _build_repair_task(issue_text: str, reason: str) -> str:
    return (
        "A previous attempt to solve the task below left the repository in an "
        "incomplete or broken state. " + reason + "\n\n"
        "Inspect the current state of the repository, then finish and correct "
        "the change so it fully and correctly solves the task. Re-read each "
        "edited region to confirm it is syntactically valid before submitting.\n\n"
        "Original task:\n" + issue_text
    )


# Untracked editor/patch scratch files an agent sometimes leaves behind while
# editing (e.g. a `cli.ts.new` next to `cli.ts`) get folded into the scored
# patch where the judge reads them as broken/messy churn. We delete them before
# the patch is collected. SAFETY: a scratch file is by definition a shadow of a
# real file, so we only delete `X<suffix>` when the sibling `X` actually exists
# -- this catches every true artifact while never touching a legitimately-named
# untracked deliverable (e.g. a real file named config.orig).
_EDIT_ARTIFACT_SUFFIXES = (".new", ".orig", ".bak", ".rej")


def _artifact_sibling(rel: str) -> Optional[str]:
    """The real file a scratch path shadows, or None if it is not an artifact."""
    if rel.endswith("~"):
        return rel[:-1] or None
    for suffix in _EDIT_ARTIFACT_SUFFIXES:
        if rel.endswith(suffix):
            return rel[: -len(suffix)] or None
    return None


def _strip_edit_artifacts(repo_dir: str) -> int:
    """Remove untracked editor/patch scratch files whose real sibling exists."""
    removed = 0
    try:
        listing = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=repo_dir, capture_output=True, text=True, timeout=30, check=False,
        ).stdout or ""
    except (OSError, subprocess.TimeoutExpired):
        return 0
    for rel in [p for p in listing.split("\0") if p]:
        sibling = _artifact_sibling(rel)
        if sibling and os.path.exists(os.path.join(repo_dir, sibling)):
            try:
                os.remove(os.path.join(repo_dir, rel))
                removed += 1
            except OSError:
                pass
    return removed


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
    started = time.monotonic()
    try:
        model_name, base_url, proxy_token = _resolve_inference_config(model, api_base, api_key)
        run_config = AgentRunConfig(
            repo_dir=repo_path,
            model_name=model_name,
            base_url=base_url,
            auth_token=proxy_token,
            max_steps=max_steps,
            command_timeout=command_timeout,
            max_tokens=max_tokens,
            max_observation_chars=MAX_OBSERVATION_CHARS,
            max_log_chars=MAX_TOTAL_LOG_CHARS,
            wall_clock_limit=WALL_CLOCK_LIMIT_SECONDS,
        )
        outcome = run_agent_loop(
            config=run_config,
            task=build_initial_user_prompt(issue, "", ""),
        )

        # Verification gate: the base agent submits on the first completion
        # signal with no check, so it ships some empty or syntactically broken
        # patches. If the emitted change is empty or leaves an edited Python file
        # unparseable AND wall-clock budget remains, run one bounded repair pass
        # and keep it only when it is strictly better (a
        # non-empty patch with no syntax errors). Never worsen the first result.
        repair_note = ""
        try:
            remaining = WALL_CLOCK_LIMIT_SECONDS - (time.monotonic() - started)
            reason = _repair_reason(repo_path, outcome.patch)
            if reason is not None and remaining >= VERIFY_REPAIR_MIN_BUDGET_SECONDS:
                repair_config = AgentRunConfig(
                    repo_dir=repo_path,
                    model_name=model_name,
                    base_url=base_url,
                    auth_token=proxy_token,
                    max_steps=min(max_steps, VERIFY_REPAIR_MAX_STEPS),
                    command_timeout=command_timeout,
                    max_tokens=max_tokens,
                    max_observation_chars=MAX_OBSERVATION_CHARS,
                    max_log_chars=MAX_TOTAL_LOG_CHARS,
                    wall_clock_limit=remaining - WALL_CLOCK_RESERVE_SECONDS,
                )
                repaired = run_agent_loop(
                    config=repair_config,
                    task=build_initial_user_prompt(_build_repair_task(issue, reason), "", ""),
                )
                if (
                    repaired.patch.strip()
                    and not _py_syntax_errors(repo_path, repaired.patch)
                ):
                    outcome = repaired
                    repair_note = " (repair pass adopted)"
        except Exception:
            repair_note = " (repair pass skipped after error)"

        # Patch hygiene: drop editor/patch scratch files (never a real
        # deliverable) and re-collect, so the judge scores only the real change.
        patch = outcome.patch
        try:
            if _strip_edit_artifacts(repo_path) > 0:
                patch = collect_repo_patch(repo_path)
                repair_note += " (stripped scratch files)"
        except Exception:
            pass

        elapsed = time.monotonic() - started
        return {
            "patch": patch,
            "logs": outcome.logs,
            "steps": outcome.steps,
            "cost": outcome.cost,
            "success": bool(patch.strip()),
            "message": f"{outcome.exit_status}: {outcome.message} in {elapsed:.1f}s{repair_note}",
        }
    except Exception:
        fallback_patch = collect_repo_patch(repo_path)
        return {
            "patch": fallback_patch,
            "logs": traceback.format_exc()[-8000:],
            "steps": 0,
            "cost": None,
            "success": bool(fallback_patch.strip()),
            "message": "agent crashed; returning the on-disk repository diff",
        }
