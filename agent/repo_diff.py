"""Collect the repository patch the same way the validator harness does:
tracked changes via `git diff --binary` plus untracked files via no-index
diffs against /dev/null.

Before collection we remove agent-created scratch artifacts (helper munge
scripts, .bak/.orig/.tmp/.rej leftovers) that the diff judge penalizes as
unnecessary churn. Only UNTRACKED files the agent itself created are eligible,
and only when no kept change references them, so a legitimately-added source or
test module is never touched. Pure stdlib, fail-open: any error skips scrubbing
and collection proceeds unchanged."""

import os
import re
import subprocess

# Verb-prefixed munge scripts and editor/patch leftovers the model writes for
# its own bookkeeping. NOTE: no "test_fix" / "test" verb here -- the prompt now
# encourages a real regression test, so test files must never be scrubbed.
_SCRATCH_NAME_RE = re.compile(
    r"^(?:"
    r"(?:fix|clean|cleanup|mock|update|patch|apply|munge|tmp|temp|scratch|"
    r"run|do|gen|generate|rewrite|migrate)_[\w.-]*\.py"
    r"|[\w.-]+\.(?:bak|orig|tmp|rej|swp|swo)"
    r"|[\w.-]+~"
    r")$",
    re.IGNORECASE,
)


def collect_repo_patch(repo_dir: str) -> str:
    untracked = _untracked_files(repo_dir)
    _scrub_scratch(repo_dir, untracked)
    diff = _run_git(["diff", "--binary", "--", "."], repo_dir)
    listing = _run_git(["ls-files", "--others", "--exclude-standard", "-z"], repo_dir)
    for relative_path in [item for item in listing.split("\0") if item]:
        file_diff = _run_git_diff_no_index(relative_path, repo_dir)
        diff += file_diff
    return diff


def _untracked_files(repo_dir: str) -> list:
    listing = _run_git(["ls-files", "--others", "--exclude-standard", "-z"], repo_dir)
    return [item for item in listing.split("\0") if item]


def _scrub_scratch(repo_dir: str, untracked: list) -> None:
    """Delete agent-created scratch artifacts not referenced by a kept change.
    Fail-open: never raises."""
    try:
        if not untracked:
            return
        candidates = [
            p for p in untracked
            if "/" not in p.rstrip("/") and _SCRATCH_NAME_RE.match(os.path.basename(p))
        ]
        if not candidates:
            return
        kept_diff = _run_git(["diff", "--", "."], repo_dir) or ""
        keep_blob = kept_diff + "\n" + "\n".join(p for p in untracked if p not in candidates)
        for rel in candidates:
            stem = os.path.splitext(os.path.basename(rel))[0]
            if stem and (stem in keep_blob or os.path.basename(rel) in keep_blob):
                continue
            abs_path = os.path.join(repo_dir, rel)
            try:
                if os.path.isfile(abs_path):
                    os.remove(abs_path)
            except OSError:
                continue
    except Exception:
        return


def _run_git(args: list, repo_dir: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_dir,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return completed.stdout or ""


def _run_git_diff_no_index(relative_path: str, repo_dir: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "diff", "--binary", "--no-index", "--", "/dev/null", relative_path],
            cwd=repo_dir,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if completed.returncode in (0, 1):
        return completed.stdout or ""
    return ""
