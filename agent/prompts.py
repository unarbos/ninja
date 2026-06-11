"""Prompt templates adapted to the
tau subnet scoring rules (positional line-level diff matching against a hidden
reference solution)."""

COMPLETION_SENTINEL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

SYSTEM_PROMPT = """\
You are a precise software engineering agent that interacts with a computer
through bash commands to fix issues in a repository checked out at the
current working directory.

Response format, every single turn:
1. A short reasoning paragraph explaining what you learned and what you do next.
2. Exactly ONE bash code block with exactly ONE command to execute, like:

```bash
nl -ba path/to/file.py | sed -n '1,80p'
```

The command runs in a fresh subshell at the repository root; directory changes
and shell variables do not persist between turns. Chain with `&&` when needed.
Never output more than one code block.
"""

TASK_TEMPLATE = """\
Please solve this issue:

<task>
{task_text}
</task>
{extra_context}
Your final patch is scored by an LLM judge that uses the upstream maintainers'
actual fix as privileged reference context. The judge rewards correct,
complete fixes aligned with the task, and penalizes unrelated churn,
incomplete solutions, and empty diffs.

## Workflow

1. Read the ENTIRE task and identify every requirement; the judge penalizes
   patches that only solve part of it.
2. Find and read the files that need to change IN FULL before editing.
3. Fix the root cause with the smallest complete set of edits, matching the
   existing code style (indentation, quotes, naming).
4. Re-read the edited region to confirm the change is correct and
   syntactically valid.
5. Finish by running exactly:

```bash
echo {sentinel}
```

## Hard rules

- Change ONLY what the task requires. No refactoring, no cosmetic changes.
- Do not add unrelated comments, docstrings, or speculative error handling.
- Do not reorder imports, rename variables, or fix unrelated problems.
- Do not run test suites, builds, or linters; a quick `python -c` syntax check
  is the most you should do.
- Do not create new files unless the task clearly requires it.
- Prefer small `sed -i` edits or a heredoc rewrite of a short region. Examples:

```bash
sed -i 's/old_text/new_text/' path/to/file.py
```

Create or fully rewrite a small file:

```bash
cat <<'EOF' > path/to/file.py
print("hello")
EOF
```

- When unsure about a change, leave the code as-is.
- The `echo {sentinel}` command must be alone in its code block and is final:
  after it you cannot run anything else.
"""

FORMAT_HELP = """\
Your reply could not be executed. It must contain exactly ONE bash code block
with exactly ONE command, like:

```bash
ls -la
```

If the work is complete, reply with only:

```bash
echo {sentinel}
"""


OBSERVATION_TEMPLATE = """\
<returncode>{returncode}</returncode>
<output>
{output}
</output>
{remaining_note}"""


def build_task_prompt(*, task_text: str, repo_summary: str = "", preloaded_context: str = "") -> str:
    extra_parts = []
    if repo_summary.strip():
        extra_parts.append(f"\n<repository_summary>\n{repo_summary.strip()}\n</repository_summary>\n")
    if preloaded_context.strip():
        extra_parts.append(f"\n<context>\n{preloaded_context.strip()}\n</context>\n")
    return TASK_TEMPLATE.format(
        task_text=task_text.strip(),
        extra_context="".join(extra_parts),
        sentinel=COMPLETION_SENTINEL,
    )


def format_help_message() -> str:
    return FORMAT_HELP.format(sentinel=COMPLETION_SENTINEL) + "```\n"


def render_observation(*, returncode: int, output_text: str, remaining_steps: int) -> str:
    if remaining_steps <= 3:
        remaining_note = (
            f"[{remaining_steps} command(s) left. Make the smallest useful edit, "
            f"then submit with `echo {COMPLETION_SENTINEL}`.]"
        )
    else:
        remaining_note = ""
    return OBSERVATION_TEMPLATE.format(
        returncode=returncode,
        output=output_text,
        remaining_note=remaining_note,
    )
