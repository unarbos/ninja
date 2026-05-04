# ninja

`ninja` is the miner-facing starter harness for Subnet 66. Miners should edit
`agent.py` and keep the validator, task generation, scoring, and hidden tests
outside this repo.

## Contract

The validator imports `agent.py` and calls:

```python
solve(
    repo_path="/tmp/task_repo",
    issue="Fix the bug...",
    model="...",
    api_base="https://...",
    api_key="...",
)
```

`solve(...)` must return:

```python
{
    "patch": "... unified git diff ...",
    "logs": "...",
    "steps": 0,
    "cost": None,
    "success": True,
}
```

The starter implementation is intentionally a single Python file with no
external Python dependencies. It uses an OpenAI-compatible
`/v1/chat/completions` endpoint and a bash action loop.

## Editing

Work directly in `agent.py`. The validator owns the task repo and sandbox, so
changes here should focus on how the agent reads, reasons, edits, and returns a
diff.

Useful local environment variables:

```bash
AGENT_MAX_STEPS=40
AGENT_COMMAND_TIMEOUT=30
AGENT_MODEL=gpt-4o-mini
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=...
AGENT_TEMPERATURE=0
AGENT_MAX_TOKENS=2048
```

## What Belongs Here

- `agent.py`
- Documentation for miners
- Small local test fixtures, if needed later

Validator services, PM2 config, task pools, R2 tooling, chain wallets, and
generated workspaces should live outside this repo.
