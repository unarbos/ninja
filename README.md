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
    model="validator-managed-model",
    api_base="http://validator-proxy/v1",
    api_key="per-run-proxy-token",
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
external Python dependencies. It uses the validator-provided OpenAI-compatible
`/v1/chat/completions` endpoint and a bash action loop.

Miners should not add their own OpenRouter/OpenAI keys or hardcode a model.
The validator passes a managed model id, proxy URL, and per-run proxy token into
`solve(...)`. That proxy is where request limits, token limits, costs, and model
routing are enforced. In production, every miner agent should hit this same
inference surface rather than choosing its own provider or model.

## Editing

Work directly in `agent.py`. The validator owns the task repo and sandbox, so
changes here should focus on how the agent reads, reasons, edits, and returns a
diff.

Useful local environment variables:

```bash
AGENT_MAX_STEPS=40
AGENT_COMMAND_TIMEOUT=30
AGENT_MODEL=validator-managed-model
OPENAI_BASE_URL=http://validator-proxy/v1
OPENAI_API_KEY=per-run-proxy-token
AGENT_TEMPERATURE=0
AGENT_MAX_TOKENS=2048
```

## What Belongs Here

- `agent.py`
- Documentation for miners
- Small local test fixtures, if needed later

Validator services, PM2 config, task pools, R2 tooling, chain wallets, and
generated workspaces should live outside this repo.

## PR Judge CI

Pull requests are judged by `.github/workflows/openrouter-pr-judge.yml`.

The workflow uses `pull_request_target`, checks out only trusted code from the
base branch, fetches the PR diff through the GitHub API, and sends that diff to
an OpenRouter-routed model. It does not run miner-submitted code.

Required GitHub secret:

```text
DOPPLER_TOKEN
```

Required Doppler variables:

```text
OPENROUTER_API_KEY
```

Optional Doppler variables:

```text
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_SITE_URL=https://github.com/unarbos/ninja
OPENROUTER_APP_NAME=ninja-pr-judge
OPENROUTER_MAX_TOKENS=1800
JUDGE_MIN_SCORE=70
JUDGE_MAX_PATCH_CHARS=120000
```
