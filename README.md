# ninja

`ninja` is the miner-facing starter harness for Subnet 66. Miners should edit
`agent.py` and keep the validator, task generation, scoring, and hidden tests
outside this repo.

Miner submissions happen through pull requests to this repo. The validator only
runs PRs that are also committed on-chain by a registered miner hotkey.

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

Sampling is also validator-owned. Do not add or tune request fields such as
`temperature`, `top_p`, `top_k`, `seed`, penalties, `logit_bias`, or `logprobs`.
The validator proxy overwrites `temperature=0.0`, overwrites `top_p=1.0`, and
strips other miner-controlled sampling fields before forwarding to OpenRouter.

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
AGENT_MAX_TOKENS=2048
```

External miner PRs are expected to touch `agent.py` only. Good areas to edit
include prompting, context gathering, command selection, result parsing,
stopping logic, patch generation, safety checks, and how the harness uses its
step budget.

Keep these boundaries intact:

- preserve the `solve(repo_path, issue, model, api_base, api_key)` entry point
- return `patch`, `logs`, `steps`, `cost`, and `success`
- use only the supplied `api_base` and `api_key`
- do not hardcode another model, provider endpoint, API key, wallet, scorer, or
  validator secret
- do not add third-party Python dependencies
- do not read or exfiltrate host secrets, hidden tests, prompts, or evaluator
  data

## Miner PR Flow

1. Fork or branch from `unarbos/ninja`.
2. Edit `agent.py`.
3. Open a PR back to `unarbos/ninja:main`.
4. Make the PR title start with your exact miner hotkey:

```text
<miner-hotkey> improve command loop
```

Do not use a label such as `hkey:` before the hotkey. The hotkey must be the
first characters in the title.

5. Commit the PR head on-chain using:

```text
github-pr:unarbos/ninja#<pr-number>@<head-sha>
```

Only one commitment per miner hotkey is eligible in each 24h window. The
validator measures the window as 7,200 chain blocks from the last accepted
commitment for that hotkey. A newer commitment made before that window expires
is skipped.

The validator binds the PR to the committing hotkey by checking that the PR
title starts with the same hotkey that made the on-chain commitment. It also
checks that the committed SHA matches the current PR head SHA.

## Scoring Target

Validation tasks are generated from real GitHub commits, but miner duel scores
are compared against the validator's Cursor baseline solution for each task.
The mined GitHub reference patch is still used to construct and filter tasks;
the round score is 1/2 Cursor-baseline similarity plus 1/2 LLM diff judgment
of the king and challenger patches.

Cursor is only the measuring stick. The challenger does not need to beat Cursor
directly; it only needs more decisive round wins than the current king.

The validator separately compares king and challenger patches for copy
detection.

When a PR challenger becomes king, the validator merges that PR into
`unarbos/ninja:main`; future miners branch from that new base harness. Validator
weights are assigned to the winning hotkey on the next allowed weight-set epoch.

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

The CI checks are:

- `PR Scope Guard`: rejects external PRs that touch files outside `agent.py`,
  break the `solve(...)` contract, add forbidden provider/secret references, or
  try to control sampling.
- `OpenRouter PR Judge`: uses `deepseek/deepseek-v4-flash` through OpenRouter to judge
  whether the PR is a real, scoped, safe miner edit.

Both checks require the PR title to start with a hotkey-shaped prefix. If the
title gate fails, the OpenRouter judge exits before making an LLM request.

Required CI secret material:

```text
OPENROUTER_API_KEY
```

Optional CI environment variables:

```text
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_SITE_URL=https://github.com/unarbos/ninja
OPENROUTER_APP_NAME=ninja-pr-judge
OPENROUTER_MAX_TOKENS=16000
JUDGE_MIN_SCORE=70
JUDGE_MAX_PATCH_CHARS=120000
```

The PR judge uses medium OpenRouter reasoning effort and excludes reasoning text
from the returned message, so the model can reason internally while still
returning the required JSON verdict.
