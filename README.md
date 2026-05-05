# ninja

`ninja` is the miner-facing starter harness for Subnet 66. Miners should edit
`agent.py` and keep validator systems, task generators, scoring, wallets, and
infrastructure out of this repo.

Miner submissions happen through pull requests to this repo. The validator only
runs PRs that are also committed on-chain by a registered miner hotkey.

## What You Are Allowed To Edit

- `agent.py`
- README/docs to clarify agent behavior for miners

Nothing else should be added here for production mining, including
( but not limited to ):

- validator service code
- PM2 configs or service orchestration
- task pool or dataset tooling
- R2 tooling
- chain wallets
- benchmark/workspace generators or generated artifacts

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

The starter implementation is intentionally one file with no external Python
dependencies. It uses the validator-provided OpenAI-compatible
`/v1/chat/completions` endpoint and a bash action loop.

Miners should not add their own OpenRouter/OpenAI keys or hardcode a model. The
validator passes a managed model id, proxy URL, and per-run proxy token into
`solve(...)`. That proxy enforces request limits, token limits, cost caps, and
model routing. In production, every miner agent should use this one inference
surface.

Sampling is also validator-owned. Do not add or tune request fields such as
`temperature`, `top_p`, `top_k`, `seed`, penalties, `logit_bias`, or
`logprobs`. The validator proxy enforces the managed policy (for example
`temperature=0.0`, `top_p=1.0`) and strips miner-controlled fields before
forwarding.

## Editing

Work directly in `agent.py`. The validator owns the task repo and sandbox, so
changes should focus on how the agent reasons, gathers context, edits files, and
returns a diff.

Useful local environment variables for sandbox runs:

```bash
AGENT_MAX_STEPS=40
AGENT_COMMAND_TIMEOUT=30
AGENT_MODEL=validator-managed-model
OPENAI_BASE_URL=http://validator-proxy/v1
OPENAI_API_KEY=per-run-proxy-token
AGENT_MAX_TOKENS=2048
```

Keep these boundaries intact:

- preserve the `solve(repo_path, issue, model, api_base, api_key)` entry point
- return `patch`, `logs`, `steps`, `cost`, and `success`
- use only the supplied `api_base` and `api_key`
- do not hardcode another model, provider endpoint, API key, wallet, scorer, or
  validator secret
- do not add third-party Python dependencies
- do not read or exfiltrate host secrets, hidden tests, prompts, or evaluator
  data

### PR-blocked edits

PRs are blocked (or fail CI) if they:

- modify files outside `agent.py`
- change the `solve(...)` entry-point contract
- add forbidden provider/secret references
- attempt to hardcode or route around the managed model/proxy (`api_base`,
  `api_key`, `model`)
- add sampling/decoding control (`temperature`, `top_p`, `top_k`, `seed`, penalties,
  `logit_bias`, `logprobs`, etc.)
- include non-miner infra files or workflow/config changes
- touch validation, scoring, or repo-control paths not related to agent
  behavior

## Miner PR Flow

1. Fork or branch from `unarbos/ninja`.
2. Edit `agent.py`.
3. Open a PR back to `unarbos/ninja:main`.
4. Make the PR title start with your exact miner hotkey:

```text
<miner-hotkey> improve command loop
```

Do not use any prefix like `hkey:` before the hotkey. The hotkey must be the
first characters in the title.

5. Commit the PR head on-chain using:

```text
github-pr:unarbos/ninja#<pr-number>@<head-sha>
```

This repo includes a helper for submitting that exact string with Bittensor.
It defaults to subnet 66:

```bash
./scripts/commit_on_chain.py \
  --wallet-name <wallet-name> \
  --wallet-hotkey <wallet-hotkey-name> \
  --hotkey <miner-hotkey-ss58> \
  --netuid 66 \
  --commit "github-pr:unarbos/ninja#<pr-number>@<head-sha>"
```

`--hotkey` is checked against the loaded wallet hotkey so the PR title, wallet,
and on-chain commitment all refer to the same miner.

Only one commitment per miner hotkey is eligible in each 24h window. The
validator enforces the window as 7,200 chain blocks since the last accepted
commitment for that hotkey. A newer commitment made before that window expires
is skipped.

The validator binds the PR to the committing hotkey by checking that the PR title
starts with the same hotkey that made the on-chain commitment. It also checks that
the committed SHA matches the current PR head SHA.

### PR Guardrails

The validator requires two checks before queuing a PR:

- `PR Scope Guard`
- `OpenRouter PR Judge`

`PR Scope Guard` rejects edits that break these boundaries (file scope,
contract scope, forbidden provider/sampling/secret usage).

`OpenRouter PR Judge` sends only the PR diff to an LLM judge and rejects poor or
unsafe edits. The judge is configured to score edits against the reference
without running miner code.

## Scoring Target

Validation tasks are generated from real GitHub commits, but miner duel scores are
compared against the validator's Cursor baseline solution for each task. The
reference patch is still used to construct and filter tasks.

Round score is half Cursor-baseline similarity and half LLM diff judgment of king
and challenger patches.

Cursor is only the measuring stick. The challenger does not need to beat Cursor
directly; it only needs more decisive round wins than the current king.

The validator separately compares king and challenger patches for copy detection.

When a PR challenger becomes king, the validator merges that PR into
`unarbos/ninja:main`; future miners branch from that new base harness. Validator
weights are assigned to the winning hotkey on the next allowed weight-set epoch.


!!
