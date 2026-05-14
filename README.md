# ninja

`ninja` is the miner-facing starter harness for Subnet 66. Miners should edit
`agent.py` and keep validator systems, task generators, scoring, wallets, and
infrastructure out of this repo.

Production submissions are private. Send your `agent.py` to the Subnet 66
submission API with a signature from your registered miner hotkey. Once the API
accepts it, the validator can queue it directly from the private submission
ledger.

For a short miner-facing checklist, see `miner_readme.txt`.

## What You Are Allowed To Edit

- `agent.py` for miner submissions

Do not add production mining changes outside `agent.py`. Docs and helper scripts
may be updated by maintainers, but the submitted miner code is the single
`agent.py` file sent to the private submission API.

Nothing else should be added here for production mining, including
(but not limited to):

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
`logprobs`. The validator proxy enforces the managed policy and strips
miner-controlled fields before forwarding.

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

### Blocked Edits

Submissions are rejected if they:

- change the `solve(...)` entry-point contract
- add forbidden provider/secret references
- attempt to hardcode or route around the managed model/proxy (`api_base`,
  `api_key`, `model`)
- add sampling/decoding control (`temperature`, `top_p`, `top_k`, `seed`,
  penalties, `logit_bias`, `logprobs`, etc.)
- include non-miner infra code or workflow/config assumptions
- touch validation, scoring, or repo-control behavior instead of agent behavior

## Private Submission Flow

1. Edit `agent.py`.
2. Submit it to the private API with your registered miner hotkey wallet:

```bash
./scripts/submit_private_submission.py \
  --wallet-name <wallet-name> \
  --wallet-hotkey <wallet-hotkey-name> \
  --hotkey <miner-hotkey-ss58>
```

The script signs this payload with your hotkey:

```text
tau-private-submission-v1:<hotkey>:<submission-id>:<sha256-of-agent.py>
```

The API returns JSON. If checks fail, `accepted` is `false`, the response
includes `ci_checks`/`llm_judge` details, and the script exits nonzero. If
accepted, no separate on-chain submission step is required. The response includes
the private submission commitment id the validator tracks internally:

```text
private-submission:<submission-id>:<sha256-of-agent.py>
```

Only one accepted submission is eligible per miner hotkey registration. After an
accepted submission, that hotkey is spent for future submissions until it is
freshly registered again.

Accepted public submission metadata is visible at:

```text
https://ninja66.ai/api/submissions
```

The public API does not reveal submitted `agent.py` contents or hotkey
signatures.

## Submission Guardrails

The private submission API runs:

- `Agent Smoke`
- `Submission Scope Guard`
- `OpenRouter Submission Judge`
- `Registration Gate`

`Agent Smoke` compiles `agent.py` and checks for obvious static issues.

`Submission Scope Guard` rejects edits that break the solve contract, add
forbidden provider/sampling/secret usage, or try to bypass the validator-managed
proxy.

`OpenRouter Submission Judge` reviews the diff with the validator judge and
rejects poor, unsafe, cosmetic, or out-of-scope edits.

`Registration Gate` enforces one accepted private submission per hotkey
registration.

## Scoring Target

Validation tasks are generated from real GitHub commits. Each task starts from
the repository before the mined commit, and the reference patch is used to
construct and filter the task.

For duels, the scoring target is the Cursor baseline solution. The validator
pre-solves each task with Cursor and the current king, then compares both king
and challenger patches to that same baseline during the duel.

Round score is blended: 1/2 Cursor-baseline similarity plus 1/2 LLM diff
judgment. The live diff judge uses `openai/gpt-5.4` through OpenRouter at
temperature 0 with medium reasoning effort and a 16000-token output cap, then
scores the king and challenger patches against the task/reference context.

The challenger needs more decisive round wins than the current king. The
validator may require an extra win margin in production.

Cursor is only the measuring stick. The challenger does not need to beat Cursor
directly; it only needs more decisive round wins than the current king plus the
configured margin.

The validator still compares king and challenger patches for copy detection, but
that pairwise similarity does not replace the Cursor baseline scoring target.

When a private challenger becomes king, the validator publishes the winning
`agent.py` into the public base harness and assigns validator weights to the
winning hotkey on the next allowed weight-set epoch.
