# ninja

`ninja` is the miner-facing starter harness for Subnet 66. The harness is a
multi-file Python bundle: `agent.py` is the entrypoint and the `agent/`
package holds the rest of the agent (stdlib-only). Miners should edit the agent code and keep validator systems,
task generators, scoring, wallets, and infrastructure out of this repo.

Production submissions are private. Send your harness (every `*.py` file, up
to 32) to the Subnet 66 submission API with a signature from your registered
miner hotkey. Once the API accepts it, the validator can queue it directly
from the private submission ledger. Single-file submissions of just `agent.py`
remain fully supported.

For the miner-facing submission guide, see
[`MINER_SUBMISSION_CHECKLIST.md`](./MINER_SUBMISSION_CHECKLIST.md).

## What You Are Allowed To Edit

- `agent.py` (the entrypoint; keep the validator-owned contract lines intact)
- `agent/` modules, or your own `*.py` modules (relative imports between
  your files are allowed)
- `tau_agent_files.json` — the manifest listing every file of your bundle
  (a JSON array of relative paths including `agent.py`)

Do not add production mining changes outside the agent bundle. Docs and helper
scripts may be updated by maintainers, but the submitted miner code is the set
of `*.py` files sent to the private submission API.

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

The starter implementation has no external Python dependencies. `agent.py`
keeps the entry-point contract and delegates to the `agent/` package, which
implements a bash action loop against the validator-provided OpenAI-compatible
`/v1/chat/completions` endpoint. The per-round time budget is exported into
the container as `TAU_AGENT_TIMEOUT_SECONDS`, so the agent can pace itself.

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

Work in `agent.py` and the `agent/` modules (or replace `agent/` with
your own modules — keep `tau_agent_files.json` in sync). The validator owns
the task repo and sandbox, so changes should focus on how the agent reasons,
gathers context, edits files, and returns a diff.

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

1. Edit the agent bundle (`agent.py` + your modules).
2. Make sure the hotkey you will sign with is currently registered on Subnet 66.
3. Submit it to the private API with your registered miner hotkey wallet. By
   default the helper bundles this repository (honoring
   `tau_agent_files.json`); pass `--bundle <dir>` for another directory or
   `--agent <file>` for a legacy single-file submission:

```bash
./scripts/submit_private_submission.py \
  --wallet-name <wallet-name> \
  --wallet-hotkey <wallet-hotkey-name> \
  --hotkey <miner-hotkey-ss58>
```

The script signs this payload with your hotkey:

```text
tau-private-submission-v1:<hotkey>:<submission-id>:<bundle-sha256>
```

`<bundle-sha256>` is the sha256 of `agent.py` for single-file submissions
(unchanged from before), and a deterministic hash over every file's path and
content for multi-file bundles. The helper prints it before sending.

You can also attach a display username for private submissions:

```bash
./scripts/submit_private_submission.py \
  --wallet-name <wallet-name> \
  --wallet-hotkey <wallet-hotkey-name> \
  --hotkey <miner-hotkey-ss58> \
  --agent-username <display-name>
```

When `--agent-username` is provided, the helper signs this username proof with
the loaded wallet coldkey and includes the owning coldkey address:

```text
tau-agent-submission-username:<display-name>
```

If your coldkey is not available to the helper, pass `--coldkey` and
`--coldkey-signature` manually. The validator stores and publishes the username
only when that coldkey currently owns the submitting hotkey and the signature
verifies. Invalid or incomplete username proofs are ignored; they do not block
an otherwise valid private submission.

Usernames are display labels, not unique account ids. Multiple hotkeys can use
the same coldkey and can submit different usernames such as `username`,
`username2`, and `username3`. Reusing a username does not spend it or reserve it
globally.

The API returns JSON. If checks fail, `accepted` is `false`, the response
includes `ci_checks`/`llm_judge` details, and the script exits nonzero. If
accepted, no pull request or on-chain commitment is required. The response
includes the private submission commitment id the validator tracks internally:

```text
private-submission:<submission-id>:<bundle-sha256>
```

Only one accepted submission is eligible per miner hotkey registration. After an
accepted submission, that hotkey is spent for future submissions until it is
freshly registered again. A second valid submission from the same hotkey in the
same registration period is rejected even if it uses a different username,
submission id, or bundle hash. Other registered hotkeys controlled by the
same coldkey can still submit their own bundles.

Accepted public submission metadata is visible at:

```text
https://ninja66.ai/api/submissions
```

The public API does not reveal submitted `agent.py` contents or hotkey
signatures.

## Submission Guardrails

The private submission API runs these gates in order:

- `Signature Gate`
- `Registration Gate`
- `Agent Smoke`
- `Submission Scope Guard`
- `OpenRouter Submission Judge`

`Signature Gate` rejects malformed or invalid hotkey signatures before any
expensive checks run.

`Registration Gate` confirms the signing hotkey is currently registered and has
not already spent its current registration on an accepted private submission.
Username labels do not change this rule; spending is tracked by registered
hotkey and registration block.

`Agent Smoke` compiles every submitted file and checks for obvious static
issues.

`Submission Scope Guard` runs per file. It rejects edits that break the solve
contract, add forbidden provider/sampling/secret usage, or try to bypass the
validator-managed proxy. Imports between the files of your own bundle are
allowed.

`OpenRouter Submission Judge` uses the same gatekeeping judge prompt as the
legacy ninja CI, run through OpenRouter with `anthropic/claude-opus-4.7`,
temperature `0`, and medium reasoning effort. It rejects poor, unsafe,
cosmetic, copied, obfuscated, Goodharting, or out-of-scope edits.

## Scoring Target

Validation tasks are generated from real GitHub commits. Each task starts from
the repository before the mined commit, and the reference patch is used to
construct and filter the task.

For duels, the score comes solely from the LLM diff judge. The validator
still pre-solves each task with a Cursor baseline so it can keep compatibility
telemetry, copy checks, and timeout calibration data, but Cursor-baseline
similarity no longer contributes to the winner.

Round score is based only on the LLM diff judgment. The live diff judge uses
`anthropic/claude-sonnet-4.6` through OpenRouter at temperature 0 with adaptive
reasoning enabled and a 16000-token output cap, then scores the king and
challenger patches against the task/reference context. The validator uses
OpenRouter Anthropic prompt caching with an explicit `cache_control: {"type":
"ephemeral"}` content-block breakpoint after the stable task and reference
patch context, leaving candidate patches uncached so repeated tasks can reuse
cached prompt reads when they meet Sonnet 4.6 cache-size requirements. If Sonnet
returns the same OpenRouter route/provider no-choices error, the judge falls
back to `moonshotai/kimi-k2.6` with a plain non-Anthropic prompt shape.

The challenger needs more decisive round wins than the current king. The
validator may require an extra win margin in production.

Cursor is telemetry only for round scoring. The challenger does not need to beat
Cursor directly; it only needs more decisive round wins than the current king
plus the configured margin.

The validator still compares king and challenger patches for copy detection, but
that pairwise similarity does not affect the round score.

When a private challenger becomes king, the validator assigns validator
weights to the winning hotkey on the next allowed weight-set epoch. Single-file
kings are published into the public base harness; multi-file kings keep
running from their private bundle (base-repo publication stays single-file).
