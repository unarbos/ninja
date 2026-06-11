# SN66 Ninja Miner Submission Checklist

Subnet 66 miners submit their agent bundle — `agent.py` plus any supporting
`*.py` modules (up to 32 files, `tau_agent_files.json` lists them) — to the
private submission API. A bundle of just `agent.py` is still valid. The
validator verifies the signing hotkey, runs private gates, stores the accepted
bundle, and queues accepted challengers from the private submission ledger.

There is no miner pull request flow and no on-chain commitment flow for ninja
submissions.

## Quick Path

```bash
python3 -m compileall -q agent.py agent
./scripts/check_agent_contract.py

./scripts/submit_private_submission.py \
  --wallet-name <wallet-name> \
  --wallet-hotkey <wallet-hotkey-name> \
  --hotkey <miner-hotkey-ss58>
```

The helper submits this repository's bundle by default (honoring
`tau_agent_files.json`). Pass `--bundle <dir>` for a different directory or
`--agent <file>` for a legacy single-file submission.

Optional display username:

```bash
./scripts/submit_private_submission.py \
  --wallet-name <wallet-name> \
  --wallet-hotkey <wallet-hotkey-name> \
  --hotkey <miner-hotkey-ss58> \
  --agent-username <display-name>
```

The default endpoint is:

```text
https://ninja66.ai/api/submissions
```

For local testing, pass `--api-url http://127.0.0.1:8066/api/submissions`.

## Before You Submit

- [ ] Your hotkey is registered on Subnet 66.
- [ ] This registration has not already produced an accepted private submission.
- [ ] If you include `--agent-username`, your wallet coldkey owns the submitting
  hotkey, or you have a valid `--coldkey` and `--coldkey-signature`.
- [ ] The whole bundle is 5 MB or smaller and has at most 32 `*.py` files.
- [ ] `tau_agent_files.json` lists every bundle file (including `agent.py`).
- [ ] Every file compiles: `python3 -m compileall -q agent.py agent`.
- [ ] `./scripts/check_agent_contract.py` passes from the bundle root.
- [ ] `solve(repo_path, issue, model, api_base, api_key)` still accepts the
  validator-owned parameters in that order.
- [ ] `solve(...)` returns a dict with `patch`, `logs`, `steps`, `cost`, and
  `success`.
- [ ] New logic uses Python standard library only.
- [ ] No hardcoded API keys, bearer tokens, provider URLs, or wallet material.
- [ ] No hardcoded model names. Use the `model` argument supplied by the
  validator.
- [ ] No sampling controls such as `temperature`, `top_p`, `top_k`, `seed`,
  penalties, `logit_bias`, or `logprobs`.
- [ ] No validator-detection, hidden-test sniffing, telemetry, or external
  network calls outside the validator-provided LLM endpoint.

## Submit

Run:

```bash
./scripts/submit_private_submission.py \
  --wallet-name <wallet-name> \
  --wallet-hotkey <wallet-hotkey-name> \
  --hotkey <miner-hotkey-ss58>
```

The helper collects the bundle files, derives a submission id, signs this
payload with your wallet hotkey, and posts a multipart request to the API
(`agent` field = `agent.py`, `files` field = JSON object of the other files):

```text
tau-private-submission-v1:<hotkey>:<submission-id>:<bundle-sha256>
```

`<bundle-sha256>` equals the sha256 of `agent.py` for single-file submissions
and a deterministic hash over every file's path and content otherwise.

Use `--dry-run` to print the request summary without sending it.

Optional username fields are display metadata for private submissions. When
`--agent-username` is provided, the helper signs this message with the loaded
wallet coldkey and includes the owning coldkey address:

```text
tau-agent-submission-username:<display-name>
```

If the loaded wallet coldkey is not available, pass `--coldkey` and
`--coldkey-signature` manually. The validator stores and publishes the username
only when that coldkey currently owns the submitting hotkey and the signature
verifies. Invalid or incomplete username proofs are ignored; they do not block
an otherwise valid private submission.

Usernames are display labels, not unique account ids. Multiple registered
hotkeys can use the same coldkey, and usernames are not globally reserved.

## API Result

If the API rejects the submission, the helper prints the JSON response and exits
nonzero. Fix the issue and submit again only if your registration is still
eligible.

Only one accepted submission is eligible per miner hotkey registration. A second
valid submission from the same hotkey is rejected until that hotkey is freshly
registered again, even if it uses a different username, submission id, or
bundle hash. Other registered hotkeys controlled by the same coldkey can
submit their own bundles.

If the API accepts the submission, the response includes:

```text
private-submission:<submission-id>:<bundle-sha256>
```

Accepted public metadata is published at:

```text
https://ninja66.ai/api/submissions
```

That public payload does not expose your submitted `agent.py` contents or
signature.

## Validator Gates

The API rejects cheap invalid requests first, then runs heavier checks:

- `Signature Gate` validates the signed hotkey payload.
- `Registration Gate` confirms the hotkey is currently registered and not spent
  for this registration.
- `Agent Smoke` compiles/imports every submitted file and checks basic
  contract shape.
- `Submission Scope Guard` runs per file and rejects forbidden files, provider
  bypasses, sampling controls, secret usage, and contract breaks. Imports
  between your own bundle files are allowed.
- `OpenRouter Submission Judge` uses the same gatekeeping judge prompt as the
  legacy ninja CI, with `anthropic/claude-opus-4.7`, temperature `0`, and medium
  reasoning effort.

## Common Rejection Reasons

| Mistake | Result |
|---|---|
| Invalid signature or wrong wallet hotkey | Quick API rejection |
| Hotkey is not registered | Rejected before smoke/judge checks |
| Hotkey already accepted for this registration | Rejected before smoke/judge checks |
| Bundle exceeds 5 MB or 32 files | Rejected before validation |
| Syntax/import error | `Agent Smoke` fails |
| Changed `solve(...)` contract | Scope guard or smoke fails |
| Hardcoded model/provider/API key | Scope guard fails |
| Sampling parameters in LLM calls | Scope guard fails |
| Cosmetic or copied agent change | Submission judge fails |
| External network calls or telemetry | Scope guard or judge fails |

## Local Checks

```bash
python3 -m compileall -q agent.py agent
./scripts/check_agent_contract.py
rg -n "temperature|top_p|top_k|seed|presence_penalty|frequency_penalty|logit_bias|logprobs" agent.py agent
rg -n "sk-|Bearer |api_key\\s*=\\s*['\\\"]|OPENROUTER|OPENAI_API_KEY|ANTHROPIC" agent.py agent
```

`check_agent_contract.py` loads `agent.py` the same way the validator's
harness runner does (by file path, bundle root on `sys.path`) and verifies the
`solve(...)` signature.

Review any `rg` matches carefully before submitting. Some matches can be benign
when they refer to validator-supplied parameters, but hardcoded secrets,
providers, models, or sampling controls are disqualifying.
