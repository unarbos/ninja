Subnet 66 Ninja Miner Quickstart
================================

This repo is the public miner harness for Subnet 66. For production mining,
edit only agent.py. The validator imports agent.py, calls solve(...), and
compares your agent against the current king on hidden tasks.

Before You Submit
-----------------

1. Edit agent.py only.
2. Keep the solve(repo_path, issue, model, api_base, api_key) contract.
3. Use only the model, api_base, and api_key passed by the validator.
4. Do not add provider keys, wallets, validator tooling, CI files, or
   third-party dependencies.
5. Make sure your Bittensor wallet hotkey is the registered miner hotkey you
   want credited.

Submit Privately
----------------

Run:

./scripts/submit_private_submission.py \
  --wallet-name <wallet-name> \
  --wallet-hotkey <wallet-hotkey-name> \
  --hotkey <miner-hotkey-ss58>

The script:

- reads agent.py
- derives a submission id and sha256
- signs tau-private-submission-v1:<hotkey>:<submission-id>:<sha256>
- uploads agent.py to https://ninja66.ai/api/submissions
- prints the API response

If checks fail, accepted is false and the response includes ci_checks and
llm_judge details. Fix agent.py and submit again after reviewing those findings.

After Acceptance
----------------

If the API accepts your submission, it returns:

private-submission:<submission-id>:<sha256-of-agent.py>

No separate on-chain commitment is required. The validator reads accepted API
submissions from the private ledger, verifies your hotkey is still registered,
and queues the bundle directly.

Important Details
-----------------

- The hotkey that signs the API payload is the hotkey credited by the validator.
- Only one accepted submission is eligible per miner hotkey registration.
- A second accepted submission from the same hotkey is allowed only after the
  hotkey is freshly registered again.
- Public accepted-submission metadata is available at:

https://ninja66.ai/api/submissions

- The public metadata does not reveal submitted agent.py contents or signatures.

Submission Checks
-----------------

The private API runs:

- Agent Smoke
- Submission Scope Guard
- OpenRouter Submission Judge
- Registration Gate

OpenRouter Submission Judge uses anthropic/claude-opus-4.7 with the same
gatekeeping prompt as ninja CI, at temperature 0.

The most common rejection reasons are changing solve(...), adding provider keys
or endpoints, adding sampling controls, importing third-party packages, or
trying to route around the validator-managed model/proxy.
