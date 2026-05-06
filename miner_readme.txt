Subnet 66 Ninja Miner Quickstart
================================

This repo is the public miner harness for Subnet 66. For production mining,
edit only agent.py. The validator imports agent.py, calls solve(...), and
compares your agent against the current king on hidden tasks.

Before You Start
----------------

1. Fork or branch from unarbos/ninja main.
2. Edit agent.py only.
3. Keep the solve(repo_path, issue, model, api_base, api_key) contract.
4. Use only the model, api_base, and api_key passed by the validator.
5. Do not add provider keys, wallets, validator tooling, CI files, or
   third-party dependencies.

Your PR title must start with your exact registered miner hotkey:

<miner-hotkey> improve command loop

Do not add hkey:, uid:, brackets, or any other prefix before the hotkey.

Option A: Pre-PR Commitment
---------------------------

Use this when you want to commit ownership of your final code before the PR is
public.

1. Commit your final agent.py change locally.
2. Run local preflight:

./scripts/precommit_ninja_pr.py \
  --hotkey <miner-hotkey-ss58> \
  --judge

The script prints a commitment like:

github-pr-head:unarbos/ninja@<head-sha>

With --judge, it calls the same OpenRouter judge prompt used by PR CI. Set
OPENROUTER_API_KEY in your environment first. Without --judge, it still runs
local static CI-style checks.

3. Submit the commitment on-chain before opening the PR:

./scripts/precommit_ninja_pr.py \
  --hotkey <miner-hotkey-ss58> \
  --judge \
  --commit-on-chain \
  --wallet-name <wallet-name> \
  --wallet-hotkey <wallet-hotkey-name>

You can also submit the printed commitment directly:

./scripts/commit_on_chain.py \
  --wallet-name <wallet-name> \
  --wallet-hotkey <wallet-hotkey-name> \
  --hotkey <miner-hotkey-ss58> \
  --netuid 66 \
  --commit "github-pr-head:unarbos/ninja@<head-sha>"

4. Push the exact same commit and open a PR to unarbos/ninja:main.
5. Make sure the PR title starts with the same hotkey.

The validator waits until it finds an open PR whose title hotkey, base branch,
and current head SHA match your pre-PR commitment.

Option B: Classic PR-Number Commitment
--------------------------------------

Use this when you are comfortable opening the PR first.

1. Commit your final agent.py change locally.
2. Push and open a PR to unarbos/ninja:main.
3. Make sure the PR title starts with your exact miner hotkey.
4. Copy the PR number and current PR head SHA.
5. Submit this commitment on-chain:

github-pr:unarbos/ninja#<pr-number>@<head-sha>

Helper command:

./scripts/commit_on_chain.py \
  --wallet-name <wallet-name> \
  --wallet-hotkey <wallet-hotkey-name> \
  --hotkey <miner-hotkey-ss58> \
  --netuid 66 \
  --commit "github-pr:unarbos/ninja#<pr-number>@<head-sha>"

Local Checks
------------

Run this before opening a PR:

./scripts/precommit_ninja_pr.py --hotkey <miner-hotkey-ss58> --judge

What it checks:

- worktree is clean by default
- HEAD is the commit that will be committed on-chain
- changed files stay inside agent.py
- solve(...) contract and basic static guardrails pass
- optional OpenRouter judge preflight approximates PR CI before upload

Important Details
-----------------

- The Git commit SHA must not change after you commit it on-chain.
- Rebasing, amending, changing the commit message, or using GitHub's web editor
  creates a new SHA and invalidates the old commitment.
- Required PR CI must still pass after upload: PR Scope Guard and OpenRouter PR
  Judge.
- One accepted submission spends the miner hotkey for the current submission
  window.
