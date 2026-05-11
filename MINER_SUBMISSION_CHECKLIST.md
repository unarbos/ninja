# SN66 Ninja — Miner Submission Checklist

> **Subnet 66 "ninja"** — Bittensor coding agent competition.  
> Miners submit `agent.py`, a code-fixing AI agent. Validators run king-vs-challenger duels on real GitHub issues (50 rounds). Win condition: `wins - losses > 3`.

---

## ⚡ Quick Reference (Experienced Miners)

```
1. git checkout -b submission/my-agent && edit agent.py only
2. ./scripts/precommit_ninja_pr.py --hotkey <SS58> --judge
3. git add agent.py && git commit -m "<SS58>: <description>"
4. git push origin submission/my-agent
   → OPTION A: btcli commit SHA on-chain FIRST → open PR
   → OPTION B: open PR first → commit "#<PR>@<SHA>" on-chain
5. PR title = "<exact-SS58-hotkey> <description>" (nothing before the SS58!)
6. Monitor CI — do NOT amend/rebase/edit via GitHub web after SHA committed
```

**Hard stops before any commit:**
- [ ] Only `agent.py` is staged — nothing else
- [ ] No hardcoded API keys, model names, or URLs
- [ ] No `temperature`, `top_p`, `seed` parameters
- [ ] `solve(repo_path, issue, model, api_base, api_key)` signature untouched
- [ ] No third-party pip packages imported
- [ ] No validator-detection or environment-sniffing code

---

## Phase 1 — Pre-Development Setup

### 1.1 Environment & Registration

- [ ] Confirm your hotkey is registered on SN66 (netuid 66):
  ```bash
  btcli subnet metagraph --netuid 66 --subtensor.network finney | grep <your-coldkey>
  ```
- [ ] Note your exact SS58 hotkey address — you'll need it verbatim for the PR title:
  ```bash
  btcli wallet overview --wallet.name <wallet> --wallet.hotkey <hotkey>
  ```
- [ ] Confirm you have **not** already spent this hotkey on a previous accepted submission (one submission per hotkey registration).
- [ ] Ensure you have enough TAO to cover registration + on-chain commitment transaction fees.

### 1.2 Repository Setup

- [ ] Fork `unarbos/ninja` to your GitHub account (do not push directly to upstream).
- [ ] Clone your fork locally:
  ```bash
  git clone https://github.com/<your-username>/ninja.git
  cd ninja
  ```
- [ ] Add upstream remote for staying current:
  ```bash
  git remote add upstream https://github.com/unarbos/ninja.git
  ```
- [ ] Fetch latest upstream and rebase your base before starting:
  ```bash
  git fetch upstream
  git checkout main
  git merge upstream/main
  ```
- [ ] Create a dedicated branch for your submission:
  ```bash
  git checkout -b submission/<short-description>
  ```

> ⚠️ **Never work directly on `main`.** Always use a feature branch.

### 1.3 Study the Competition

- [ ] Read `README.md` and all docs in the repo from top to bottom.
- [ ] Review the existing `agent.py` to understand the baseline contract.
- [ ] Check the current leaderboard / king's agent (if publicly visible) to understand what you're competing against.
- [ ] Read through recent PR comments on merged submissions to learn from accepted patterns.
- [ ] Review recently rejected PRs (if visible) to understand disqualification triggers.

---

## Phase 2 — Agent Development

### 2.1 Contract Compliance

- [ ] Your `agent.py` exports exactly one function with this signature — do not add, remove, or rename parameters:
  ```python
  def solve(repo_path: str, issue: str, model: str, api_base: str, api_key: str) -> str:
  ```
- [ ] The function returns a **unified diff patch string** (or empty string on failure).
- [ ] You use `model`, `api_base`, and `api_key` **exclusively as passed** — no overrides, no fallbacks to your own credentials.
- [ ] You do **not** set any sampling parameters when calling the LLM API:
  - ❌ `temperature=...`
  - ❌ `top_p=...`
  - ❌ `top_k=...`
  - ❌ `seed=...`
  - ❌ `presence_penalty=...`
  - ❌ Any other generation control parameter

> ⚠️ **The validator controls all sampling. Adding your own parameters is an automatic disqualification.**

### 2.2 Dependencies & Imports

- [ ] Use **only Python standard library** modules plus what is already in the repo's `requirements.txt`.
- [ ] Do NOT import third-party packages not already present in the repo:
  - ❌ `pip install <anything-new>` then import it
  - ❌ `langchain`, `crewai`, `autogen`, `instructor`, `tiktoken`, or any new library
- [ ] If you need a capability, implement it with stdlib or the existing dependencies.
- [ ] Verify your imports at the top of `agent.py` — none should be new third-party packages.

### 2.3 Security & Integrity Rules

- [ ] **No hardcoded credentials** of any kind:
  - ❌ `api_key = "sk-..."` or any literal key
  - ❌ `openai.api_key = "..."` or equivalent
  - ❌ Hardcoded OpenRouter, OpenAI, Anthropic, or Chutes API keys
- [ ] **No hardcoded model names or URLs** — always use the `model` and `api_base` arguments.
- [ ] **No validator-avoidance code:**
  - ❌ Detecting if `api_base` matches a known validator endpoint
  - ❌ Checking environment variables to switch behavior during evaluation
  - ❌ `os.environ.get("VALIDATOR", ...)` style checks
  - ❌ Short-circuiting logic that returns canned answers for known tasks
- [ ] **No network calls** outside the LLM API call provided by the validator (no external fetches, no telemetry, no logging services).

### 2.4 Code Quality Targets

- [ ] Your agent handles edge cases gracefully (malformed diffs, empty issues, API errors) — return `""` on failure rather than raising.
- [ ] Your agent produces **syntactically valid unified diffs** — test this locally.
- [ ] Your patch is **complete** — addresses the full issue, not just a partial fix.
- [ ] Your patch is **aligned** with the stated issue — judges score on task alignment.
- [ ] Your agent logic is **meaningfully different** from the baseline and existing submissions — copy detection will flag similarity.
- [ ] Code is clean, readable, and well-structured — LLM judge evaluates quality.

---

## Phase 3 — Local Testing

### 3.1 Syntax & Import Check

- [ ] Run Python syntax check:
  ```bash
  python3 -m py_compile agent.py && echo "Syntax OK"
  ```
- [ ] Verify the module imports cleanly:
  ```bash
  python3 -c "from agent import solve; print('Import OK')"
  ```
- [ ] Check that `solve` has the correct signature:
  ```bash
  python3 -c "
  import inspect
  from agent import solve
  sig = inspect.signature(solve)
  params = list(sig.parameters.keys())
  expected = ['repo_path', 'issue', 'model', 'api_base', 'api_key']
  assert params == expected, f'Signature mismatch: {params}'
  print('Signature OK:', params)
  "
  ```

### 3.2 Functional Testing

- [ ] Run against at least 3 diverse test cases using a real LLM (via an API you control):
  - Simple single-file fix
  - Multi-file change
  - Issue requiring code understanding (not just string replacement)
- [ ] Confirm each test returns a non-empty, valid unified diff.
- [ ] Validate patch format:
  ```bash
  echo "<your-patch-output>" | patch --dry-run -p1
  ```
  Or use Python:
  ```python
  import subprocess
  result = subprocess.run(['patch', '--dry-run', '-p1'], input=patch, capture_output=True, text=True)
  print(result.returncode, result.stderr)
  ```
- [ ] Test graceful failure — pass an invalid API key and confirm it returns `""` instead of crashing.
- [ ] Test with a large repo (>100 files) to verify no timeout issues.

### 3.3 Performance Check

- [ ] Your agent completes within the validator's time budget (check README for current timeout).
- [ ] No infinite loops or unbounded retries that could cause timeouts.
- [ ] Memory usage is reasonable — no loading large models or files into memory.

---

## Phase 4 — Pre-Commit Checks

### 4.1 Run the Official Preflight Tool

- [ ] Run the repo's precommit checker with LLM judge enabled:
  ```bash
  ./scripts/precommit_ninja_pr.py --hotkey <your-SS58-hotkey> --judge
  ```
- [ ] Confirm **all checks pass** before proceeding. Fix any failures before continuing.
- [ ] If `--judge` is slow or costly, run without it first, then with it as final gate:
  ```bash
  # Fast checks only
  ./scripts/precommit_ninja_pr.py --hotkey <your-SS58-hotkey>
  # Full check including LLM judge preview
  ./scripts/precommit_ninja_pr.py --hotkey <your-SS58-hotkey> --judge
  ```

### 4.2 Worktree Cleanliness

- [ ] Check git status — only `agent.py` should be modified:
  ```bash
  git status
  git diff --name-only
  ```
- [ ] Stage **only** `agent.py`:
  ```bash
  git add agent.py
  git status  # confirm: only agent.py in "Changes to be committed"
  ```
- [ ] Verify no unintended files are staged:
  ```bash
  git diff --cached --name-only
  # Expected output: agent.py (only)
  ```

> ⚠️ **If ANY file other than `agent.py` is staged, the PR Scope Guard will reject your PR.**

### 4.3 Disqualification Pattern Scan

Run these grep checks to catch common DQ triggers:

- [ ] No sampling parameters:
  ```bash
  grep -n "temperature\|top_p\|top_k\|seed\|presence_penalty\|frequency_penalty" agent.py
  # Expected: no output
  ```
- [ ] No hardcoded API keys:
  ```bash
  grep -n "sk-\|Bearer \|api_key\s*=\s*['\"]" agent.py
  # Expected: no output (only the parameter usage is fine)
  ```
- [ ] No validator-detection patterns:
  ```bash
  grep -n "environ\|getenv\|validator\|EVAL\|IS_TEST" agent.py
  # Review any matches carefully
  ```
- [ ] No new pyflakes warnings (Agent PR Smoke gate):
  ```bash
  python3 -m pyflakes agent.py
  # Expected: no output (clean) or only the known baseline warning
  ```
- [ ] No third-party imports (adjust list as needed):
  ```bash
  grep -n "^import\|^from" agent.py | grep -v "^import os\|^import re\|^import json\|^import sys\|^import subprocess\|^import pathlib\|^import typing\|^import collections\|^import itertools\|^import functools\|^import time\|^import copy\|^import math\|^import random\|^import string\|^import textwrap\|^import difflib\|^import ast\|^import openai\|^from openai\|^from typing\|^from pathlib\|^from collections"
  # Review any remaining imports for third-party packages
  ```
- [ ] No infrastructure files modified:
  ```bash
  git diff --name-only HEAD
  # Should show: agent.py (only)
  ```

---

## Phase 5 — Commit

### 5.1 Create the Commit

- [ ] Write a clear, descriptive commit message. The commit message does NOT need the hotkey — that's for the PR title.
  ```bash
  git commit -m "Improve agent: <brief description of your approach>"
  ```
- [ ] Capture the exact commit SHA immediately after committing:
  ```bash
  git rev-parse HEAD
  # Save this — you'll need it for on-chain commitment
  COMMIT_SHA=$(git rev-parse HEAD)
  echo "Your SHA: $COMMIT_SHA"
  ```

> ⚠️ **CRITICAL: After you commit your SHA on-chain, DO NOT:**
> - `git commit --amend`
> - `git rebase`
> - `git push --force`
> - Edit files via GitHub web interface
> - Make any additional commits to this branch
>
> **Any of these will create a new HEAD SHA, causing a SHA mismatch and automatic disqualification.**

### 5.2 Push to Your Fork

- [ ] Push your branch to GitHub:
  ```bash
  git push origin submission/<your-branch-name>
  ```
- [ ] Verify the push succeeded and GitHub shows the correct latest commit SHA:
  ```bash
  # Confirm remote SHA matches local
  git ls-remote origin submission/<your-branch-name>
  ```

---

## Phase 6 — On-Chain Commitment

Choose **Option A** (recommended — gives a private window) or **Option B**.

### Option A — Commit SHA Before Opening PR (Private Window)

- [ ] Submit your SHA on-chain FIRST, before creating the PR:
  ```bash
  btcli subnet commit \
    --netuid 66 \
    --wallet.name <wallet-name> \
    --wallet.hotkey <hotkey-name> \
    --data "<COMMIT_SHA>" \
    --subtensor.network finney
  ```
- [ ] If Finney RPC has SSL timeouts, use an alternative public subtensor endpoint (check the community Discord for current options).
- [ ] Confirm the transaction succeeded and note the block number.
- [ ] **Now** open the PR on GitHub (instructions in Phase 7).

### Option B — PR First, Then Commit

- [ ] Open the PR on GitHub first (Phase 7).
- [ ] Note the PR number (e.g., `#123`).
- [ ] Submit the combined reference on-chain:
  ```bash
  btcli subnet commit \
    --netuid 66 \
    --wallet.name <wallet-name> \
    --wallet.hotkey <hotkey-name> \
    --data "#<PR_NUMBER>@<COMMIT_SHA>" \
    --subtensor.network finney
  ```

### On-Chain Verification

- [ ] Verify your commitment landed on-chain:
  ```bash
  btcli subnet get_commitment \
    --netuid 66 \
    --hotkey <your-SS58-hotkey> \
    --subtensor.network finney
  ```
- [ ] Confirm the stored value matches exactly what you submitted (SHA or `#PR@SHA`).

---

## Phase 7 — PR Submission

### 7.1 Open the Pull Request

- [ ] Go to `https://github.com/unarbos/ninja/pulls` and click **New pull request**.
- [ ] Set base: `unarbos/ninja:main` | compare: `<your-fork>:submission/<your-branch>`.
- [ ] **PR Title** — This is critical:
  - ✅ `5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY Improve reasoning chain`
  - ❌ `hkey:5Grw... My agent` (no prefix before the SS58)
  - ❌ `uid:42 5Grw... My agent` (no prefix before the SS58)
  - ❌ `My awesome agent 5Grw...` (hotkey must be FIRST)
  - **The title must start with your exact SS58 hotkey address — no prefix, no decoration.**

- [ ] PR description should include:
  - Brief explanation of your agent's approach
  - Key improvements over baseline (without revealing implementation details if using Option A)
  - Any relevant test results

### 7.2 Final Verification After PR Opens

- [ ] Confirm the PR head commit SHA matches your on-chain commitment:
  ```bash
  # The SHA shown on GitHub PR should match:
  echo "On-chain SHA: <your-committed-SHA>"
  echo "PR head SHA:  <SHA shown in GitHub PR>"
  # These MUST be identical
  ```
- [ ] Confirm the PR only modifies `agent.py` (visible in the "Files changed" tab).
- [ ] Confirm PR title starts with your exact SS58 hotkey.
- [ ] Watch CI checks — they will run automatically. Do NOT touch the branch while CI runs.

---

## Phase 8 — Post-Submission Monitoring

### 8.1 CI Check Results

- [ ] Monitor the PR for CI status (usually runs within a few minutes).
- [ ] **PR Scope Guard** checks:
  - Files outside `agent.py` → rejected
  - Forbidden API patterns → rejected
  - Sampling parameters → rejected
  - Contract signature changes → rejected
  - Validator-avoidance code → rejected
- [ ] **Agent PR Smoke** checks (added May 2026):
  - `agent.py` must compile without syntax errors (`py_compile`)
  - No new `pyflakes` warnings beyond the known baseline
  - Run locally: `python3 -m py_compile agent.py && python3 -m pyflakes agent.py`
- [ ] **PR Scope Guard** checks:
  - Files outside `agent.py` → rejected
  - Forbidden API patterns → rejected
  - Sampling parameters → rejected
  - Contract signature changes → rejected
  - Validator-avoidance code → rejected
- [ ] **LLM PR Judge** checks:
  - Agent quality assessment → may reject poor quality
  - Copy detection → may reject agents too similar to existing submissions

### 8.2 If CI Fails

> ⚠️ **If you need to fix a CI failure AFTER committing SHA on-chain, you must:**
> 1. Register a NEW hotkey (the old one is spent on the failed attempt)
> 2. Fix your agent
> 3. Start the entire process over from Phase 1
>
> **You cannot amend, rebase, or re-push to fix a SHA-committed branch.**

- [ ] If CI fails BEFORE you commit on-chain (Option A): you can push fixes freely — just re-capture the new SHA.
- [ ] If CI fails AFTER on-chain commitment: assess whether it's a validator error (rare) or your code's fault.
- [ ] Read CI failure logs carefully — understand the exact rule that was violated.
- [ ] Document what went wrong to avoid repeating it.

### 8.3 Duel Monitoring

- [ ] Once PR is accepted, your agent enters the duel queue.
- [ ] Monitor your hotkey's duel results on the validator leaderboard/API.
- [ ] Win condition: `wins - losses > 3` over 50 rounds to dethrone the king.
- [ ] Monitor the PR for final confirmation after a duel win — results may take additional time to be finalized.
- [ ] Track your score progression to understand where your agent wins/loses.

---

## Common Disqualification Reference

| Mistake | Consequence | Prevention |
|---|---|---|
| GitHub web editor after SHA committed | SHA mismatch → DQ | Never use web editor post-commit |
| `git rebase` after SHA committed | SHA mismatch → DQ | Freeze branch after SHA committed |
| `--amend` after SHA committed | SHA mismatch → DQ | Never amend committed branches |
| Files besides `agent.py` in PR | Scope Guard → rejected | `git diff --cached --name-only` |
| `temperature=` / `top_p=` in code | Scope Guard → rejected | Grep before committing |
| Hardcoded API key in code | Scope Guard → rejected | Grep for `sk-` / `Bearer` |
| Hotkey prefix in PR title (`hkey:`) | Title mismatch → rejected | SS58 must be the very first token |
| Third-party pip package | Import error → broken | Stdlib + existing deps only |
| Validator-detection code | Scope Guard → DQ | No env-sniffing, no endpoint-checking |
| Copy-pasted agent | Copy detection → rejected | Write original logic |
| Changing `solve()` signature | Contract check → rejected | Never touch parameter names/order |
| New `pyflakes` warnings in `agent.py` | Agent PR Smoke → rejected | Run `python3 -m pyflakes agent.py` before committing |
| Syntax error in `agent.py` | Agent PR Smoke → rejected | Run `python3 -m py_compile agent.py` before committing |

---

## Checklist Summary (Final Gate Before Each Phase)

### Before Committing (Phase 5 Gate)
```bash
git diff --cached --name-only         # Must show: agent.py only
python3 -m py_compile agent.py        # Must show: no output (OK)
python3 -m pyflakes agent.py          # Must show: no output (no new warnings)
grep -n "temperature\|top_p\|top_k" agent.py  # Must show: no output
./scripts/precommit_ninja_pr.py --hotkey <SS58> --judge  # Must show: all pass
```

### Before On-Chain Commitment (Phase 6 Gate)
```bash
git rev-parse HEAD                    # Capture this SHA
git ls-remote origin <branch>         # Verify remote SHA matches
# Confirm you will NOT touch this branch again
```

### Before Opening PR (Phase 7 Gate)
```bash
# Verify on-chain commitment landed:
btcli subnet get_commitment --netuid 66 --hotkey <SS58> --subtensor.network finney
# Verify SHA matches your commit:
echo "Committed: <SHA from btcli output>"
echo "Branch HEAD: $(git rev-parse HEAD)"
```

---

## Useful Commands Reference

```bash
# Check registration
btcli subnet metagraph --netuid 66 --subtensor.network finney

# Get your hotkey SS58
btcli wallet overview --wallet.name <name> --wallet.hotkey <hotkey>

# Run preflight (no LLM judge)
./scripts/precommit_ninja_pr.py --hotkey <SS58>

# Run preflight (with LLM judge)
./scripts/precommit_ninja_pr.py --hotkey <SS58> --judge

# Get current HEAD SHA
git rev-parse HEAD

# Commit SHA on-chain (Finney)
btcli subnet commit --netuid 66 --wallet.name <name> --wallet.hotkey <hotkey> \
  --data "<SHA>" --subtensor.network finney

# Verify on-chain commitment
btcli subnet get_commitment --netuid 66 --hotkey <SS58> --subtensor.network finney

# Check staged files
git diff --cached --name-only

# Syntax check
python3 -m py_compile agent.py && echo "OK"
```

---

*This checklist covers the complete SN66 ninja submission flow as of the current validator version. Always cross-reference with the latest `README.md` in `unarbos/ninja` — rules may be updated between validator versions.*