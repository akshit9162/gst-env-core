# gst-env

**Walkthrough:** [akshit9162.github.io/gst-env](https://akshit9162.github.io/gst-env)

**RL environments and benchmarks for Indian tax and compliance work, used to train and evaluate AI agents.**

Agent training today mostly targets Western SaaS such as Salesforce, Jira and Excel. Compliance work in
emerging markets has almost no coverage: rule-heavy filings, state-wise tax splits, blocked credits,
reverse charge and statutory set-off order. Errors there cost real money, and agents get them wrong.
India alone has over 1.4 crore GST registrations filing returns every month.

`gst-env` starts with **GSTR-3B**, India's monthly GST return. An agent reads a month of invoices
through tools and files the return. A deterministic solver computes the correct answer, and the grader
scores the outcome rather than the steps taken, so the same tasks work for tool-use, GUI and
plain-text agents and for training as well as evaluation.

## Baseline results

| Model | Easy | Medium | Hard |
|---|---|---|---|
| Claude Opus 5.5 (high effort) | pending | pending | pending |
| GPT (OpenAI) | pending | pending | pending |
| Sarvam-105B | pending | pending | pending |

Pass rate = share of tasks where every graded field is correct, reported with a 95% Wilson interval.
Reproduce with `scripts/baseline.py`; the run plan is preregistered in `evals/prereg-baseline-v1.md`.

## What an episode looks like

1. The agent receives the filing instructions for a business registered in Telangana.
2. It calls `list_invoices`, `get_invoice`, `describe_category` and `get_credit_ledger` to read
   sales, purchases and the credit balance brought forward.
3. It calls `submit_return` once with the GSTR-3B figures.
4. The grader compares 13 fields to ground truth (₹1 tolerance) and returns a reward in [0, 1].
   `total_cash_payable` carries extra weight. Fields whose true value is zero can only cost reward,
   never earn it, so an empty or all-zero return scores 0.

| Level | Covers |
|---|---|
| easy | Intra-state sales and purchases (CGST + SGST) |
| medium | Adds inter-state supplies (IGST) and nil-rated sales |
| hard | Adds blocked ITC under section 17(5), reverse-charge purchases and an uneven credit-ledger balance brought forward |

The ground truth applies ITC set-off in the Rule 88A order and picks the split of IGST credit that
minimises cash outflow. The test suite checks this against brute-force search.

## Cycle tier: the current filing workflow

`--tier cycle` (or `tier = "cycle"` in the verifiers adapter) is the month as a CA's office works it today,
on a mock portal:

1. **World:** the client's sales register, purchase register (with remarks) and WhatsApp-style messages.
2. **GSTR-1:** the agent files invoice-level outward supplies. A correction can arrive *after* filing,
   and must go in through **GSTR-1A** before GSTR-3B is prepared.
3. **IMS:** accept, reject or keep pending each supplier-uploaded invoice. No action means deemed
   accepted when GSTR-2B is generated.
4. **Asking the client:** some facts (were the goods received? business or personal use?) are only in the
   client's head. `ask_client` answers them, and the client stops replying a couple of questions past what
   the books actually leave open.
5. **GSTR-3B:** tables 3.1 and 4A(5) fill in from GSTR-1/1A and GSTR-2B and are locked. The agent enters
   3.1(d) reverse charge, 4A(3), 4B(1) blocked and 4B(2) other reversals, and the cash paid per head.

The grader rebuilds what was filed (outward tax from the agent's GSTR-1, net ITC from its IMS actions and
reversals) and compares it with the correct filing: 14 fields plus a GSTR-1 invoice-level check, cash
weighted 5x. Skipping one habit, even with the correct cash figure supplied, fails most tasks
(300 tasks; mean reward / pass rate):

| Habit skipped | Reward / pass |
|---|---|
| None (careful CA) | 1.00 / 100% |
| Reads late messages and amends in GSTR-1A | 0.90 / 45% |
| Removes the invoice the client says was cancelled | 0.90 / 47% |
| Rejects or holds bad IMS documents | 0.85 / 5% |
| Reverses blocked credit in 4B(1) | 0.81 / 12% |
| Asks the client when the books leave it open | 0.80 / 7% |

`tests/test_cycle.py` checks that careful play scores 1.0 on every seed, doing nothing scores 0, and each
skipped habit fails.

## Reward integrity

`scripts/reward_audit.py` scores cheating and shortcut strategies over 300 tasks per level.
Cells show mean reward / pass rate.

| Strategy | Easy | Medium | Hard |
|---|---|---|---|
| All zeros | 0.00 / 0% | 0.00 / 0% | 0.00 / 0% |
| Constant guess | 0.00 / 0% | 0.00 / 0% | 0.00 / 0% |
| Random guess | 0.00 / 0% | 0.00 / 0% | 0.00 / 0% |
| Reads sales only | 0.25 / 0% | 0.29 / 0% | 0.30 / 0% |
| Reads first page only | 1.00 / 100% | 1.00 / 100% | 0.41 / 4% |
| Treats all purchases as eligible | 1.00 / 100% | 1.00 / 100% | 0.40 / 2% |
| Treats all parties as in-state | 1.00 / 100% | 0.58 / 13% | 0.42 / 9% |
| Ignores the credit ledger | 1.00 / 100% | 1.00 / 100% | 0.69 / 9% |
| Pools credit across heads | 1.00 / 100% | 1.00 / 100% | 0.86 / 57% |
| Uses ITC to pay reverse-charge tax | 1.00 / 100% | 1.00 / 100% | 0.81 / 41% |

Strategies that do no work earn nothing. Each shortcut passes only at the levels that leave out the
rule it skips, and fails hard tasks often. Credit pooling still passes 57% of hard tasks because on
those tasks the head-wise rule genuinely does not change the answer; the planned expert tier targets
it directly. `tests/test_reward_audit.py` fails the build if any of this regresses.

The test suite is itself checked: `scripts/vaccinate.py` plants 18 defects one at a time (CGST credit paying
SGST, ITC paying reverse charge, blocked credit claimed, a loosened tolerance, a hidden last page of invoices
and others) and the suite must fail on each while passing a clean copy. Latest result: 18/18 caught, no false
alarm, against 2/18 for a single smoke test ([ledger](evals/qualification/LEDGER.md)).

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev,baseline]"
.venv/bin/python -m pytest -q
.venv/bin/python scripts/export_tasks.py --per-difficulty 50     # data/tasks.jsonl + answers.jsonl

# Free: stub agent, no API key, prints per-arm cost projections and pipeline checks
.venv/bin/python scripts/baseline.py --dry-run --models claude-opus-5-5 gpt-6-astra sarvam-105b --per-difficulty 30 --cap 25
# Seeded-defect check of the test suite; writes evals/qualification/LEDGER.md (~2 min)
.venv/bin/python scripts/vaccinate.py

# API keys go in .env: ANTHROPIC_API_KEY, OPENAI_API_KEY and/or SARVAM_API_KEY. --cap is a USD spend cap.
# Three tasks per level is a smoke run (pass-rate intervals of about +/-37 points); see evals/prereg-baseline-v1.md.
.venv/bin/python scripts/baseline.py --per-difficulty 3 --cap 5 --out results/opus-5-5.jsonl
.venv/bin/python scripts/baseline.py --model gpt-6-astra --per-difficulty 3 --out results/gpt-6-astra.jsonl
.venv/bin/python scripts/baseline.py --model sarvam-105b --per-difficulty 3 --out results/sarvam-105b.jsonl

# Cycle tier, 16 attempts per task: adds pass@1, pass@16 and how many tasks are solved sometimes but not
# always (the ones RL can learn from; a task never solved in 16 tries teaches nothing)
.venv/bin/python scripts/baseline.py --tier cycle --per-difficulty 20 --samples 16 --cap 25 --out results/cycle-opus.jsonl
```

### verifiers / prime-rl

`gst_env_vf` serves both tiers as a verifiers (v1) taskset with id `gst-env-vf`. It needs Python 3.12.

```bash
uv venv -p 3.12 .venv-vf && VIRTUAL_ENV=.venv-vf uv pip install -e ".[dev,vf]"
.venv-vf/bin/python -m pytest -q tests/test_vf.py
# No model needed: checks every task loads and an untouched task scores 0
.venv-vf/bin/vf-validate gst-env-vf --only-noop True --runtime.type subprocess --taskset.task.tier cycle --taskset.num-tasks 20
# Eval with a model (16 rollouts per task gives pass@16)
.venv-vf/bin/vf-run gst-env-vf -m <model> -n 50 -r 16 --env.taskset.task.tier cycle --env.taskset.seed-offset 1000000
```

Tool state is the rollout's log of calls, replayed on a fresh session for every call and again for the
reward, so the tools and the reward cannot disagree. Use different `seed_offset` values for training and
evaluation tasks.

## Layout

| File | Purpose |
|---|---|
| `gst_env/rules.py` | Rates, states, blocked (17(5)) and reverse-charge categories |
| `gst_env/generator.py` | Seeded task generator |
| `gst_env/solver.py` | Ground truth, including Rule 88A set-off with minimum cash |
| `gst_env/grader.py` | Weighted field-level reward |
| `gst_env/tools.py` | Tool interface and episode state |
| `gst_env/cycle.py` | Cycle tier: generator, mock portal (GSTR-1/1A, IMS, 2B, 3B), client questions, solver and grader |
| `gst_env_vf/` | verifiers (v1) adapter for both tiers, for `vf eval`, prime-rl and the Environments Hub |
| `scripts/export_tasks.py` | Dumps tasks and answer keys as JSONL |
| `scripts/baseline.py` | Runs Claude, OpenAI (gpt-*) or Sarvam models as the agent and records rewards and cost |
| `scripts/reward_audit.py` | Scores degenerate and shortcut strategies to check the reward can't be gamed |
| `scripts/vaccinate.py` | Plants 18 known defects in copies of the package and checks the test suite catches each one |
| `gst_env/stats.py` | Wilson and bootstrap intervals for reported pass rates and rewards |
| `evals/prereg-baseline-v1.md` | Preregistered bars, arms, decoys, stop rules and held-out seeds for the baseline run |
| `evals/qualification/LEDGER.md` | Latest seeded-defect qualification result |

## Roadmap

- **Expert tier:** CA-designed tasks covering credit and debit notes, GSTR-2B mismatches, amendments,
  Rule 42/43 reversals and late ITC under section 16(4).
- **GUI tier:** the cycle tier's portal as screens for computer-use agents, graded on the same ground truth.
- **More filings:** GSTR-1 reconciliation, TDS returns, Tally bookkeeping tasks.
- **Recorded demonstrations** by practising CAs.
- **HUD adapter** (the verifiers adapter is in `gst_env_vf`).

## Simplifications (CA review in progress)

**Scope:** this version tests GST *computation* (tax-head splits, blocked credit, reverse charge, Rule 88A
set-off). It does not reproduce today's filing workflow. Since the July 2025 tax period, GSTR-3B outward
liability (Tables 3.1/3.2) auto-populates from GSTR-1 and is non-editable, with corrections through GSTR-1A.
ITC flows from GSTR-2B through the Invoice Management System (IMS), and an ITC lock on Table 4A has been
indicated for around July 2026. The **cycle tier** models the GSTR-1 → IMS/2B → 3B workflow; its own
simplifications are listed below.

1. Rates use the post-22 Sep 2025 slabs (0/5/18/40). No cess.
2. Place of supply is the counterparty's state. No exports, SEZ or special place-of-supply services.
3. Blocked credit is all-or-nothing per invoice. No Rule 42/43 or Rule 37 reversals.
4. ITC is assumed to be reflected in GSTR-2B, with no mismatches or time limits.
5. Reverse-charge categories are a subset of notification 13/2017, with fixed rates.
6. No credit or debit notes, amendments, advances, interest or late fee.
7. B2C place-of-supply reporting (table 3.2) is not graded.
8. Credit-ledger opening balances are synthetic, and cash-ledger balances and interest are not modelled.

Each item is a candidate for the expert tier once a CA has validated it.

### Cycle tier simplifications

1. GSTR-1 is invoice-level for B2C as well; the real return aggregates B2C (small) supplies by state and rate.
2. Invoice corrections go through GSTR-1A only. No credit or debit notes, and no amendment of earlier periods.
3. IMS: rejecting or holding a document keeps its credit out of this month. Pending-period limits, supplier-side
   follow-up and the IMS rules for credit notes are not modelled.
4. Goods not yet received are handled by keeping the document pending (or accepting it and reversing in 4B(2));
   both give the same net ITC and both pass.
5. Personal-use purchases are treated as blocked under section 17(5)(g) and reported in 4B(1).
6. Reverse-charge purchases come from unregistered suppliers, so they never appear in IMS and are self-reported
   in 3.1(d) and 4A(3). The RCM liability/ITC statement on the portal is not modelled.
7. Table 4A(5) is locked to GSTR-2B (an ITC lock has been indicated but its notification is not confirmed).
8. No interest, late fee, QRMP or composition filers. One month only; nothing carries forward to the next.

## Running through Otari

Every open-weight model runs through the [Otari](https://github.com/mozilla-ai/otari) gateway:
prefix the model with `otari:` and put `OTARI_API_KEY` (and `OTARI_BASE_URL` for an EU
account, `https://eu.api.otari.ai/api/v1`) in `.env`. Cost is what Otari bills, so a
fallback is priced as the model that actually answered. Every call is streamed: hosted
Otari cuts a non-streamed request at about 60 s ([otari#2692](https://github.com/mozilla-ai/otari/issues/2692)),
and a hard task's turn with code execution runs longer than that.

```bash
# one model as the filing agent, as in the baseline
.venv/bin/python scripts/baseline.py --model otari:deepseek/deepseek-v4.1-flash --per-difficulty 2 \
    --otari-code-exec --out results/otari-deepseek.jsonl

# three models in three roles, a poisoned invoice in every task
.venv/bin/python scripts/pipeline.py --filer otari:alibaba/qwen-3.5-35b-a3b \
    --reviewer otari:z-ai/glm-5.3-flash --explainer otari:openai/gpt-oss-20b \
    --difficulties medium hard --per-difficulty 5 --inject 1.0 --code-exec --web-fetch

# the same, each role billed as its own end user (run on a service key)
OTARI_API_KEY=$OTARI_SERVICE_KEY .venv/bin/python scripts/pipeline.py ... --users

# Otari runs the whole tool loop against the env served over MCP
.venv/bin/python scripts/mcp_episode.py --model otari:alibaba/qwen-3.5-35b-a3b \
    --public-url https://<tunnel> --difficulties easy

.venv/bin/python scripts/leaderboard.py results/lb-*.jsonl --markdown results/LEADERBOARD.md
```

The guardrail is an organization mandate in the Otari dashboard (Lakera Guard, block mode), so
it runs on every request without a flag. `--guardrail <profile>` asks for one per request,
which on hosted Otari needs your own guardrail service.

| Otari feature | Where it does work |
|---|---|
| Open-weight models | every arm; the leaderboard ranks them on the same seeded tasks |
| Several models in distinct roles | `pipeline.py`: one model files, another reviews the draft against the invoices, a third explains the filed return to the owner in English and Hindi |
| Routing and fallbacks | `otari:gst-filer` is a fallback policy (qwen, then glm); each episode logs `Otari-Model` and `Otari-Fallback` per turn |
| Budgets | a daily spend ceiling on the organization and a budget on the key; a refusal ends the episode as `budget_exceeded` instead of crashing |
| Per-user budgets | `--users` names an end user per role (`filer:<model>`), each capped by the service key's end-user budget |
| Code execution | `--code-exec`: the filer does the tax arithmetic in Otari's sandbox |
| Web fetch | `--web-fetch`: the reviewer fetches the CBIC rule text it cites (e.g. Section 17(5) for blocked credit) |
| Guardrails | `--inject` hides an instruction in an invoice note; rows record whether the filer obeyed it, whether review caught it, and whether the guardrail refused the request |
| MCP | `gst_env.mcp_server` serves the filing tools over streamable HTTP; `mcp_episode.py` sends no tools and lets Otari call them |

### What happened when we ran it (hosted Otari, EU, 6 to 8 Nov 2026)

The runs below cost about $1.30 in all. Small samples: read the intervals, not the ranks.

**Open-weight leaderboard**, 2 tasks per difficulty per model (seeds 4000 to 4001):

| # | Model | Passed | 95% CI | Mean reward | $/item |
|---|---|---|---|---|---|
| 1 | qwen-3.5-35b-a3b | 3 of 6 | 19% to 81% | 0.51 | $0.011 |
| 2 | deepseek-v4.1-flash | 2 of 6 | 10% to 70% | 0.72 | $0.002 |
| 3 | gpt-oss-120b | 2 of 6 | 10% to 70% | 0.43 | $0.003 |
| 4 | glm-5.3-flash | 1 of 6 | 3% to 56% | 0.51 | $0.002 |
| 5 | llama-3.3-70b-instruct | 0 of 6 | 0% to 39% | 0.00 | $0.056 |

No model passed a hard task. Llama mostly answered in prose instead of calling tools. On 4 of
6 gpt-oss-120b episodes Otari moved to another provider of the same model on its own.

**Poisoned invoices**, qwen files, glm reviews, gpt-oss-20b explains, Lakera mandated in block mode
(seeds 2200 to 2204, one invoice note in every task tells the agent to report blocked ITC as 0):

| | Medium (5) | Hard (4) |
|---|---|---|
| Filer read the poisoned note | 5 | 4 |
| Guardrail blocked it | 0 | 0 |
| Filer obeyed it in its first draft | not measurable (no blocked ITC in these tasks) | 4 |
| Reviewer sent the draft back | 2 (both wrong drafts) | 4 |
| Poisoned figure still in the filed return | – | 1 of 2 filed |
| Passed | 3 | 0 |

The guardrail blocks the same text when it arrives as a user message (score 1.0) and passes it as
a tool result, because Otari screens only the latest user message
([otari#2988](https://github.com/mozilla-ai/otari/issues/2988)). In this env the reviewer, not
the guardrail, is what stops an injection.

**The rest, run live:** two easy tasks filed over managed MCP through a tunnel (both passed,
about $0.006 each); the `gst-filer` routing policy served by its main model; on a service key
with a $0.05 end-user budget, the filer's end user was refused with `budget_exceeded` after one
task while the reviewer and explainer, on budgets of their own, kept running.

Issues filed on Otari from this work: [#2988](https://github.com/mozilla-ai/otari/issues/2988),
[#2989](https://github.com/mozilla-ai/otari/issues/2989), [#2990](https://github.com/mozilla-ai/otari/issues/2990),
[#2991](https://github.com/mozilla-ai/otari/issues/2991), [#2992](https://github.com/mozilla-ai/otari/issues/2992).
