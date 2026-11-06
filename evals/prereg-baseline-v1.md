# Preregistration: GSTR-3B baseline v1

Status: **draft, not frozen.** Freeze by committing this file before the first priced call. After that,
changing a bar, arm, decoy or stop rule means a new preregistration (v2), not an edit.

Follows the `eval-plan` protocol (preregister, zero-cost dry run, three-item priced pilot, then approval).
The gstack runtime that skill ships with is not installed, so the dry run and spend ledger are built into
`scripts/baseline.py` (`--dry-run`, `--cap`).

## Bars

| Metric | Bar | Direction | What a miss means |
|---|---|---|---|
| Submission rate (any arm, any level) | >= 95% | higher | Harness or instruction bug, not model skill. Read transcripts before publishing any number. |
| Best arm, easy pass rate | >= 70% | higher | Easy tasks are meant to be solvable. A miss points at the instructions or tool design. |
| Best arm, hard pass rate | <= 80% | lower | Hard tier does not discriminate. Build the expert tier before publishing the table. |
| Actual cost per item vs dry-run projection | within 0.5x-2x | inside | The cost model is wrong; recompute the full-run projection before approving it. |

Pass rates are reported with 95% Wilson intervals and mean reward with a bootstrap interval
(`gst_env/stats.py`). No claim of "model A beats model B" unless their intervals at that level do not overlap.

## Arms

Per-item cost from `python scripts/baseline.py --dry-run --models claude-opus-5-5 gpt-6-astra sarvam-105b --per-difficulty 30`
(estimate at 1,500 output tokens per turn, list prices as of Oct 2026, treat as +/-50%):

| Arm | Settings | Est. $/item (easy / medium / hard) | Est. 90 items |
|---|---|---|---|
| `claude-opus-5-5` | effort high, max_tokens 16000, automatic prompt caching | 0.17 / 0.20 / 0.30 | $20 |
| `gpt-6-astra` | reasoning effort high, max_output_tokens 16000 | 0.42 / 0.52 / 0.78 | $52 |
| `sarvam-105b` | max_tokens 8192, tool_choice auto | 0.007 / 0.008 / 0.013 | $0.85 |

## Comparison

stack: none
why: every arm gets only the filing instructions (`gst_env/tools.py: INSTRUCTIONS`) and the five tool
definitions. No skill files, system prompts or hooks are loaded into any candidate.

## Decoys

The scorer must give low or zero reward to inputs that do no real work or skip a rule:

- 3 degenerate strategies (all zeros, constant guess, random guess) and 7 shortcut strategies
  (sales only, first page only, all purchases eligible, all in-state, ignores ledger, pooled set-off,
  ITC pays reverse charge), from `scripts/reward_audit.py` over 300 tasks per level. Each degenerate
  strategy must score under 0.02 with 0% pass; each shortcut must pass at most 65% of hard tasks.
  Enforced by `tests/test_reward_audit.py`.
- 18 seeded defects in the solver, grader, tools and generator (`scripts/vaccinate.py`). The suite must
  catch every one and pass a clean copy. Ledger: `evals/qualification/LEDGER.md` (18/18 on 2026-11-01).

A decoy that passes proves the scorer rewards something other than a correct return, and blocks the run.

## Stop rules

- Spend cap per arm: **$25** (`--cap 25`). An episode is admitted only while spent + reserved + its
  estimate fits. A refused reservation ends that arm's run; no retries to make room.
- Pilot: stop without asking for approval if any arm submits fewer than 2 of its 3 pilot items, if actual
  pilot cost is more than 2x the projection, or if any decoy check above fails.
- `budget_exhausted` on any arm stops that arm and is reported in the results table.
- The GPT arm's 90-item projection ($52) is over the $25 cap. Approval for it needs a higher cap or fewer
  items (see sample size below).

## Held-out exposure

| Seed range | Use | Exposure |
|---|---|---|
| 0-49 per level | `data/tasks.jsonl` (public in the repo) | Public. Never used for reported numbers. |
| 1000-1029 per level | This baseline (`--seed-offset 1000`) | None before this run. |
| 5000-5299 | Reward audit | Scorer checks only, no model sees them. |
| 7000-7149 | Vaccinate probes | Scorer checks only, no model sees them. |
| 9000-9999 | **Held out** for a future headline result | Never run, never put in a prompt, never summarised. |

## Blinding

Fixture root: tasks are generated in memory, so there is no directory for a candidate to read. The task id
(which names the difficulty) is never shown. A check on 2026-11-01 found no evaluation words (eval, test,
judge, score, benchmark, grade, and similar) in the instructions, the tool descriptions or the retry prompt.
Grading is by code (`gst_env/grader.py`), so there is no judge to blind.

## Sample size

Wilson half-width at the worst case (p = 0.5): 3 tasks per level gives about +/-37 points, 30 gives about
+/-17, 43 gives +/-14 and 97 gives +/-10. The README's `--per-difficulty 3` is a smoke run, not a result.
Proposed full run: 30 per level (90 items per arm). Move to 97 per level before claiming a gap between
two arms smaller than about 20 points.

## Pilot and approval commands

```bash
# Step 2, zero cost (no API key needed)
.venv/bin/python scripts/baseline.py --dry-run --models claude-opus-5-5 gpt-6-astra sarvam-105b --per-difficulty 30 --cap 25

# Step 3, priced pilot: 3 items per arm (one per level)
.venv/bin/python scripts/baseline.py --model claude-opus-5-5 --per-difficulty 1 --cap 3 --out results/pilot-opus-5-5.jsonl
.venv/bin/python scripts/baseline.py --model gpt-6-astra --per-difficulty 1 --cap 5 --out results/pilot-gpt-6-astra.jsonl
.venv/bin/python scripts/baseline.py --model sarvam-105b --per-difficulty 1 --cap 1 --out results/pilot-sarvam-105b.jsonl

# Step 4, only after the owner approves the pilot's spend table
.venv/bin/python scripts/baseline.py --model claude-opus-5-5 --per-difficulty 30 --cap 25 --out results/opus-5-5.jsonl
```
