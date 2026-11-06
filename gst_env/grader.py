"""Outcome-based grading: compares a submitted return to ground truth,
independent of how the agent got there (GUI, tools or plain text)."""

from .generator import Task
from .solver import HEADS, solve

TOLERANCE = 1  # rupees

# Field -> weight. The headline number carries the most weight; per-head
# fields catch agents that get the total right for the wrong reasons.
WEIGHTS = {
    "outward_taxable_value": 1,
    "nil_exempt_value": 1,
    **{f"outward_tax.{h}": 1 for h in HEADS},
    **{f"rcm_tax.{h}": 1 for h in HEADS},
    **{f"itc_available.{h}": 1 for h in HEADS},
    "blocked_itc": 1,
    "total_cash_payable": 5,
}


def _get(d: dict, path: str):
    for key in path.split("."):
        if not isinstance(d, dict) or key not in d:
            return None
        d = d[key]
    return d


def grade(task: Task, submission: dict) -> dict:
    return score(solve(task), submission, WEIGHTS)


def score(truth: dict, submission: dict, weights: dict[str, int]) -> dict:
    fields = {}
    for path in weights:
        got = _get(submission, path)
        try:
            ok = got is not None and abs(float(got) - _get(truth, path)) <= TOLERANCE
        except (TypeError, ValueError):
            ok = False
        fields[path] = ok

    # Fields whose true value is zero can only cost reward, never earn it;
    # otherwise an all-zero submission scores well on simple tasks.
    earnable = [p for p in weights if _get(truth, p) != 0]
    zero = [p for p in weights if _get(truth, p) == 0]
    earned = sum(weights[p] for p in earnable if fields[p])
    penalty = sum(weights[p] for p in zero if not fields[p])
    value = max(earned - penalty, 0) / sum(weights[p] for p in earnable)
    return {
        "reward": round(value, 4),
        "passed": all(fields.values()),
        "fields": fields,
    }
