"""Rank models from baseline and pipeline result files.

    python scripts/leaderboard.py results/otari-*.jsonl results/pipeline.jsonl \\
        --markdown results/LEADERBOARD.md --json docs/leaderboard.json

One row per filer model: pass rate with a 95% Wilson interval, mean reward, cost per
item and per passing return (as Otari billed it), how often a fallback served, and on
poisoned tasks how often the filer obeyed the injected note or a guardrail refused it.
Rows are sorted by the lower bound of the pass-rate interval, so a lucky 2/2 does not
outrank a steady 40/50.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from gst_env.stats import wilson  # noqa: E402


def load(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def rate(hits: int, n: int) -> float | None:
    return round(hits / n, 3) if n else None


def board(rows: list[dict]) -> list[dict]:
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    out = []
    for model, rs in by_model.items():
        n = len(rs)
        passed = sum(r["passed"] for r in rs)
        lo, hi = wilson(passed, n)
        cost = sum(r.get("total_cost_usd", r["cost_usd"]) for r in rs)
        turns = [t for r in rs if r.get("otari") for t in [r["otari"]]]
        poisoned = [r for r in rs if r.get("injected")]
        read_note = [r for r in poisoned if r.get("obeyed_injection") is not None]
        reviewed = [r for r in rs if r.get("review")]
        out.append({
            "model": model,
            "n": n,
            "pass_rate": rate(passed, n),
            "pass_ci95": [round(lo, 3), round(hi, 3)],
            "mean_reward": round(sum(r["reward"] for r in rs) / n, 3),
            "by_difficulty": {d: rate(sum(r["passed"] for r in rs if r["difficulty"] == d),
                                      sum(r["difficulty"] == d for r in rs))
                              for d in sorted({r["difficulty"] for r in rs})},
            "cost_usd": round(cost, 4),
            "usd_per_item": round(cost / n, 4),
            "usd_per_pass": round(cost / passed, 4) if passed else None,
            "fallback_share": rate(sum(bool(t.get("fallback_turns")) for t in turns), len(turns)),
            "served_models": sorted({m for t in turns for m in t.get("served_models", [])}),
            "budget_stops": sum(r["stop_reason"] == "budget_exceeded" for r in rs),
            "poisoned": len(poisoned),
            "obeyed_injection": rate(sum(r["obeyed_injection"] for r in read_note), len(read_note)),
            "guardrail_blocked": rate(sum(r.get("guardrail_blocked", False) for r in poisoned), len(poisoned)),
            "review_rescued": sum(not r["first_draft_passed"] and r["passed"] for r in reviewed),
            "review_broke": sum(r["first_draft_passed"] and not r["passed"] for r in reviewed),
        })
    return sorted(out, key=lambda b: (-b["pass_ci95"][0], -(b["pass_rate"] or 0), b["usd_per_item"]))


def pct(x: float | None) -> str:
    return "–" if x is None else f"{x:.0%}"


def markdown(entries: list[dict]) -> str:
    lines = ["| # | Model | Passed | 95% CI | Mean reward | $/item | $/pass | Fallback | Obeyed injection | Guardrail blocked |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for i, b in enumerate(entries, 1):
        lo, hi = b["pass_ci95"]
        per_pass = "–" if b["usd_per_pass"] is None else f"${b['usd_per_pass']:.3f}"
        lines.append(f"| {i} | `{b['model']}` | {pct(b['pass_rate'])} of {b['n']} | {lo:.0%}–{hi:.0%} | "
                     f"{b['mean_reward']:.2f} | ${b['usd_per_item']:.3f} | {per_pass} | {pct(b['fallback_share'])} | "
                     f"{pct(b['obeyed_injection'])} | {pct(b['guardrail_blocked'])} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--markdown", type=Path)
    p.add_argument("--json", type=Path)
    args = p.parse_args()
    entries = board(load(args.files))
    table = markdown(entries)
    print(table)
    if args.markdown:
        args.markdown.write_text(table)
    if args.json:
        args.json.write_text(json.dumps(entries, indent=1) + "\n")


if __name__ == "__main__":
    main()
