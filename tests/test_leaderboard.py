import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("leaderboard", Path(__file__).parents[1] / "scripts" / "leaderboard.py")
leaderboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(leaderboard)


def row(model, passed, difficulty="easy", cost=0.01, **kw):
    return {"model": model, "passed": passed, "reward": 1.0 if passed else 0.2, "difficulty": difficulty,
            "cost_usd": cost, "stop_reason": "tool_calls", **kw}


def test_ranks_by_interval_lower_bound_and_counts_otari_signals(tmp_path):
    rows = ([row("otari:steady", i < 8, otari={"fallback_turns": int(i == 0), "served_models": ["a"]}) for i in range(10)]
            + [row("otari:lucky", True), row("otari:lucky", True)]
            + [row("otari:poisoned", False, "hard", injected=True, obeyed_injection=True, guardrail_blocked=False),
               row("otari:poisoned", False, "hard", injected=True, obeyed_injection=None, guardrail_blocked=True,
                   stop_reason="guardrail_violation")])
    path = tmp_path / "r.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))

    entries = leaderboard.board(leaderboard.load([path]))

    assert [e["model"] for e in entries] == ["otari:steady", "otari:lucky", "otari:poisoned"]
    steady = entries[0]
    assert steady["pass_rate"] == 0.8 and steady["fallback_share"] == 0.1 and steady["usd_per_pass"] == 0.0125
    poisoned = entries[2]
    assert poisoned["obeyed_injection"] == 1.0 and poisoned["guardrail_blocked"] == 0.5 and poisoned["usd_per_pass"] is None
    assert "| 1 | `otari:steady` | 80% of 10 |" in leaderboard.markdown(entries)
