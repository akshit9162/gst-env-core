"""Build docs/otari/data.json, the data behind the Otari results page, from results/.

Every number on https://akshit9162.github.io/gst-env-core/otari/ comes from this file.

    .venv/bin/python scripts/build_otari_page.py
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).parent.parent
RES = ROOT / "results"
OUT = ROOT / "docs" / "otari" / "data.json"


def rows(name: str) -> list[dict]:
    return [json.loads(line) for line in (RES / name).open() if line.strip()]


def episode(r: dict) -> dict:
    """The parts of a pipeline episode the page shows."""
    o = r.get("otari", {})
    rv = r.get("reviewer_otari") or []
    return {
        "task": r["task_id"], "difficulty": r["difficulty"], "invoices": r.get("n_invoices"),
        "roles": r.get("role_models"), "passed": r["passed"], "reward": r["reward"],
        "first_draft_passed": r.get("first_draft_passed"), "failed": r.get("failed_fields", []),
        "injected": r.get("injected"), "obeyed": r.get("obeyed_injection"),
        "blocked": r.get("guardrail_blocked"), "review": r.get("review"),
        "explanation": r.get("explanation"), "error": o.get("error_code"),
        "code_executions": o.get("code_executions", 0),
        "web_fetches": sum(x.get("web_fetches", 0) for x in rv),
        "cost": r.get("total_cost_usd", r.get("cost_usd")),
        "filer_request_ids": o.get("request_ids", []),
        "reviewer_request_ids": [i for x in rv for i in x.get("request_ids", [])],
        "filed": r.get("filed_return"),
    }


def main() -> None:
    lb = json.loads((RES / "leaderboard.json").read_text())
    lb_ids = {}
    for f in sorted(RES.glob("lb-*.jsonl")):
        rs = rows(f.name)
        lb_ids[rs[0]["model"]] = rs[0]["otari"]["request_ids"][0]
    inj = rows("injection-batch.jsonl")
    per_user = rows("per-user.jsonl")
    mcp = rows("mcp-live.jsonl")
    routing = rows("routing-gst-filer.jsonl")
    direct = rows("guardrail-direct.jsonl")
    fb = next(r for r in rows("lb-openai_gpt-oss-120b.jsonl") if r["otari"]["fallback_turns"])
    data = {
        "leaderboard": [{**m, "request_id": lb_ids.get(m["model"])} for m in lb],
        "injection": {
            "episodes": [episode(r) for r in inj],
            "poisoned": sum(1 for r in inj if r.get("injected")),
            "reached_filer": sum(1 for r in inj if r.get("saw_injection")),
            "blocked": sum(1 for r in inj if r.get("guardrail_blocked")),
            "obeyed": sum(1 for r in inj if r.get("obeyed_injection")),
            "obeyed_and_sent_back": sum(1 for r in inj if r.get("obeyed_injection")
                                        and (r.get("review") or {}).get("verdict") == "revise"),
            "code_executions": sum(r["otari"].get("code_executions", 0) for r in inj),
            "web_fetches": sum(x.get("web_fetches", 0) for r in inj for x in (r.get("reviewer_otari") or [])),
        },
        "per_user": [{"task": r["task_id"], "passed": r["passed"], "error": r["otari"].get("error_code"),
                      "detail": r["otari"].get("error_detail"),
                      "request_id": (r["otari"].get("request_ids") or [None])[0]} for r in per_user],
        "mcp": [{"task": r["task_id"], "passed": r["passed"], "cost": r["cost_usd"],
                 "request_id": r["otari"]["request_ids"][0]} for r in mcp],
        "routing": [{"model": r["model"], "served": r["otari"]["served_models"],
                     "request_id": r["otari"]["request_ids"][0]} for r in routing],
        "fallback": {"task": fb["task_id"], "served": fb["otari"]["served_models"],
                     "fallback_turns": fb["otari"]["fallback_turns"], "request_id": fb["otari"]["request_ids"][0]},
        "guardrail_direct": [{"kind": r["kind"], "prompt": r["prompt"], "status": r["status"],
                              "error": r.get("error_code"),
                              "score": ((r.get("detail") or {}).get("guardrails") or [{}])[0].get("score"),
                              "request_id": r["headers"].get("Otari-Request-ID")
                              or (r.get("detail") or {}).get("Otari-Request-ID"),
                              "at": r["at"]} for r in direct],
    }
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    print(f"wrote {OUT.relative_to(ROOT)}: {len(data['leaderboard'])} models, "
          f"{len(data['injection']['episodes'])} pipeline episodes, {len(direct)} guardrail probes")


if __name__ == "__main__":
    main()
