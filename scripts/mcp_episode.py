"""File GSTR-3B returns with Otari running the tool loop over MCP (managed MCP).

    python -m gst_env.mcp_server --port 8808 --token "$MCP_TOKEN" &   # or let this script start it
    cloudflared tunnel --url http://127.0.0.1:8808                     # Otari must reach it over https
    python scripts/mcp_episode.py --model otari:deepseek/deepseek-v4-pro-0813 \\
        --public-url https://<name>.trycloudflare.com --difficulties easy --per-difficulty 3

The env's tools are not declared in the request: Otari connects to the gst-env MCP
server, lists its tools and calls them itself until the model answers. The grade comes
from the server-side session, so nothing the model says in its final text counts.
"""

import argparse
import json
import os
import secrets
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import baseline  # noqa: E402

from gst_env import DIFFICULTIES, FilingSession, generate_task  # noqa: E402
from gst_env.mcp_server import Registry, make_server  # noqa: E402

PROMPT = """You have the gst-env MCP tools. Your episode id is {episode}; pass it to every tool.

{instructions}"""

TOOL_NAMES = ["list_invoices", "get_invoice", "get_credit_ledger", "describe_category", "submit_return"]


def run_mcp_episode(client, model: str, task, registry: Registry, public_url: str, token: str | None,
                    max_iterations: int = 25, max_tokens: int = 8192) -> dict:
    session = FilingSession(task)
    episode = registry.add(session)
    server = {"name": "gstenv", "url": public_url.rstrip("/") + "/mcp", "allowed_tools": TOOL_NAMES,
              "purpose_hint": "GSTR-3B filing tools for one episode: read invoices and the ledger, then submit."}
    if token:
        server["authorization_token"] = token
    stop, cost, gateway = None, 0.0, {}
    try:
        raw = client.chat.completions.with_raw_response.create(
            model=baseline.otari_model(model),
            messages=[{"role": "user", "content": PROMPT.format(episode=episode, instructions=session.instructions())}],
            max_tokens=max_tokens,
            extra_body={"mcp_servers": [server], "max_tool_iterations": max_iterations},
        )
        r = raw.parse()
        gateway = {h: raw.headers[h] for h in baseline.OTARI_HEADERS if h in raw.headers}
        cost = float(getattr(r.usage, "cost_usd", 0) or 0) if r.usage else 0.0
        stop = r.choices[0].finish_reason if r.choices else "no_choices"
    except baseline.openai.APIStatusError as e:
        stop = e.response.headers.get("Otari-Error-Code") or f"api_error_{e.status_code}"
    result = session.result or {"reward": 0.0, "passed": False, "fields": {}}
    return {
        "task_id": task.task_id, "difficulty": task.difficulty, "n_invoices": len(task.invoices),
        "model": model, "transport": "mcp", "reward": result["reward"], "passed": result["passed"],
        "submitted": session.done, "failed_fields": [f for f, ok in result["fields"].items() if not ok],
        "stop_reason": stop, "cost_usd": round(cost, 4), "otari": baseline.otari_summary([gateway] if gateway else []),
    }


def main() -> None:
    baseline.load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--public-url", required=True, help="https URL that reaches the MCP server port")
    p.add_argument("--port", type=int, default=8808)
    p.add_argument("--difficulties", nargs="+", default=["easy"], choices=list(DIFFICULTIES))
    p.add_argument("--per-difficulty", type=int, default=2)
    p.add_argument("--seed-offset", type=int, default=3000)
    p.add_argument("--out", type=Path, default=Path("results/mcp.jsonl"))
    args = p.parse_args()

    token = os.environ.get("MCP_TOKEN") or secrets.token_urlsafe(24)
    registry = Registry()
    server = make_server(registry, args.port, token)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    client = baseline.OtariAgent(args.model).client
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a") as f:
        for d in args.difficulties:
            for i in range(args.per_difficulty):
                row = run_mcp_episode(client, args.model, generate_task(args.seed_offset + i, d),
                                      registry, args.public_url, token)
                f.write(json.dumps(row) + "\n")
                print(f"{row['task_id']:<20} passed={row['passed']!s:<5} reward={row['reward']:.2f} "
                      f"${row['cost_usd']:.4f} stop={row['stop_reason']}")
    server.shutdown()


if __name__ == "__main__":
    main()
