"""The filing tools as an MCP server (streamable HTTP, JSON responses), standard library only.

    python -m gst_env.mcp_server --port 8808 [--token SECRET]

Any MCP client can take the benchmark: new_episode starts a seeded task and returns its
episode id and instructions; every other tool takes that id. submit_return grades the
return and reports the reward. Episodes live in memory for the life of the process.

Implements the subset of MCP a tool server needs: initialize, ping, tools/list and
tools/call over POST; notifications get 202. GET (server-initiated streams) is refused
with 405, which the spec allows.
"""

import argparse
import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .generator import DIFFICULTIES, generate_task
from .tools import FilingSession

PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
EPISODE = {"type": "string", "description": "Episode id from new_episode."}
HEADS = {"type": "object", "properties": {h: {"type": "integer"} for h in ("igst", "cgst", "sgst")},
         "required": ["igst", "cgst", "sgst"]}
RETURN = {
    "type": "object",
    "properties": {
        "outward_taxable_value": {"type": "integer"}, "nil_exempt_value": {"type": "integer"},
        "outward_tax": HEADS, "rcm_tax": HEADS, "itc_available": HEADS,
        "blocked_itc": {"type": "integer"}, "total_cash_payable": {"type": "integer"},
    },
    "required": ["outward_taxable_value", "nil_exempt_value", "outward_tax", "rcm_tax",
                 "itc_available", "blocked_itc", "total_cash_payable"],
}


def _tool(name, description, props=None, required=()):
    return {"name": name, "description": description,
            "inputSchema": {"type": "object", "properties": {"episode": EPISODE, **(props or {})},
                            "required": ["episode", *required]}}


TOOLS = [
    {"name": "new_episode", "description": "Start a GSTR-3B filing task. Returns the episode id and filing instructions.",
     "inputSchema": {"type": "object", "properties": {
         "difficulty": {"type": "string", "enum": list(DIFFICULTIES)},
         "seed": {"type": "integer"}}, "required": ["difficulty", "seed"]}},
    _tool("list_invoices", "List invoice IDs of one kind for the filing period, 10 per page.",
          {"kind": {"type": "string", "enum": ["sale", "purchase"]}, "page": {"type": "integer"}}, ("kind",)),
    _tool("get_invoice", "Full details of one invoice: counterparty state code, taxable value, rate, "
          "purchase category and reverse-charge flag.", {"invoice_id": {"type": "string"}}, ("invoice_id",)),
    _tool("get_credit_ledger", "Electronic credit ledger balance (IGST, CGST, SGST) brought forward."),
    _tool("describe_category", "What a purchase category code means.", {"category": {"type": "string"}}, ("category",)),
    _tool("submit_return", "Submit the completed GSTR-3B in whole rupees. Once per episode; returns the reward.",
          {"gstr3b": RETURN}, ("gstr3b",)),
]


class Registry:
    """Episode id -> session. Callers may register their own sessions (an injected or
    reviewed one) so a model driving tools over MCP gets the same environment."""

    def __init__(self):
        self.sessions: dict[str, FilingSession] = {}
        self.lock = threading.Lock()

    def add(self, session: FilingSession) -> str:
        episode = secrets.token_urlsafe(9)
        with self.lock:
            self.sessions[episode] = session
        return episode

    def call(self, name: str, args: dict) -> tuple[dict, bool]:
        if name == "new_episode":
            try:
                task = generate_task(int(args["seed"]), args["difficulty"])
            except (KeyError, ValueError, TypeError) as e:
                return {"error": f"bad arguments: {e}"}, True
            session = FilingSession(task)
            return {"episode": self.add(session), "instructions": session.instructions()}, False
        session = self.sessions.get(str(args.get("episode", "")))
        if session is None:
            return {"error": "unknown episode; call new_episode first"}, True
        out = session.call(name, {k: v for k, v in args.items() if k != "episode"})
        if name == "submit_return" and session.result is not None and "error" not in out:
            out = {**out, "reward": session.result["reward"], "passed": session.result["passed"]}
        return out, "error" in out


def handle(registry: Registry, msg: dict) -> dict | None:
    """One JSON-RPC message in, one response out (None for notifications)."""
    if not isinstance(msg, dict) or "method" not in msg:
        return {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "invalid request"}}
    if "id" not in msg:
        return None
    mid, method, params = msg["id"], msg["method"], msg.get("params") or {}
    if method == "initialize":
        asked = params.get("protocolVersion")
        result = {"protocolVersion": asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0],
                  "capabilities": {"tools": {"listChanged": False}},
                  "serverInfo": {"name": "gst-env", "version": "0.1.0"}}
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        name = params.get("name")
        if name not in {t["name"] for t in TOOLS}:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": f"unknown tool {name}"}}
        out, is_error = registry.call(name, params.get("arguments") or {})
        result = {"content": [{"type": "text", "text": json.dumps(out)}], "isError": is_error}
    else:
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}}
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def make_server(registry: Registry, port: int, token: str | None = None, host: str = "127.0.0.1"):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _send(self, code: int, body: dict | list | None = None):
            data = b"" if body is None else json.dumps(body).encode()
            self.send_response(code)
            if body is not None:
                self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._send(405)

        def do_POST(self):
            if token and self.headers.get("Authorization") != f"Bearer {token}":
                return self._send(401, {"error": "unauthorized"})
            try:
                payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"null")
            except json.JSONDecodeError:
                return self._send(400, {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}})
            if isinstance(payload, list):
                replies = [r for r in (handle(registry, m) for m in payload) if r is not None]
                return self._send(200, replies) if replies else self._send(202)
            reply = handle(registry, payload)
            self._send(202) if reply is None else self._send(200, reply)

    return ThreadingHTTPServer((host, port), Handler)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8808)
    p.add_argument("--token", help="require Authorization: Bearer <token>")
    args = p.parse_args()
    server = make_server(Registry(), args.port, args.token, args.host)
    print(f"gst-env MCP server on http://{args.host}:{args.port}/mcp")
    server.serve_forever()


if __name__ == "__main__":
    main()
