import importlib.util
import json
import sys
import threading
import urllib.request
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from gst_env import generate_task, solve
from gst_env.mcp_server import Registry, make_server

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("mcp_episode", SCRIPTS / "mcp_episode.py")
mcp_episode = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mcp_episode)


@pytest.fixture
def server():
    registry = Registry()
    srv = make_server(registry, 0, token="t0k")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield registry, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def rpc(url, method, params=None, token="t0k", mid=1):
    body = {"jsonrpc": "2.0", "method": method, **({"id": mid} if mid is not None else {}), "params": params or {}}
    req = urllib.request.Request(url + "/mcp", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
                                  **({"Authorization": f"Bearer {token}"} if token else {})})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            raw = r.read()
            return r.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        return e.code, None


def call(url, name, **arguments):
    status, reply = rpc(url, "tools/call", {"name": name, "arguments": arguments})
    assert status == 200
    return json.loads(reply["result"]["content"][0]["text"]), reply["result"]["isError"]


def test_any_mcp_client_can_take_the_benchmark(server):
    _, url = server
    status, init = rpc(url, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                            "clientInfo": {"name": "test", "version": "0"}})
    assert status == 200 and init["result"]["protocolVersion"] == "2025-06-18"
    assert rpc(url, "notifications/initialized", mid=None)[0] == 202
    names = [t["name"] for t in rpc(url, "tools/list")[1]["result"]["tools"]]
    assert names[0] == "new_episode" and "submit_return" in names

    started, _ = call(url, "new_episode", difficulty="hard", seed=7)
    episode = started["episode"]
    assert "GSTR-3B" in started["instructions"]
    first = call(url, "list_invoices", episode=episode, kind="sale")[0]["invoice_ids"][0]
    assert call(url, "get_invoice", episode=episode, invoice_id=first)[0]["invoice_id"] == first
    graded, is_error = call(url, "submit_return", episode=episode, gstr3b=solve(generate_task(7, "hard")))
    assert not is_error and graded["passed"] and graded["reward"] == 1.0
    again, is_error = call(url, "submit_return", episode=episode, gstr3b={})
    assert is_error and "already" in again["error"]


def test_auth_and_bad_input(server):
    _, url = server
    assert rpc(url, "tools/list", token=None)[0] == 401
    assert rpc(url, "tools/list", token="wrong")[0] == 401
    assert call(url, "get_credit_ledger", episode="nope") == ({"error": "unknown episode; call new_episode first"}, True)
    assert rpc(url, "tools/call", {"name": "rm_rf"})[1]["error"]["code"] == -32602
    assert rpc(url, "resources/list")[1]["error"]["code"] == -32601


def test_otari_runs_the_loop_over_mcp_and_the_server_grades(server, monkeypatch):
    registry, url = server
    task = generate_task(21, "easy")
    sent = []

    def gateway(**kw):
        """Stands in for Otari's managed MCP loop: connect, list tools, call them."""
        sent.append(kw)
        srv = kw["extra_body"]["mcp_servers"][0]
        base, token = srv["url"].removesuffix("/mcp"), srv["authorization_token"]
        episode = kw["messages"][0]["content"].split("episode id is ")[1].split(";")[0]
        assert rpc(base, "tools/list", token=token)[0] == 200
        rpc(base, "tools/call", {"name": "submit_return", "arguments": {"episode": episode, "gstr3b": solve(task)}},
            token=token)
        usage = NS(cost_usd=0.0021)
        done = NS(choices=[NS(finish_reason="stop", message=NS(content="Filed."))], usage=usage)
        return NS(headers={"Otari-Model": "qwen", "Otari-Request-ID": "m1"}, parse=lambda: done)

    client = NS(chat=NS(completions=NS(with_raw_response=NS(create=gateway))))
    row = mcp_episode.run_mcp_episode(client, "otari:qwen", task, registry, url, "t0k")

    assert row["passed"] and row["transport"] == "mcp" and row["cost_usd"] == 0.0021
    assert "tools" not in sent[0] and sent[0]["extra_body"]["max_tool_iterations"] == 25
    assert row["otari"]["request_ids"] == ["m1"]
