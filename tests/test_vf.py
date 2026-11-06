"""The verifiers adapter. Skipped unless the optional `vf` extra is installed."""

import asyncio
import json
from types import SimpleNamespace

import pytest

vf = pytest.importorskip("verifiers.v1")

import gst_env_vf as g  # noqa: E402
from gst_env import solve  # noqa: E402
from gst_env.cycle import solve_cycle  # noqa: E402


def run(coro):
    return asyncio.run(coro)


def tools_for(task):
    toolset = g.GstTask.toolsets(task.config)[0]
    run(toolset.setup_task(task))
    return toolset


def score(task, toolset) -> tuple[float, float, bool]:
    trace = SimpleNamespace(state=toolset.state)
    return run(task.outcome(trace)), run(task.passed(trace)), run(task.filed(trace))


def test_gstr3b_rollout_through_tools_and_reward():
    task = g.GstTaskset(g.GstTasksetConfig(num_tasks=1, difficulties=["hard"])).load()[0]
    tools = tools_for(task)
    assert "invoice_ids" in json.loads(tools.list_invoices("sale"))
    assert score(task, tools) == (0.0, 0.0, False)
    truth = solve(g.make_session(task.data.info).task)
    tools.submit_return(truth)
    assert score(task, tools) == (1.0, 1.0, True)
    assert "error" in json.loads(tools.submit_return(truth))  # replayed state remembers the first submit


def test_cycle_tools_replay_state_between_calls():
    cfg = g.GstTasksetConfig(num_tasks=1, task=g.GstTaskConfig(tier="cycle"))
    task = g.GstTaskset(cfg).load()[0]
    tools = tools_for(task)
    assert "error" in json.loads(tools.gstr3b_prepare())
    tools.gstr1_file([])
    assert json.loads(tools.gstr3b_prepare())["status"] == "prepared"  # GSTR-1 filing survived the replay
    zero = {"igst": 0, "cgst": 0, "sgst": 0}
    tools.gstr3b_file({k: zero for k in ("rcm_tax_3_1_d", "itc_rcm_4a_3", "itc_reversed_blocked_4b_1",
                                          "itc_reversed_other_4b_2", "cash_payable")})
    reward, passed, filed = score(task, tools)
    assert filed and reward == 0.0 and not passed
    assert solve_cycle(g.make_session(task.data.info).task)["total_cash_payable"] >= 0


@pytest.mark.parametrize("tier,expected", [
    ("gstr3b", {"list_invoices", "get_invoice", "get_credit_ledger", "describe_category", "submit_return"}),
    ("cycle", {"get_sales_register", "get_purchase_register", "read_messages", "get_credit_ledger", "ask_client",
               "gstr1_file", "gstr1a_amend", "ims_list", "ims_action", "gstr3b_prepare", "gstr3b_file"}),
])
def test_tools_register_on_an_mcp_server(tier, expected):
    from mcp.server.mcpserver import MCPServer
    mcp = MCPServer("gst")
    g.GstTask.toolsets(g.GstTaskConfig(tier=tier))[0].register(mcp)
    listed = run(mcp.list_tools())
    assert {t.name for t in listed} == expected
    assert all(t.description for t in listed)
