import importlib.util
from pathlib import Path

from gst_env import generate_task, solve

spec = importlib.util.spec_from_file_location("baseline", Path(__file__).parents[1] / "scripts" / "baseline.py")
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


class ScriptedAgent:
    """Lists invoices, reads each one, then submits the true answer."""

    def __init__(self, task, answer, chatter: bool = False):
        self.task, self.answer, self.chatter, self.step = task, answer, chatter, 0

    def start(self, prompt):
        return [prompt]

    def turn(self, messages):
        self.step += 1
        if self.chatter and self.step == 1:
            return baseline.Turn([], "end_turn", 10, 5)  # replies without a tool call
        if self.step <= 2:
            calls = [baseline.ToolCall("l", "list_invoices", {"kind": "sale", "page": 1}),
                     baseline.ToolCall("g", "get_invoice", {"invoice_id": "S-001"}),
                     baseline.ToolCall("bad", "get_invoice", None)]
        else:
            calls = [baseline.ToolCall("s", "submit_return", {"gstr3b": self.answer})]
        return baseline.Turn(calls, "tool_use", 100, 20)

    def add_results(self, messages, results):
        messages.extend(out for _, out in results)

    def nudge(self, messages, text=None):
        messages.append("nudge")


def test_episode_submits_and_scores():
    task = generate_task(5, "hard")
    row = baseline.run_episode(ScriptedAgent(task, solve(task)), "claude-opus-5-5", "high", task, max_turns=10)
    assert row["passed"] and row["reward"] == 1.0
    assert row["tool_calls"] == 7 and row["turns"] == 3
    assert row["cost_usd"] > 0


def test_nudge_then_wrong_answer_and_turn_cap():
    task = generate_task(6, "medium")
    wrong = {**solve(task), "total_cash_payable": -1}
    row = baseline.run_episode(ScriptedAgent(task, wrong, chatter=True), "sarvam-105b", "high", task, max_turns=10)
    assert row["submitted"] and not row["passed"] and "total_cash_payable" in row["failed_fields"]
    assert row["effort"] is None

    capped = baseline.run_episode(ScriptedAgent(task, wrong), "claude-opus-5-5", "high", task, max_turns=2)
    assert not capped["submitted"] and capped["reward"] == 0.0


def test_bad_arguments_return_errors_not_crashes():
    session = baseline.FilingSession(generate_task(1, "easy"))
    assert "error" in baseline.call_tool(session, baseline.ToolCall("x", "get_invoice", {}))
    assert "error" in baseline.call_tool(session, baseline.ToolCall("x", "nope", {}))
    assert "error" in baseline.call_tool(session, baseline.ToolCall("x", "submit_return", None))


def test_openai_agent_parses_function_calls_and_replays_items(monkeypatch):
    from types import SimpleNamespace as NS

    monkeypatch.setenv("OPENAI_API_KEY", "test")
    agent = baseline.make_agent("gpt-6-astra", "high")
    assert isinstance(agent, baseline.OpenAIAgent)

    class Item(NS):
        def model_dump(self, **_):
            return dict(self.__dict__)

    output = [Item(type="reasoning", id="r1"),
              Item(type="function_call", call_id="c1", name="get_invoice", arguments='{"invoice_id": "S-001"}'),
              Item(type="function_call", call_id="c2", name="get_invoice", arguments="{broken")]
    agent.client = NS(responses=NS(create=lambda **_: NS(status="completed", output=output,
                                                         usage=NS(input_tokens=50, output_tokens=7,
                                                                  input_tokens_details=NS(cached_tokens=30, cache_write_tokens=5)))))
    messages = agent.start("file it")
    turn = agent.turn(messages)
    assert [c.args for c in turn.calls] == [{"invoice_id": "S-001"}, None]
    assert (turn.input_tokens, turn.cache_read_tokens, turn.cache_write_tokens, turn.output_tokens) == (15, 30, 5, 7)
    assert [m.get("type") for m in messages[1:]] == ["reasoning", "function_call", "function_call"]
    agent.add_results(messages, [(turn.calls[0], {"ok": 1})])
    assert messages[-1] == {"type": "function_call_output", "call_id": "c1", "output": '{"ok": 1}'}


def test_model_routing(monkeypatch):
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "SARVAM_API_KEY"):
        monkeypatch.setenv(key, "test")
    assert isinstance(baseline.make_agent("sarvam-105b", "high"), baseline.SarvamAgent)
    assert isinstance(baseline.make_agent("claude-opus-5-5", "high"), baseline.ClaudeAgent)


def test_claude_agent_requests_caching_and_cost_counts_cache_rates(monkeypatch):
    from types import SimpleNamespace as NS

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    agent = baseline.make_agent("claude-opus-5-5", "high")
    task = generate_task(2, "easy")
    sent = []

    def create(**kwargs):
        sent.append(kwargs)
        block = NS(type="tool_use", id="t1", name="submit_return", input={"gstr3b": solve(task)})
        usage = NS(input_tokens=1_000, output_tokens=2_000,
                   cache_read_input_tokens=100_000, cache_creation_input_tokens=10_000)
        return NS(stop_reason="tool_use", content=[block], usage=usage)

    agent.client = NS(messages=NS(create=create))
    row = baseline.run_episode(agent, "claude-opus-5-5", "high", task, max_turns=3)

    assert sent[0]["cache_control"] == {"type": "ephemeral"}
    assert row["passed"] and row["cache_read_tokens"] == 100_000
    # 1k input @ $4 + 10k write @ $5 + 100k read @ $0.20 + 2k output @ $20, per million
    assert row["cost_usd"] == round((4_000 + 50_000 + 20_000 + 40_000) / 1e6, 4)
    # Same tokens all billed as plain input
    assert row["uncached_cost_usd"] == round((111_000 * 4 + 2_000 * 20) / 1e6, 4)


def test_usage_splitting_and_pricing_for_openai_and_sarvam():
    from types import SimpleNamespace as NS

    sarvam = baseline.sarvam_usage(NS(prompt_tokens=1_000, completion_tokens=200,
                                      prompt_tokens_details=NS(cached_tokens=600)))
    assert sarvam == {"input_tokens": 400, "output_tokens": 200, "cache_read_tokens": 600}
    assert baseline.sarvam_usage(None) == {"input_tokens": 0, "output_tokens": 0}

    p = baseline.PRICES["sarvam-105b"]
    assert round(p.input * baseline.INR_PER_USD, 6) == 15.0  # ₹15 per million input tokens

    gpt = baseline.PRICES["gpt-6-astra"]
    assert (gpt.input, gpt.output, gpt.cache_read, gpt.cache_write) == (10.0, 50.0, 1.0, 12.5)
    assert baseline.PRICES.get("unknown-model", baseline.UNPRICED).input == 0


def test_stub_reads_everything_and_costs_only_on_paper():
    task = generate_task(9, "hard")
    row = baseline.run_episode(baseline.StubAgent(task, 1000), "claude-opus-5-5", "high", task, max_turns=60)
    assert row["submitted"] and row["reward"] == 0.0
    assert row["tool_calls"] >= len(task.invoices) + 3  # lists, every invoice, ledger, submit
    assert row["cache_read_tokens"] > 0 and row["cost_usd"] > 0


def test_budget_refuses_reservations_past_the_cap():
    budget = baseline.Budget(cap_usd=1.0, estimate_usd=0.4)
    assert budget.reserve() and budget.reserve()
    assert not budget.reserve()  # 0.8 reserved + 0.4 > 1.0
    budget.settle(0.1)
    assert budget.reserve()  # actual spend came in under the estimate
    assert baseline.Budget(None, 99).reserve()


def test_cycle_episode_and_stub_through_runner():
    from gst_env.cycle import generate_cycle_task
    task = generate_cycle_task(7)
    row = baseline.run_episode(baseline.StubAgent(task, 100), "claude-opus-5-5", "high", task, max_turns=20)
    assert row["submitted"] and row["reward"] == 0.0 and row["difficulty"] == "cycle"
    assert row["n_invoices"] == len(task.sales) + len(task.purchases)


def test_summarize_samples_counts_learnable_tasks(capsys):
    rows = [{"difficulty": "hard", "task_id": t, "passed": p}
            for t, ps in {"a": [0, 0, 0, 0], "b": [1, 0, 0, 0], "c": [1, 1, 1, 1]}.items() for p in ps]
    baseline.summarize_samples(rows, ["hard"], 4)
    out = capsys.readouterr().out
    assert "pass@1=42%" in out and "pass@4=67%" in out and "1 learnable, 1 never solved, 1 always solved" in out


def _otari_client(responses, sent):
    """Fake Otari: with_raw_response.create returns headers + a parsed chat completion."""
    from types import SimpleNamespace as NS

    def create(**kwargs):
        sent.append(kwargs)
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        headers, completion = item
        return NS(headers=headers, parse=lambda: completion)

    return NS(chat=NS(completions=NS(with_raw_response=NS(create=create))))


def _completion(calls, cost, finish="tool_calls"):
    from types import SimpleNamespace as NS

    tool_calls = [NS(id=f"c{i}", function=NS(name=n, arguments=a)) for i, (n, a) in enumerate(calls)]
    usage = NS(prompt_tokens=120, completion_tokens=30, prompt_tokens_details=NS(cached_tokens=20), cost_usd=cost)
    return NS(choices=[NS(finish_reason=finish, message=NS(content="", tool_calls=tool_calls))], usage=usage)


def test_otari_agent_files_a_return_and_bills_what_otari_reports(monkeypatch):
    import json

    monkeypatch.setenv("OTARI_API_KEY", "test")
    task = generate_task(3, "easy")
    agent = baseline.make_agent("otari:deepseek/deepseek-v4-pro-0813", "high",
                                user="arm-deepseek", guardrails=["prompt-injection"], code_exec=True)
    assert isinstance(agent, baseline.OtariAgent) and agent.model == "deepseek/deepseek-v4-pro-0813"
    sent = []
    first = {"Otari-Model": "deepseek-v4-pro", "Otari-Fallback": "false", "Otari-Request-ID": "r1"}
    second = {"Otari-Model": "glm-5p3", "Otari-Fallback": "true", "Otari-Request-ID": "r2",
              "Otari-Guardrails": "prompt-injection=pass"}
    agent.client = _otari_client([
        (first, _completion([("list_invoices", '{"kind": "sale", "page": 1}')], 0.0012)),
        (second, _completion([("submit_return", json.dumps({"gstr3b": solve(task)}))], 0.0030)),
    ], sent)

    row = baseline.run_episode(agent, "otari:deepseek/deepseek-v4-pro-0813", "high", task, max_turns=5)

    assert row["passed"] and row["effort"] is None
    assert row["cost_usd"] == 0.0042  # Otari's bill, not the local price table
    assert row["input_tokens"] == 200 and row["cache_read_tokens"] == 40
    assert row["otari"]["served_models"] == ["deepseek-v4-pro", "glm-5p3"]
    assert row["otari"]["fallback_turns"] == 1 and row["otari"]["request_ids"] == ["r1", "r2"]
    assert row["otari"]["guardrails"] == ["prompt-injection=pass"]
    extra = sent[0]["extra_body"]
    assert extra["user"] == "arm-deepseek"
    assert extra["guardrails"] == [{"profile": "prompt-injection", "mode": "block", "on_unavailable": "block"}]
    assert {"type": "otari_code_execution"} in sent[0]["tools"]
    # the tool result went back as a Chat Completions tool message
    assert sent[1]["messages"][2]["role"] == "tool" and sent[1]["messages"][2]["tool_call_id"] == "c0"


def test_otari_budget_refusal_stops_the_episode_cleanly(monkeypatch):
    import httpx
    import openai

    monkeypatch.setenv("OTARI_API_KEY", "test")
    task = generate_task(4, "easy")
    agent = baseline.make_agent("otari:gst-filer", "high")
    assert "extra_body" not in agent.extra and agent.extra == {}
    response = httpx.Response(403, headers={"Otari-Error-Code": "budget_exceeded"},
                              request=httpx.Request("POST", "https://api.otari.ai/api/v1/chat/completions"))
    agent.client = _otari_client([openai.PermissionDeniedError("budget", response=response, body=None)], [])

    row = baseline.run_episode(agent, "otari:gst-filer", "high", task, max_turns=5)

    assert row["stop_reason"] == "budget_exceeded" and not row["submitted"]
    assert row["otari"]["error_code"] == "budget_exceeded" and row["cost_usd"] == 0.0


def test_otari_dry_run_needs_no_price_row():
    assert baseline.agent_class("otari:gst-filer") is baseline.OtariAgent
