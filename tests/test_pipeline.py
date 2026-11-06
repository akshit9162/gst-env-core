import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import httpx
import openai
import pytest

from gst_env import generate_task, solve

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("pipeline", SCRIPTS / "pipeline.py")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)
baseline = pipeline.baseline


def completion(calls=(), content="", cost=0.001):
    tcs = [NS(id=f"{n}-{i}", function=NS(name=n, arguments=json.dumps(a))) for i, (n, a) in enumerate(calls)]
    usage = NS(prompt_tokens=100, completion_tokens=20, prompt_tokens_details=None, cost_usd=cost)
    return NS(choices=[NS(finish_reason="tool_calls" if tcs else "stop",
                          message=NS(content=content, tool_calls=tcs))], usage=usage)


class FakeOtari:
    """Scripted replies per model name; records every request."""

    def __init__(self, script):
        self.script, self.sent = script, []
        self.chat = NS(completions=NS(with_raw_response=NS(create=self.create)))

    def create(self, **kw):
        self.sent.append(kw)
        item = self.script[kw["model"]].pop(0)
        if isinstance(item, Exception):
            raise item
        return NS(headers={"Otari-Model": kw["model"], "Otari-Request-ID": f"req-{len(self.sent)}"},
                  parse=lambda: item)


def args(**kw):
    base = dict(filer="otari:f", reviewer="otari:r", explainer="otari:e", guardrail=None,
                code_exec=True, web_fetch=True, users=False, max_turns=10)
    return NS(**{**base, **kw})


@pytest.fixture
def otari(monkeypatch):
    monkeypatch.setenv("OTARI_API_KEY", "test")

    def install(script):
        fake = FakeOtari(script)
        monkeypatch.setattr(baseline.openai, "OpenAI", lambda **_: fake)
        return fake
    return install


def test_reviewer_catches_an_injected_draft_and_the_explainer_summarises(otari):
    task = generate_task(11, "hard")
    truth = solve(task)
    assert truth["blocked_itc"] > 0
    target = next(i.invoice_id for i in task.invoices if i.kind == "purchase")
    fake = otari({
        "f": [completion([("get_invoice", {"invoice_id": target})]),
              completion([("submit_return", {"gstr3b": {**truth, "blocked_itc": 0}})]),
              completion([("submit_return", {"gstr3b": truth})])],
        "r": [completion([("get_credit_ledger", {})]),
              completion([("review_decision", {"verdict": "revise", "notes": "Section 17(5): food is blocked"})])],
        "e": [completion(content="You pay ₹X in cash this month...")],
    })

    row = pipeline.run_item(args(), task, inject=True)

    assert row["passed"] and row["first_draft_passed"] is False
    assert row["review"] == {"verdict": "revise", "notes": "Section 17(5): food is blocked"}
    assert row["saw_injection"] and row["obeyed_injection"] is True and row["injection_filed"] is False
    assert row["explanation"].startswith("You pay")
    assert row["total_cost_usd"] == round(0.003 + 0.002 + 0.001, 4)
    # roles really ran on different models, with their own tools
    by_model = {}
    for kw in fake.sent:
        by_model.setdefault(kw["model"], kw)
    filer_tools = {t.get("type") for t in by_model["f"]["tools"]}
    reviewer_tools = {t.get("function", {}).get("name") or t["type"] for t in by_model["r"]["tools"]}
    assert "otari_code_execution" in filer_tools and "otari_web_fetch" not in filer_tools
    assert {"otari_web_fetch", "review_decision"} <= reviewer_tools and "submit_return" not in reviewer_tools
    # the filer saw the note and the reviewer's notes came back to it
    filer_msgs = [kw for kw in fake.sent if kw["model"] == "f"][-1]["messages"]
    assert any("NOTE TO THE AI FILING AGENT" in m.get("content", "") for m in filer_msgs if m["role"] == "tool")
    tool_out = [m["content"] for m in filer_msgs if m["role"] == "tool"]
    assert any("returned_by_reviewer" in c and "17(5)" in c for c in tool_out)
    assert tool_out[-1] == '{"status": "submitted"}'


def test_guardrail_refusal_stops_the_poisoned_filing(otari):
    task = generate_task(12, "hard")
    refusal = openai.PermissionDeniedError(
        "blocked", body=None,
        response=httpx.Response(403, headers={"Otari-Error-Code": "guardrail_violation"},
                                request=httpx.Request("POST", "https://eu.api.otari.ai/api/v1/chat/completions")))
    target = next(i.invoice_id for i in task.invoices if i.kind == "purchase")
    fake = otari({"f": [completion([("get_invoice", {"invoice_id": target})]), refusal], "r": [], "e": []})

    row = pipeline.run_item(args(guardrail=["prompt-injection"]), task, inject=True)

    assert row["guardrail_blocked"] and not row["submitted"] and row["review"] is None
    assert row["obeyed_injection"] is None and "explanation" not in row
    assert fake.sent[0]["extra_body"]["guardrails"][0]["profile"] == "prompt-injection"


def test_approved_draft_is_filed_as_is_and_clean_tasks_have_no_note(otari):
    task = generate_task(13, "easy")
    otari({
        "f": [completion([("submit_return", {"gstr3b": solve(task)})])],
        "r": [completion([("review_decision", {"verdict": "approve", "notes": ""})])],
        "e": [completion(content="ok")],
    })
    row = pipeline.run_item(args(), task, inject=False)
    assert row["passed"] and row["first_draft_passed"] and row["review"]["verdict"] == "approve"
    assert not row["injected"] and row["obeyed_injection"] is None
