"""Run a model as the filing agent and report rewards.

    python scripts/baseline.py --per-difficulty 3 --out results/opus-5-5.jsonl
    python scripts/baseline.py --model sarvam-105b --per-difficulty 3 --out results/sarvam-105b.jsonl
    python scripts/baseline.py --model gpt-6-astra --per-difficulty 3 --out results/gpt-6-astra.jsonl
    python scripts/baseline.py --dry-run --models claude-opus-5-5 gpt-6-astra sarvam-105b --per-difficulty 30 --cap 25
    python scripts/baseline.py --tier cycle --per-difficulty 20 --samples 16 --cap 25 --out results/cycle-opus.jsonl

--tier cycle runs the monthly GSTR-1 -> IMS -> GSTR-3B workflow (--per-difficulty is
then the number of tasks). --samples k runs each task k times and adds pass@1, pass@k
and the count of tasks that are solved sometimes but not always: the ones RL can learn from.

--dry-run plays a stub agent that reads every invoice and submits zeros, so it
costs nothing, needs no API key, and prints a per-arm cost projection. --cap
stops admitting new episodes once spend plus the next reservation would pass it.

Claude models read ANTHROPIC_API_KEY, Sarvam models SARVAM_API_KEY and OpenAI
models (gpt-*) OPENAI_API_KEY. Models prefixed otari: go through the Otari gateway
on OTARI_API_KEY (OTARI_BASE_URL overrides https://api.otari.ai/api/v1); the rest of
the name is an Otari catalog id or routing policy, and cost is what Otari bills:

    python scripts/baseline.py --model otari:deepseek/deepseek-v4-pro-0813 --per-difficulty 3 --out results/otari-deepseek.jsonl

All can be set in a local .env file.
"""

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import anthropic
import openai
from sarvamai import SarvamAI
from sarvamai.core.api_error import ApiError as SarvamApiError

from gst_env import DIFFICULTIES, FilingSession, generate_task
from gst_env.cycle import TOOLS as CYCLE_TOOLS
from gst_env.cycle import CycleSession, CycleTask, generate_cycle_task
from gst_env.stats import bootstrap_mean, pass_at_k, tasks_for_half_width, wilson

@dataclass(frozen=True)
class Price:
    """USD per million tokens."""
    input: float
    output: float
    cache_read: float
    cache_write: float


# Sarvam bills in rupees; converted at this rate. Update when it drifts.
INR_PER_USD = 88.0


def _inr(input: float, output: float, cache_read: float) -> Price:
    return Price(input / INR_PER_USD, output / INR_PER_USD, cache_read / INR_PER_USD, input / INR_PER_USD)


# List prices as of Oct 2026; check each provider's pricing page before quoting costs.
# Claude cache writes (5-minute TTL) bill at 1.25x input. Sarvam does not bill writes separately.
PRICES = {
    "claude-opus-5-5": Price(4.00, 20.00, 0.20, 5.00),
    "claude-sonnet-5-5": Price(2.00, 10.00, 0.20, 2.50),
    "claude-haiku-5-5": Price(0.10, 0.50, 0.01, 0.125),
    "gpt-6-astra": Price(10.00, 50.00, 1.00, 12.50),
    "sarvam-105b": _inr(15.0, 60.0, 5.0),  # ₹ per million tokens after the 15 Oct 2026 price cut
}
UNPRICED = Price(0, 0, 0, 0)

HEADS = {
    "type": "object",
    "properties": {h: {"type": "integer"} for h in ("igst", "cgst", "sgst")},
    "required": ["igst", "cgst", "sgst"],
    "additionalProperties": False,
}

TOOLS = [
    {
        "name": "list_invoices",
        "description": "List invoice IDs of one kind for the filing period, 10 per page.",
        "input_schema": {
            "type": "object",
            "properties": {"kind": {"type": "string", "enum": ["sale", "purchase"]}, "page": {"type": "integer"}},
            "required": ["kind", "page"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_invoice",
        "description": "Get full details of one invoice: counterparty state code, taxable value, rate, purchase category and reverse-charge flag.",
        "input_schema": {
            "type": "object",
            "properties": {"invoice_id": {"type": "string"}},
            "required": ["invoice_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_credit_ledger",
        "description": "Get the electronic credit ledger balance (IGST, CGST, SGST) brought forward into this period.",
        "input_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    },
    {
        "name": "describe_category",
        "description": "Describe what a purchase category code means.",
        "input_schema": {
            "type": "object",
            "properties": {"category": {"type": "string"}},
            "required": ["category"],
            "additionalProperties": False,
        },
    },
    {
        "name": "submit_return",
        "description": "Submit the completed GSTR-3B. All amounts in whole rupees. Can only be called once and ends the task.",
        "input_schema": {
            "type": "object",
            "properties": {
                "gstr3b": {
                    "type": "object",
                    "properties": {
                        "outward_taxable_value": {"type": "integer"},
                        "nil_exempt_value": {"type": "integer"},
                        "outward_tax": HEADS,
                        "rcm_tax": HEADS,
                        "itc_available": HEADS,
                        "blocked_itc": {"type": "integer"},
                        "total_cash_payable": {"type": "integer"},
                    },
                    "required": [
                        "outward_taxable_value", "nil_exempt_value", "outward_tax", "rcm_tax",
                        "itc_available", "blocked_itc", "total_cash_payable",
                    ],
                    "additionalProperties": False,
                }
            },
            "required": ["gstr3b"],
            "additionalProperties": False,
        },
    },
]

NUDGE = "You have not submitted yet. Call submit_return with your answer."
CYCLE_NUDGE = "You have not filed GSTR-3B yet. Finish the cycle and call gstr3b_file."

# Reasoning models spend output tokens thinking; a smaller budget truncates the answer.
REASONING_FLOOR = 8192


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict | None  # None when the model sent arguments that are not valid JSON


@dataclass
class Turn:
    calls: list[ToolCall]
    stop: str
    input_tokens: int  # uncached input only
    output_tokens: int
    fatal: bool = False  # stop the episode (API error, refusal, truncation)
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    gateway_cost_usd: float | None = None  # what Otari billed for this turn, when it says
    gateway: dict | None = None  # Otari response headers worth keeping


def sarvam_usage(u) -> dict:
    """prompt_tokens includes cached tokens; split them out."""
    if not u:
        return dict(input_tokens=0, output_tokens=0)
    cached = (u.prompt_tokens_details.cached_tokens or 0) if u.prompt_tokens_details else 0
    return dict(input_tokens=u.prompt_tokens - cached, output_tokens=u.completion_tokens, cache_read_tokens=cached)


def openai_usage(u) -> dict:
    """input_tokens includes cache reads and writes; split them out."""
    if not u:
        return dict(input_tokens=0, output_tokens=0)
    d = u.input_tokens_details
    read = (d.cached_tokens or 0) if d else 0
    write = (getattr(d, "cache_write_tokens", 0) or 0) if d else 0
    return dict(input_tokens=u.input_tokens - read - write, output_tokens=u.output_tokens,
                cache_read_tokens=read, cache_write_tokens=write)


class ClaudeAgent:
    max_tokens = 16000

    def __init__(self, model: str, effort: str, tools: list = TOOLS):
        self.client = anthropic.Anthropic()
        self.model, self.effort = model, effort
        self.tools = [{**t, "strict": True} for t in tools]

    def start(self, prompt: str) -> list:
        return [{"role": "user", "content": prompt}]

    def turn(self, messages: list) -> Turn:
        try:
            r = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                tools=self.tools,
                output_config={"effort": self.effort},
                # The conversation only ever grows by appending, so each turn
                # re-reads the previous prefix from cache.
                cache_control={"type": "ephemeral"},
                messages=messages,
            )
        except anthropic.APIStatusError as e:
            return Turn([], f"api_error_{e.status_code}", 0, 0, fatal=True)
        usage = dict(
            input_tokens=r.usage.input_tokens,
            output_tokens=r.usage.output_tokens,
            cache_read_tokens=r.usage.cache_read_input_tokens or 0,
            cache_write_tokens=r.usage.cache_creation_input_tokens or 0,
        )
        if r.stop_reason in ("refusal", "max_tokens"):
            return Turn([], r.stop_reason, fatal=True, **usage)
        messages.append({"role": "assistant", "content": r.content})
        calls = [ToolCall(b.id, b.name, b.input) for b in r.content if b.type == "tool_use"]
        return Turn(calls, r.stop_reason, **usage)

    def add_results(self, messages: list, results: list[tuple[ToolCall, dict]]) -> None:
        messages.append({"role": "user", "content": [
            {
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": json.dumps(out),
                **({"is_error": True} if "error" in out else {}),
            }
            for call, out in results
        ]})

    def nudge(self, messages: list, text: str = NUDGE) -> None:
        messages.append({"role": "user", "content": text})


class SarvamAgent:
    max_tokens = 8192

    def __init__(self, model: str, tools: list = TOOLS):
        self.client = SarvamAI(api_subscription_key=os.environ.get("SARVAM_API_KEY"))
        self.model = model
        self.tools = [
            {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
            for t in tools
        ]

    def start(self, prompt: str) -> list:
        return [{"role": "user", "content": prompt}]

    def turn(self, messages: list) -> Turn:
        try:
            r = self.client.chat.completions(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                max_tokens=self.max_tokens,
            )
        except SarvamApiError as e:
            return Turn([], f"api_error_{e.status_code}", 0, 0, fatal=True)
        usage = sarvam_usage(r.usage)
        choice = r.choices[0]
        if choice.finish_reason == "length":
            return Turn([], "length", fatal=True, **usage)
        tool_calls = choice.message.tool_calls or []
        messages.append({
            "role": "assistant",
            "content": choice.message.content or "",
            **({"tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in tool_calls
            ]} if tool_calls else {}),
        })
        calls = []
        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = None
            calls.append(ToolCall(tc.id, tc.function.name, args))
        return Turn(calls, choice.finish_reason, **usage)

    def add_results(self, messages: list, results: list[tuple[ToolCall, dict]]) -> None:
        for call, out in results:
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(out)})

    def nudge(self, messages: list, text: str = NUDGE) -> None:
        messages.append({"role": "user", "content": text})


class OpenAIAgent:
    """Responses API; every output item (including reasoning) is replayed each turn."""

    max_tokens = 16000

    def __init__(self, model: str, effort: str, tools: list = TOOLS):
        self.client = openai.OpenAI()
        self.model, self.effort = model, effort
        self.tools = [
            {"type": "function", "name": t["name"], "description": t["description"], "parameters": t["input_schema"], "strict": True}
            for t in tools
        ]

    def start(self, prompt: str) -> list:
        return [{"role": "user", "content": prompt}]

    def turn(self, messages: list) -> Turn:
        try:
            r = self.client.responses.create(
                model=self.model,
                input=messages,
                tools=self.tools,
                reasoning={"effort": self.effort},
                max_output_tokens=self.max_tokens,
            )
        except openai.APIStatusError as e:
            return Turn([], f"api_error_{e.status_code}", 0, 0, fatal=True)
        usage = openai_usage(r.usage)
        if r.status != "completed":
            return Turn([], r.status, fatal=True, **usage)
        messages.extend(item.model_dump(exclude_none=True, by_alias=True) for item in r.output)
        calls = []
        for item in r.output:
            if item.type != "function_call":
                continue
            try:
                args = json.loads(item.arguments)
            except json.JSONDecodeError:
                args = None
            calls.append(ToolCall(item.call_id, item.name, args))
        return Turn(calls, "tool_use" if calls else "end_turn", **usage)

    def add_results(self, messages: list, results: list[tuple[ToolCall, dict]]) -> None:
        for call, out in results:
            messages.append({"type": "function_call_output", "call_id": call.id, "output": json.dumps(out)})

    def nudge(self, messages: list, text: str = NUDGE) -> None:
        messages.append({"role": "user", "content": text})


OTARI_BASE_URL = "https://api.otari.ai/api/v1"
OTARI_HEADERS = ("Otari-Provider", "Otari-Model", "Otari-Fallback", "Otari-Attempt-Count",
                 "Otari-Guardrails", "Otari-Request-ID")


CODE_EXEC_HINT = ("\n\nYou can run Python in a sandbox with the otari_code_execution tool. "
                  "Use it for the tax arithmetic instead of adding up figures in your head.")
WEB_FETCH_HINT = ("\n\nYou can read a web page with the otari_web_fetch tool. When a decision rests on a GST "
                  "rule, fetch the rule's text from cbicgst.gov.in or cbic-gst.gov.in and cite the section.")


def otari_model(model: str) -> str:
    """'otari:deepseek/deepseek-v4-pro-0813' -> the name Otari takes (catalog id or routing policy)."""
    return model.removeprefix("otari:")


def otari_error(e: "openai.APIStatusError") -> tuple[str | None, dict]:
    """Otari's refusal code and details. The code comes in the Otari-Error-Code header on some
    refusals and only in the body (detail.code) on others, e.g. guardrail_violation."""
    code = e.response.headers.get("Otari-Error-Code")
    info = {}
    if rid := e.response.headers.get("Otari-Request-ID"):
        info["Otari-Request-ID"] = rid
    try:
        detail = e.response.json().get("detail")
    except (ValueError, AttributeError):
        detail = None
    if isinstance(detail, dict):
        code = code or detail.get("code")
        if detail.get("guardrails"):
            info["guardrails"] = [{"profile": g.get("profile"), "score": g.get("score")} for g in detail["guardrails"]]
    elif isinstance(detail, str):
        info["detail"] = detail[:300]
    return code, info


def collect_stream(parsed):
    """Fold a streamed chat completion back into the non-streamed shape turn() reads:
    .choices[0].message.{content,tool_calls}, .choices[0].finish_reason, .usage and
    .otari_tool_calls (one entry per finished gateway tool run). A non-stream object
    (anything with .choices) is returned as is."""
    from types import SimpleNamespace as NS

    if hasattr(parsed, "choices"):
        return parsed
    content, calls, finish, usage, ran = [], {}, None, None, {}
    for chunk in parsed:
        for run in getattr(chunk, "otari_tool_calls", None) or []:
            run = run if isinstance(run, dict) else dict(run)
            if run.get("status") != "in_progress":
                ran[run.get("id") or len(ran)] = run
        if getattr(chunk, "usage", None):
            usage = chunk.usage
        for choice in chunk.choices or []:
            finish = choice.finish_reason or finish
            delta = choice.delta
            if delta is None:
                continue
            if delta.content:
                content.append(delta.content)
            for tc in delta.tool_calls or []:
                slot = calls.setdefault(tc.index, {"id": None, "name": "", "arguments": []})
                slot["id"] = tc.id or slot["id"]
                if tc.function:
                    slot["name"] = tc.function.name or slot["name"]
                    slot["arguments"].append(tc.function.arguments or "")
    tool_calls = [NS(id=c["id"], function=NS(name=c["name"], arguments="".join(c["arguments"]) or "{}"))
                  for _, c in sorted(calls.items()) if c["name"]]
    choices = [NS(finish_reason=finish, message=NS(content="".join(content), tool_calls=tool_calls))] if finish or tool_calls or content else []
    return NS(choices=choices, usage=usage, otari_tool_calls=list(ran.values()))


def infra_stop(stop: str | None) -> bool:
    """Stops caused by the gateway, not the model: timeouts, resets, 5xx, upstream rate limits."""
    return stop in ("timeout", "connection_error", "upstream_rate_limited", "api_error_429") or \
        bool(stop and stop.startswith("api_error_5"))


class OtariAgent:
    """Chat Completions through the Otari gateway. Function tools run here, in the env;
    Otari's own tools (code execution) run in the gateway inside the same request.
    Cost is what Otari reports in usage.cost_usd, not a local price table."""

    max_tokens = 8192

    def __init__(self, model: str, tools: list = TOOLS, user: str | None = None,
                 guardrails: list[str] | None = None, code_exec: bool = False, web_fetch: bool = False):
        self.client = openai.OpenAI(base_url=os.environ.get("OTARI_BASE_URL", OTARI_BASE_URL),
                                    api_key=os.environ.get("OTARI_API_KEY"))
        self.model = otari_model(model)
        self.tools = [
            {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
            for t in tools
        ] + ([{"type": "otari_code_execution"}] if code_exec else []) \
          + ([{"type": "otari_web_fetch"}] if web_fetch else [])
        self.hints = (CODE_EXEC_HINT if code_exec else "") + (WEB_FETCH_HINT if web_fetch else "")
        self.extra = {}
        if user:
            self.extra["user"] = user  # needs a service key: one end user, one budget, per arm
        if guardrails:
            self.extra["guardrails"] = [{"profile": p, "mode": "block", "on_unavailable": "block"} for p in guardrails]

    # The task goes in the system message: Otari's prompt-injection guardrail (Lakera) reads a
    # user message full of rules for an AI as an attack and refuses the whole episode.
    START = "Begin. Read the data with the tools, then submit."

    def start(self, prompt: str) -> list:
        return [{"role": "system", "content": prompt + self.hints}, {"role": "user", "content": self.START}]

    RETRIES = 3  # gateway timeouts, resets, 5xx and upstream 429 come and go; retry before giving up
    RETRY_WAIT = 5.0  # seconds, doubled each retry

    def turn(self, messages: list) -> Turn:
        for attempt in range(self.RETRIES + 1):
            t = self._turn(messages)
            if not infra_stop(t.stop) or attempt == self.RETRIES:
                return t
            time.sleep(self.RETRY_WAIT * 2 ** attempt)

    def _turn(self, messages: list) -> Turn:
        try:
            # Streamed: hosted Otari's CDN cuts a non-streamed request at ~60 s with a 504
            # (mozilla-ai/otari#2692), and a hard task's turn with code execution runs longer.
            raw = self.client.chat.completions.with_raw_response.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                max_tokens=self.max_tokens,
                extra_body=self.extra or None,
                stream=True,
                stream_options={"include_usage": True},
            )
            r = collect_stream(raw.parse())
        except openai.APIStatusError as e:
            code, info = otari_error(e)
            # budget_exceeded and guardrail_violation are Otari doing its job, not a crash
            return Turn([], code or f"api_error_{e.status_code}", 0, 0, fatal=True,
                        gateway={"error_code": code, "status": e.status_code, **info})
        except openai.APIConnectionError as e:  # includes timeouts; one stuck request must not end a batch
            kind = "timeout" if isinstance(e, openai.APITimeoutError) else "connection_error"
            return Turn([], kind, 0, 0, fatal=True, gateway={"error_code": kind})
        gateway = {h: raw.headers[h] for h in OTARI_HEADERS if h in raw.headers}
        # Otari's own tools (sandbox, fetch) run inside the request and are reported here
        ran = getattr(r, "otari_tool_calls", None) or []
        if ran:
            gateway["tool_calls"] = [c.get("type") if isinstance(c, dict) else getattr(c, "type", None) for c in ran]
        usage = sarvam_usage(r.usage)  # same OpenAI Chat Completions usage shape
        cost = getattr(r.usage, "cost_usd", None) if r.usage else None
        meta = dict(gateway_cost_usd=float(cost) if cost is not None else None, gateway=gateway)
        if not r.choices:
            return Turn([], "no_choices", fatal=True, **usage, **meta)
        choice = r.choices[0]
        if choice.finish_reason == "length":
            return Turn([], "length", fatal=True, **usage, **meta)
        tool_calls = [tc for tc in (choice.message.tool_calls or []) if getattr(tc, "function", None)]
        messages.append({
            "role": "assistant",
            "content": choice.message.content or "",
            **({"tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in tool_calls
            ]} if tool_calls else {}),
        })
        calls = []
        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = None
            calls.append(ToolCall(tc.id, tc.function.name, args))
        return Turn(calls, choice.finish_reason, **usage, **meta)

    def add_results(self, messages: list, results: list[tuple[ToolCall, dict]]) -> None:
        for call, out in results:
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(out)})

    def nudge(self, messages: list, text: str = NUDGE) -> None:
        messages.append({"role": "user", "content": text})


class StubAgent:
    """Zero-cost stand-in for the dry run. Reads the task the way a careful agent
    would (every page, every invoice, ten per turn, then the ledger) and submits
    zeros. Token counts are estimated at 4 characters per token, with each turn
    re-reading the earlier prefix from cache and writing the new part to it."""

    CHARS_PER_TOKEN = 4

    def __init__(self, task, output_tokens_per_turn: int):
        self.task, self.out = task, output_tokens_per_turn
        tools = CYCLE_TOOLS if isinstance(task, CycleTask) else TOOLS
        self.prefix = len(json.dumps(tools)) // self.CHARS_PER_TOKEN  # tool definitions are part of every prompt
        self.step = 0
        zero = {h: 0 for h in ("igst", "cgst", "sgst")}
        if isinstance(task, CycleTask):
            self.plan = [[ToolCall(n, n, {})] for n in (
                "get_sales_register", "get_purchase_register", "read_messages", "get_credit_ledger", "ims_list")]
            self.plan += [[ToolCall("g1", "gstr1_file", {"entries": []})], [ToolCall("p", "gstr3b_prepare", {})],
                          [ToolCall("f", "gstr3b_file", {"gstr3b": {k: zero for k in (
                              "rcm_tax_3_1_d", "itc_rcm_4a_3", "itc_reversed_blocked_4b_1",
                              "itc_reversed_other_4b_2", "cash_payable")}})]]
            return
        session = FilingSession(task)
        self.plan = [[ToolCall("l", "list_invoices", {"kind": k, "page": p})]
                     for k in ("sale", "purchase")
                     for p in range(1, session.list_invoices(k)["total_pages"] + 1)]
        ids = [i.invoice_id for i in task.invoices]
        self.plan += [[ToolCall(i, "get_invoice", {"invoice_id": i}) for i in ids[n:n + 10]] for n in range(0, len(ids), 10)]
        self.plan.append([ToolCall("c", "get_credit_ledger", {})])
        self.plan.append([ToolCall("s", "submit_return", {"gstr3b": {
            "outward_taxable_value": 0, "nil_exempt_value": 0, "outward_tax": zero, "rcm_tax": zero,
            "itc_available": zero, "blocked_itc": 0, "total_cash_payable": 0}})])

    def start(self, prompt: str) -> list:
        return [prompt]

    def turn(self, messages: list) -> Turn:
        total = self.prefix + len(json.dumps(messages)) // self.CHARS_PER_TOKEN
        cached = self.prefix if self.step else 0
        calls = self.plan[min(self.step, len(self.plan) - 1)]
        self.step += 1
        self.prefix = total + self.out  # this turn's output is next turn's cached prefix
        return Turn(calls, "tool_use", 0, self.out, cache_read_tokens=cached, cache_write_tokens=total - cached)

    def add_results(self, messages: list, results: list[tuple[ToolCall, dict]]) -> None:
        messages.extend(out for _, out in results)

    def nudge(self, messages: list, text: str = NUDGE) -> None:
        messages.append(text)


class Budget:
    """Spend ledger shared by the worker threads. An episode is admitted only
    while spent + reserved + its estimate stays within the cap; a refused
    reservation is final for that episode."""

    def __init__(self, cap_usd: float | None, estimate_usd: float):
        self.cap, self.estimate = cap_usd, estimate_usd
        self.spent = self.reserved = 0.0
        self.lock = threading.Lock()

    def reserve(self) -> bool:
        with self.lock:
            if self.cap is not None and self.spent + self.reserved + self.estimate > self.cap:
                return False
            self.reserved += self.estimate
            return True

    def settle(self, actual_usd: float) -> None:
        with self.lock:
            self.reserved -= self.estimate
            self.spent += actual_usd


def make_agent(model: str, effort: str, tools: list = TOOLS, **otari):
    if model.startswith("otari:"):
        return OtariAgent(model, tools, **otari)
    if model.startswith("sarvam"):
        return SarvamAgent(model, tools)
    if model.startswith("gpt"):
        return OpenAIAgent(model, effort, tools)
    return ClaudeAgent(model, effort, tools)


def call_tool(session, call: ToolCall) -> dict:
    return session.call(call.name, call.args)


def make_session(task):
    return CycleSession(task) if isinstance(task, CycleTask) else FilingSession(task)


def run_episode(agent, model: str, effort: str, task, max_turns: int, session=None) -> dict:
    session = session or make_session(task)
    messages = agent.start(session.instructions())
    input_tokens = output_tokens = cache_read = cache_write = turns = tool_calls = 0
    stop = None
    gateway_cost, gateway_turns = 0.0, []

    while not session.done and turns < max_turns:
        turns += 1
        t = agent.turn(messages)
        input_tokens += t.input_tokens
        output_tokens += t.output_tokens
        cache_read += t.cache_read_tokens
        cache_write += t.cache_write_tokens
        stop = t.stop
        if t.gateway_cost_usd is not None:
            gateway_cost += t.gateway_cost_usd
        if t.gateway:
            gateway_turns.append(t.gateway)
        if t.fatal:
            break
        if not t.calls:
            agent.nudge(messages, CYCLE_NUDGE if isinstance(task, CycleTask) else NUDGE)
            continue
        tool_calls += len(t.calls)
        agent.add_results(messages, [(c, call_tool(session, c)) for c in t.calls])

    p = PRICES.get(model, UNPRICED)
    cost = (input_tokens * p.input + cache_write * p.cache_write
            + cache_read * p.cache_read + output_tokens * p.output) / 1e6
    # What the same episode would have cost with every input token billed at full price.
    uncached_cost = ((input_tokens + cache_write + cache_read) * p.input + output_tokens * p.output) / 1e6
    via_otari = model.startswith("otari:")
    if via_otari:
        # Otari prices the model it actually routed to; fallbacks included.
        cost = uncached_cost = gateway_cost
    result = session.result or {"reward": 0.0, "passed": False, "fields": {}}
    row = {
        "task_id": task.task_id,
        "difficulty": task.difficulty,
        "n_invoices": len(task.invoices),
        "model": model,
        "effort": None if model.startswith(("sarvam", "otari:")) else effort,
        "reward": result["reward"],
        "passed": result["passed"],
        "submitted": session.done,
        "failed_fields": [f for f, ok in result["fields"].items() if not ok],
        "turns": turns,
        "tool_calls": tool_calls,
        "stop_reason": stop,
        "infra_error": not session.done and infra_stop(stop),
        "input_tokens": input_tokens,
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write,
        "output_tokens": output_tokens,
        "cost_usd": round(cost, 4),
        "uncached_cost_usd": round(uncached_cost, 4),
    }
    if via_otari:
        row["otari"] = otari_summary(gateway_turns)
    return row


def otari_summary(turns: list[dict]) -> dict:
    """Which models actually answered, how often a fallback served, what guardrails said."""
    served = sorted({t["Otari-Model"] for t in turns if "Otari-Model" in t})
    return {
        "served_models": served,
        "fallback_turns": sum(t.get("Otari-Fallback") == "true" for t in turns),
        "code_executions": sum(c == "code_execution" for t in turns for c in t.get("tool_calls", [])),
        "web_fetches": sum(c in ("web_fetch", "fetch") for t in turns for c in t.get("tool_calls", [])),
        "guardrails": [t["Otari-Guardrails"] for t in turns if "Otari-Guardrails" in t],
        "guardrail_scores": [g for t in turns for g in t.get("guardrails", [])],
        "error_code": next((t["error_code"] for t in turns if t.get("error_code")), None),
        "error_detail": next((t["detail"] for t in turns if t.get("detail")), None),
        "request_ids": [t["Otari-Request-ID"] for t in turns if "Otari-Request-ID" in t],
    }


def load_dotenv(path: Path = Path(".env")) -> None:
    """Fill unset or empty env vars from a local .env (KEY=value lines)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and not key.startswith("#") and not os.environ.get(key):
            os.environ[key] = value.strip().strip("'\"")


def agent_class(model: str):
    if model.startswith("otari:"):
        return OtariAgent
    if model.startswith("sarvam"):
        return SarvamAgent
    if model.startswith("gpt"):
        return OpenAIAgent
    return ClaudeAgent


def stub_rows(model: str, effort: str, tasks: list, max_turns: int, output_tokens: int) -> list[dict]:
    return [run_episode(StubAgent(t, output_tokens), model, effort, t, max_turns) for t in tasks]


def summarize(rows: list[dict], difficulties: list[str]) -> None:
    """Pass rates with 95% Wilson intervals and mean rewards with bootstrap intervals."""
    for d in difficulties:
        sub = [r for r in rows if r["difficulty"] == d]
        if not sub:
            continue
        n, k = len(sub), sum(r["passed"] for r in sub)
        lo, hi = wilson(k, n)
        rewards = [r["reward"] for r in sub]
        rlo, rhi = bootstrap_mean(rewards)
        print(f"  {d:<7} pass={k}/{n} ({k / n:.0%}, 95% CI {lo:.0%}-{hi:.0%})  "
              f"mean_reward={sum(rewards) / n:.3f} (95% CI {rlo:.3f}-{rhi:.3f})")
    smallest = min((sum(r["difficulty"] == d for r in rows) for d in difficulties if any(r["difficulty"] == d for r in rows)), default=0)
    if smallest and smallest < tasks_for_half_width(0.15):
        print(f"  note: {smallest} tasks per level; pass-rate intervals are wider than +/-15 points. "
              f"{tasks_for_half_width(0.15)} per level gets them under that, {tasks_for_half_width(0.10)} under +/-10.")


def summarize_samples(rows: list[dict], difficulties: list[str], k: int) -> None:
    """With k samples per task: pass@1, pass@k, and how many tasks give a learning
    signal (solved sometimes, not always). A task never solved in k tries teaches nothing."""
    lost = sum(bool(r.get("infra_error")) for r in rows)
    rows = [r for r in rows if not r.get("infra_error")]
    if lost:
        print(f"  {lost} attempts lost to gateway errors, left out of pass@k")
    for d in difficulties:
        by_task: dict[str, list[bool]] = {}
        for r in rows:
            if r["difficulty"] == d:
                by_task.setdefault(r["task_id"], []).append(r["passed"])
        if not by_task:
            continue
        n = len(by_task)
        p1 = sum(pass_at_k(len(v), sum(v), 1) for v in by_task.values()) / n
        pk = sum(pass_at_k(len(v), sum(v), min(k, len(v))) for v in by_task.values()) / n
        never = sum(not any(v) for v in by_task.values())
        always = sum(all(v) for v in by_task.values())
        print(f"  {d:<7} pass@1={p1:.0%}  pass@{k}={pk:.0%}  over {n} tasks: "
              f"{n - never - always} learnable, {never} never solved, {always} always solved")


def dry_run(args, tasks: list) -> None:
    """Zero-cost pipeline check and cost projection, one block per arm."""
    ok = True
    print(f"CHECK: retry-prompt  {NUDGE!r}")
    for model in args.models:
        price_ok = model in PRICES or model.startswith("otari:")
        budget = agent_class(model).max_tokens
        rows = stub_rows(model, args.effort, tasks, args.max_turns, args.est_output_tokens)
        costs = [r["cost_usd"] for r in rows]
        mean = sum(costs) / len(costs)
        sd = (sum((c - mean) ** 2 for c in costs) / max(len(costs) - 1, 1)) ** 0.5
        pipeline_ok = all(r["submitted"] and r["reward"] == 0.0 for r in rows)
        fits = args.cap is None or max(costs) <= args.cap
        ok &= price_ok and budget >= REASONING_FLOOR and pipeline_ok and fits
        print(f"ARM: {model}")
        if model.startswith("otari:"):
            print("  CHECK: price-table           n/a (Otari bills it; set a budget on the key: the local cap cannot see this spend)")
        else:
            print(f"  CHECK: price-table           {'ok' if price_ok else 'MISSING (add a PRICES row)'}")
        print(f"  CHECK: output-budget         {'ok' if budget >= REASONING_FLOOR else 'LOW'} ({budget} >= {REASONING_FLOOR})")
        print(f"  CHECK: pipeline              {'ok' if pipeline_ok else 'FAIL'} ({len(rows)} stub episodes submitted, all-zero return scored 0)")
        if args.cap is not None:
            print(f"  CHECK: reservation-fits-cap  {'ok' if fits else 'FAIL'} (largest item ${max(costs):.3f}, cap ${args.cap:.2f})")
        for d in args.difficulties:
            sub = [r["cost_usd"] for r in rows if r["difficulty"] == d]
            if sub:
                print(f"  {d:<7} est ${sum(sub) / len(sub):.3f}/item  turns={sum(r['turns'] for r in rows if r['difficulty'] == d) / len(sub):.1f}")
        verdict = "" if args.cap is None else f"  verdict={'within' if sum(costs) <= args.cap else 'exceeds'} cap"
        print(f"  PROJECTION: {len(rows)} items  mean ${mean:.3f}/item (sd {sd:.3f})  total ${sum(costs):.2f}{verdict}")
    print(f"\nEstimate assumes {args.est_output_tokens} output tokens per turn and 4 chars per token; "
          "real spend reconciles from provider usage. Treat it as +/-50%.")
    print(f"DRY_RUN_RESULT: {'pass' if ok else 'fail'}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="claude-opus-5-5")
    parser.add_argument("--models", nargs="+", help="arms to project in --dry-run (default: --model)")
    parser.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--tier", default="gstr3b", choices=["gstr3b", "cycle"],
                        help="gstr3b: single-return tasks by difficulty; cycle: the GSTR-1 -> IMS -> GSTR-3B month")
    parser.add_argument("--difficulties", nargs="+", default=list(DIFFICULTIES), help="gstr3b tier only")
    parser.add_argument("--samples", type=int, default=1, help="attempts per task; >1 reports pass@1 and pass@k")
    parser.add_argument("--per-difficulty", type=int, default=3)
    parser.add_argument("--seed-offset", type=int, default=1000, help="keeps baseline seeds apart from exported tasks")
    parser.add_argument("--max-turns", type=int, default=60)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("results/baseline.jsonl"))
    parser.add_argument("--dry-run", action="store_true", help="stub agent, no API calls, prints a cost projection")
    parser.add_argument("--cap", type=float, help="USD spend cap for this run")
    parser.add_argument("--otari-user", help="end-user id sent with every request (service key: per-arm budget)")
    parser.add_argument("--otari-guardrail", action="append", help="guardrail profile to run in block mode; repeatable")
    parser.add_argument("--otari-code-exec", action="store_true", help="offer Otari's sandboxed code execution tool")
    parser.add_argument("--otari-web-fetch", action="store_true", help="offer Otari's web fetch tool")
    parser.add_argument("--est-output-tokens", type=int, default=1500, help="output tokens per turn assumed by the estimate")
    args = parser.parse_args()
    args.models = args.models or [args.model]

    if args.tier == "cycle":
        args.difficulties = ["cycle"]
        tasks = [generate_cycle_task(args.seed_offset + i) for i in range(args.per_difficulty)]
    else:
        tasks = [
            generate_task(args.seed_offset + i, d)
            for d in args.difficulties
            for i in range(args.per_difficulty)
        ]
    tasks = [t for t in tasks for _ in range(args.samples)]
    if args.dry_run:
        dry_run(args, tasks)
        return

    otari = dict(user=args.otari_user, guardrails=args.otari_guardrail, code_exec=args.otari_code_exec,
                 web_fetch=args.otari_web_fetch) \
        if args.model.startswith("otari:") else {}
    agent = make_agent(args.model, args.effort, CYCLE_TOOLS if args.tier == "cycle" else TOOLS, **otari)
    # Each reservation is the stub's worst-case estimate for one item.
    estimate = max(r["cost_usd"] for r in stub_rows(args.model, args.effort, tasks, args.max_turns, args.est_output_tokens))
    budget = Budget(args.cap, estimate)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def admitted(task):
        if not budget.reserve():
            return None
        row = run_episode(agent, args.model, args.effort, task, args.max_turns)
        budget.settle(row["cost_usd"])
        return row

    with ThreadPoolExecutor(args.workers) as pool, open(args.out, "a") as f:
        futures = [pool.submit(admitted, t) for t in tasks]
        rows, skipped = [], 0
        for fut in futures:
            row = fut.result()
            if row is None:
                skipped += 1
                continue
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"{row['task_id']:<22} reward={row['reward']:.2f} passed={row['passed']!s:<5} "
                  f"calls={row['tool_calls']:<3} ${row['cost_usd']:.3f}  missed={row['failed_fields']}")

    print("\nsummary")
    summarize(rows, args.difficulties)
    if args.samples > 1:
        summarize_samples(rows, args.difficulties, args.samples)
    if skipped:
        print(f"  budget_exhausted: {skipped} tasks not run (cap ${args.cap:.2f}, reservation ${estimate:.3f}/item)")
    cost = sum(r["cost_usd"] for r in rows)
    uncached = sum(r["uncached_cost_usd"] for r in rows)
    saved = f"  (without caching ${uncached:.2f}, saved {1 - cost / uncached:.0%})" if uncached else ""
    print(f"  total cost ${cost:.2f}{saved}")
    print(f"  tokens: input {sum(r['input_tokens'] for r in rows)}, cache read {sum(r['cache_read_tokens'] for r in rows)}, "
          f"cache write {sum(r['cache_write_tokens'] for r in rows)}, output {sum(r['output_tokens'] for r in rows)}")


if __name__ == "__main__":
    main()
