"""Monthly-cycle tier: the current GST filing workflow on a mock portal.

The agent works for a CA firm filing one client's month:
  1. read the client's sales register, purchase register and messages (the "world"),
  2. file GSTR-1 (invoice level) and, if a correction arrives afterwards, amend it in GSTR-1A,
  3. accept, reject or keep pending each supplier invoice in IMS,
  4. open GSTR-3B: tables 3.1 and 4A(5) are auto-filled from GSTR-1/1A and GSTR-2B and locked,
  5. enter the manual residue (3.1(d) reverse charge, 4A(3), 4B reversals, cash) and file.

Some facts are only in the client's head: the agent must ask (`ask_client`), and the
client stops answering after a few questions more than needed.

Grading is outcome-based: the filed numbers, not the clicks. Every simplification is
listed in README.md under the cycle tier and needs CA review.
"""

import random
from dataclasses import asdict, dataclass, field

from .generator import _gstin, _state, _value
from .grader import score
from .rules import BLOCKED_CATEGORIES, ELIGIBLE_CATEGORIES, HOME_STATE, RATES, RCM_CATEGORIES, STATES
from .solver import HEADS, minimum_cash

TAXED = [r for r in RATES if r]
QUESTIONS = ("goods_received", "business_use")
SPARE_QUESTIONS = 2
IMS_ACTIONS = ("accept", "reject", "pending")


@dataclass
class Sale:
    invoice_id: str
    customer_gstin: str | None  # None for B2C
    customer_state: str
    taxable_value: int
    rate: int
    status: str = "issued"  # register status; a message may say otherwise


@dataclass
class Purchase:
    invoice_id: str
    supplier_gstin: str | None  # None for an unregistered supplier
    supplier_state: str
    taxable_value: int
    rate: int
    category: str
    reverse_charge: bool = False
    remark: str = ""


@dataclass
class ImsDoc:
    doc_id: str
    supplier_gstin: str
    supplier_state: str
    invoice_no: str
    taxable_value: int
    rate: int


@dataclass
class CycleTask:
    task_id: str
    difficulty: str
    period: str
    filer_gstin: str
    sales: list[Sale] = field(default_factory=list)
    purchases: list[Purchase] = field(default_factory=list)
    ims: list[ImsDoc] = field(default_factory=list)
    opening_credit: dict[str, int] = field(default_factory=lambda: {h: 0 for h in HEADS})
    messages: list[str] = field(default_factory=list)  # visible from the start
    late_message: str | None = None  # arrives once GSTR-1 is filed
    # Ground truth the agent cannot read directly.
    cancelled: list[str] = field(default_factory=list)
    corrections: dict[str, int] = field(default_factory=dict)  # sale id -> true taxable value
    client_facts: dict[str, dict[str, bool]] = field(default_factory=dict)  # purchase id -> answers

    @property
    def invoices(self) -> list:
        return [*self.sales, *self.purchases]

    def to_dict(self) -> dict:
        return asdict(self)


def tax_heads(value: int, rate: int, state: str) -> dict[str, int]:
    tax = value * rate // 100
    if state == HOME_STATE:
        return {"igst": 0, "cgst": tax // 2, "sgst": tax // 2}
    return {"igst": tax, "cgst": 0, "sgst": 0}


def _add(acc: dict, tax: dict) -> None:
    for h in HEADS:
        acc[h] += tax[h]


def _zero() -> dict[str, int]:
    return {h: 0 for h in HEADS}


def generate_cycle_task(seed: int, period: str = "2026-09") -> CycleTask:
    rng = random.Random(f"cycle:{seed}")
    task = CycleTask(task_id=f"cycle-{seed:05d}", difficulty="cycle", period=period,
                     filer_gstin=_gstin(rng, HOME_STATE))
    inter_share = rng.uniform(0.1, 0.6)

    for i in range(rng.randint(8, 14)):
        state = _state(rng, inter_share)
        b2b = rng.random() < 0.7
        task.sales.append(Sale(
            invoice_id=f"S-{i + 1:03d}",
            customer_gstin=_gstin(rng, state) if b2b else None,
            customer_state=state,
            taxable_value=_value(rng),
            rate=0 if rng.random() < 0.1 else rng.choice(TAXED),
        ))
    if rng.random() < 0.5:
        s = rng.choice(task.sales)
        task.cancelled.append(s.invoice_id)
        task.messages.append(f"{s.invoice_id} was cancelled before dispatch (customer backed out). "
                             "It is still marked issued in the register, sorry.")
    issued = [s for s in task.sales if s.invoice_id not in task.cancelled]
    if rng.random() < 0.6:
        s = rng.choice(issued)
        true_value = s.taxable_value + rng.choice([-1, 1]) * rng.randint(1, 20) * 200
        task.corrections[s.invoice_id] = max(true_value, 200)
        task.late_message = (f"Correction for {s.invoice_id}: the taxable value is "
                             f"Rs {task.corrections[s.invoice_id]}, not Rs {s.taxable_value}. "
                             "Our accounts team typed it wrong in the register.")

    kinds = (["clean"] * rng.randint(4, 7) + ["blocked"] * rng.randint(0, 1)
             + ["ask_use"] * rng.randint(1, 2) + ["ask_receipt"] * rng.randint(1, 2)
             + ["mismatch"] * rng.randint(0, 1) + ["missing_upload"] * rng.randint(0, 1)
             + ["rcm"] * rng.randint(0, 1))
    rng.shuffle(kinds)
    for i, kind in enumerate(kinds):
        pid = f"P-{i + 1:03d}"
        state = _state(rng, inter_share)
        if kind == "rcm":
            category = rng.choice(list(RCM_CATEGORIES))
            task.purchases.append(Purchase(pid, None, HOME_STATE, _value(rng, hi=200_000),
                                           RCM_CATEGORIES[category][1], category, reverse_charge=True))
            continue
        category = rng.choice(list(BLOCKED_CATEGORIES if kind == "blocked" else ELIGIBLE_CATEGORIES))
        remark = ""
        if kind == "ask_use":
            category, remark = "miscellaneous", "Check with client: business or personal?"
        elif kind == "ask_receipt":
            remark = "Delivery status not confirmed."
        p = Purchase(pid, _gstin(rng, state), state, _value(rng, hi=300_000), rng.choice(TAXED), category, remark=remark)
        task.purchases.append(p)
        task.client_facts[pid] = {
            "goods_received": not (kind == "ask_receipt" and rng.random() < 0.6),
            "business_use": not (kind == "ask_use" and rng.random() < 0.6),
        }
        if kind == "missing_upload":
            continue
        uploaded = p.taxable_value
        if kind == "mismatch":
            uploaded += rng.randint(5, 50) * 200
        task.ims.append(ImsDoc(f"IMS-{len(task.ims) + 1:03d}", p.supplier_gstin, p.supplier_state, pid, uploaded, p.rate))

    # A supplier upload that is not the client's purchase at all.
    if rng.random() < 0.5:
        state = _state(rng, inter_share)
        task.ims.append(ImsDoc(f"IMS-{len(task.ims) + 1:03d}", _gstin(rng, state), state,
                               f"INV-{rng.randint(1000, 9999)}", _value(rng, hi=200_000), rng.choice(TAXED)))
    rng.shuffle(task.ims)

    heavy, light = rng.sample(["cgst", "sgst"], 2)
    task.opening_credit = {"igst": rng.choice([0, rng.randint(1, 100_000)]),
                           heavy: rng.randint(0, 300_000), light: rng.randint(0, 30_000)}
    return task


def _purchase(task: CycleTask, pid: str) -> Purchase | None:
    return next((p for p in task.purchases if p.invoice_id == pid), None)


def solve_cycle(task: CycleTask) -> dict:
    """The correct filing, in the same shape `CycleSession` reports what was filed."""
    gstr1 = {}
    for s in task.sales:
        if s.invoice_id in task.cancelled:
            continue
        gstr1[s.invoice_id] = {"customer_gstin": s.customer_gstin, "place_of_supply": s.customer_state,
                               "taxable_value": task.corrections.get(s.invoice_id, s.taxable_value), "rate": s.rate}

    rcm, itc_net, blocked = _zero(), _zero(), _zero()
    for p in task.purchases:
        tax = tax_heads(p.taxable_value, p.rate, p.supplier_state)
        if p.reverse_charge:
            _add(rcm, tax)
            _add(itc_net, tax)
            continue
        doc = next((d for d in task.ims if d.invoice_no == p.invoice_id), None)
        facts = task.client_facts[p.invoice_id]
        if doc is None or doc.taxable_value != p.taxable_value or not facts["goods_received"]:
            continue  # not in GSTR-2B this month: no credit yet
        if p.category in BLOCKED_CATEGORIES or not facts["business_use"]:
            _add(blocked, tax)
        else:
            _add(itc_net, tax)
    return _summary(gstr1, rcm, itc_net, blocked, task.opening_credit)


def _summary(gstr1: dict, rcm: dict, itc_net: dict, blocked: dict, opening: dict, cash: dict | None = None) -> dict:
    outward, value, nil = _zero(), 0, 0
    for e in gstr1.values():
        if e["rate"] == 0:
            nil += e["taxable_value"]
        else:
            value += e["taxable_value"]
            _add(outward, tax_heads(e["taxable_value"], e["rate"], e["place_of_supply"]))
    if cash is None:  # the truth: Rule 88A minimum cash, plus reverse charge in cash
        cash = minimum_cash(outward, {h: itc_net[h] + opening[h] for h in HEADS})
        _add(cash, rcm)
    return {
        "gstr1": gstr1,
        "outward_taxable_value": value,
        "nil_exempt_value": nil,
        "outward_tax": outward,
        "rcm_tax": rcm,
        "itc_net": itc_net,
        "blocked_itc": sum(blocked.values()),
        "cash_payable": cash,
        "total_cash_payable": sum(cash.values()),
    }


WEIGHTS = {
    "outward_taxable_value": 1,
    "nil_exempt_value": 1,
    **{f"outward_tax.{h}": 1 for h in HEADS},
    **{f"rcm_tax.{h}": 1 for h in HEADS},
    **{f"itc_net.{h}": 1 for h in HEADS},
    "blocked_itc": 1,
    "gstr1_mismatches": 2,  # truth is 0, so it can only cost reward
    "total_cash_payable": 5,
}


def grade_cycle(task: CycleTask, filed: dict) -> dict:
    truth = solve_cycle(task)
    keys = set(truth["gstr1"]) | set(filed["gstr1"])
    mismatches = sum(truth["gstr1"].get(k) != filed["gstr1"].get(k) for k in keys)
    return score({**truth, "gstr1_mismatches": 0}, {**filed, "gstr1_mismatches": mismatches}, WEIGHTS)


HEADS_SCHEMA = {
    "type": "object",
    "properties": {h: {"type": "integer"} for h in HEADS},
    "required": list(HEADS),
    "additionalProperties": False,
}
ENTRY_SCHEMA = {
    "type": "object",
    "properties": {
        "invoice_id": {"type": "string"},
        "customer_gstin": {"type": ["string", "null"]},
        "place_of_supply": {"type": "string", "description": "two-digit state code"},
        "taxable_value": {"type": "integer"},
        "rate": {"type": "integer"},
    },
    "required": ["invoice_id", "customer_gstin", "place_of_supply", "taxable_value", "rate"],
    "additionalProperties": False,
}
NO_ARGS = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}


def _spec(name: str, description: str, properties: dict | None = None) -> dict:
    schema = NO_ARGS if properties is None else {
        "type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}
    return {"name": name, "description": description, "input_schema": schema}


TOOLS = [
    _spec("get_sales_register", "The client's sales register for the period."),
    _spec("get_purchase_register", "The client's purchase register (their books) for the period, with any remarks."),
    _spec("read_messages", "Messages from the client. New messages can arrive while you work."),
    _spec("get_credit_ledger", "Electronic credit ledger balance (IGST, CGST, SGST) brought forward."),
    _spec("ask_client", "Ask the client one question about one purchase. They answer only a few questions, so ask only when the books leave it open.",
          {"invoice_id": {"type": "string"}, "question": {"type": "string", "enum": list(QUESTIONS)}}),
    _spec("gstr1_file", "File GSTR-1: one entry per sale invoice to report. Can be filed once.",
          {"entries": {"type": "array", "items": ENTRY_SCHEMA}}),
    _spec("gstr1a_amend", "Amend filed GSTR-1 entries before GSTR-3B is prepared. Each entry replaces the one with the same invoice_id; taxable_value 0 removes it.",
          {"entries": {"type": "array", "items": ENTRY_SCHEMA}}),
    _spec("ims_list", "Supplier-uploaded invoices in the Invoice Management System (IMS) and your current action on each."),
    _spec("ims_action", "Accept, reject or keep pending one IMS document. Documents with no action are deemed accepted when GSTR-2B is generated.",
          {"doc_id": {"type": "string"}, "action": {"type": "string", "enum": list(IMS_ACTIONS)}}),
    _spec("gstr3b_prepare", "Generate GSTR-2B from IMS and open GSTR-3B. Locks GSTR-1/1A and IMS. Returns the auto-filled, non-editable tables 3.1 and 4A(5)."),
    _spec("gstr3b_file", "File GSTR-3B with the manual tables and the cash paid per head, in whole rupees. Ends the task.",
          {"gstr3b": {"type": "object", "properties": {
              "rcm_tax_3_1_d": HEADS_SCHEMA, "itc_rcm_4a_3": HEADS_SCHEMA, "itc_reversed_blocked_4b_1": HEADS_SCHEMA,
              "itc_reversed_other_4b_2": HEADS_SCHEMA, "cash_payable": HEADS_SCHEMA},
              "required": ["rcm_tax_3_1_d", "itc_rcm_4a_3", "itc_reversed_blocked_4b_1", "itc_reversed_other_4b_2", "cash_payable"],
              "additionalProperties": False}}),
]

INSTRUCTIONS = """You work for a CA firm and are filing GST for client {gstin} ({state}) for period {period}.
Do the whole monthly cycle on the portal tools: read the client's registers and messages, file GSTR-1,
act on every supplier invoice in IMS, prepare GSTR-3B (tables 3.1 and 4A(5) fill in automatically and
cannot be edited), then file GSTR-3B with the reverse-charge tax, reversals and the cash paid per head.
Claim input tax credit only where the law allows it this month. Set off liability using the credit ledger
balance plus this month's net ITC as per Rule 88A, minimising cash; reverse-charge tax is paid in cash.
State codes: {states}. All amounts are whole rupees."""


class CycleSession:
    """The mock portal. Tools return dicts; errors come back as {"error": ...}."""

    tools = TOOLS

    def __init__(self, task: CycleTask):
        self.task = task
        self.gstr1: dict[str, dict] | None = None
        self.ims_actions: dict[str, str] = {}
        self.gstr2b: dict[str, int] | None = None
        self.questions = 0
        self.result: dict | None = None

    @property
    def done(self) -> bool:
        return self.result is not None

    def instructions(self) -> str:
        return INSTRUCTIONS.format(gstin=self.task.filer_gstin, state=STATES[HOME_STATE], period=self.task.period,
                                   states=", ".join(f"{k} {v}" for k, v in STATES.items()))

    def call(self, name: str, args: dict | None) -> dict:
        if not isinstance(args, dict):
            return {"error": "arguments were not valid JSON"}
        fn = getattr(self, f"_t_{name}", None)
        if fn is None:
            return {"error": f"unknown tool {name}"}
        if self.done:
            return {"error": "GSTR-3B already filed"}
        try:
            return fn(**args)
        except (KeyError, TypeError, ValueError) as e:
            return {"error": f"bad arguments: {e}"}

    # Reading the world.
    def _t_get_sales_register(self) -> dict:
        return {"sales": [asdict(s) for s in self.task.sales]}

    def _t_get_purchase_register(self) -> dict:
        return {"purchases": [asdict(p) for p in self.task.purchases]}

    def _t_read_messages(self) -> dict:
        msgs = list(self.task.messages)
        if self.gstr1 is not None and self.task.late_message:
            msgs.append(self.task.late_message)
        return {"messages": msgs or ["Please file this month's returns. Thanks!"]}

    def _t_get_credit_ledger(self) -> dict:
        return {"opening_balance": dict(self.task.opening_credit)}

    def _t_ask_client(self, invoice_id: str, question: str) -> dict:
        if question not in QUESTIONS:
            return {"error": f"question must be one of {list(QUESTIONS)}"}
        if invoice_id not in self.task.client_facts:
            return {"error": f"no question to ask about {invoice_id}"}
        self.questions += 1
        limit = sum(bool(p.remark) for p in self.task.purchases) + SPARE_QUESTIONS
        if self.questions > limit:
            return {"answer": None, "note": "The client has stopped replying today."}
        return {"answer": "yes" if self.task.client_facts[invoice_id][question] else "no"}

    # GSTR-1 and GSTR-1A.
    @staticmethod
    def _entries(entries: list[dict]) -> dict[str, dict]:
        out = {}
        for e in entries:
            if e["place_of_supply"] not in STATES:
                raise ValueError(f"unknown place_of_supply {e['place_of_supply']}")
            out[e["invoice_id"]] = {"customer_gstin": e["customer_gstin"], "place_of_supply": e["place_of_supply"],
                                    "taxable_value": int(e["taxable_value"]), "rate": int(e["rate"])}
        return out

    def _t_gstr1_file(self, entries: list[dict]) -> dict:
        if self.gstr1 is not None:
            return {"error": "GSTR-1 already filed; use gstr1a_amend"}
        self.gstr1 = self._entries(entries)
        return {"status": "filed", "entries": len(self.gstr1)}

    def _t_gstr1a_amend(self, entries: list[dict]) -> dict:
        if self.gstr1 is None:
            return {"error": "file GSTR-1 first"}
        if self.gstr2b is not None:
            return {"error": "GSTR-3B is prepared; GSTR-1A is locked"}
        for k, e in self._entries(entries).items():
            if e["taxable_value"] == 0:
                self.gstr1.pop(k, None)
            else:
                self.gstr1[k] = e
        return {"status": "amended", "entries": len(self.gstr1)}

    # IMS and GSTR-2B.
    def _t_ims_list(self) -> dict:
        return {"documents": [{**asdict(d), "action": self.ims_actions.get(d.doc_id, "no action")} for d in self.task.ims]}

    def _t_ims_action(self, doc_id: str, action: str) -> dict:
        if self.gstr2b is not None:
            return {"error": "GSTR-2B is generated; IMS is locked"}
        if action not in IMS_ACTIONS:
            return {"error": f"action must be one of {list(IMS_ACTIONS)}"}
        if all(d.doc_id != doc_id for d in self.task.ims):
            return {"error": f"document {doc_id} not found"}
        self.ims_actions[doc_id] = action
        return {"status": "saved", "doc_id": doc_id, "action": action}

    def _auto_tables(self) -> dict:
        s = _summary(self.gstr1 or {}, _zero(), _zero(), _zero(), self.task.opening_credit, cash=_zero())
        return {"3_1_a_taxable_value": s["outward_taxable_value"], "3_1_a_tax": s["outward_tax"],
                "3_1_c_nil_exempt_value": s["nil_exempt_value"], "4a_5_itc_from_gstr2b": dict(self.gstr2b)}

    def _t_gstr3b_prepare(self) -> dict:
        if self.gstr1 is None:
            return {"error": "file GSTR-1 first"}
        if self.gstr2b is None:
            self.gstr2b = _zero()
            for d in self.task.ims:
                if self.ims_actions.get(d.doc_id, "accept") == "accept":  # no action = deemed accepted
                    _add(self.gstr2b, tax_heads(d.taxable_value, d.rate, d.supplier_state))
        return {"status": "prepared", "auto_filled_locked": self._auto_tables()}

    def _t_gstr3b_file(self, gstr3b: dict) -> dict:
        if self.gstr2b is None:
            return {"error": "prepare GSTR-3B first"}
        heads = {k: {h: int(gstr3b[k][h]) for h in HEADS} for k in (
            "rcm_tax_3_1_d", "itc_rcm_4a_3", "itc_reversed_blocked_4b_1", "itc_reversed_other_4b_2", "cash_payable")}
        itc_net = {h: self.gstr2b[h] + heads["itc_rcm_4a_3"][h] - heads["itc_reversed_blocked_4b_1"][h]
                   - heads["itc_reversed_other_4b_2"][h] for h in HEADS}
        filed = _summary(self.gstr1, heads["rcm_tax_3_1_d"], itc_net, heads["itc_reversed_blocked_4b_1"],
                         self.task.opening_credit, cash=heads["cash_payable"])
        self.result = grade_cycle(self.task, filed)
        return {"status": "filed"}
