"""verifiers (v1) adapter: serves gst-env to `vf eval`, prime-rl and the Environments Hub.

Taskset id: `gst-env-vf`. Choose the tier with the task config:

    vf eval gst-env-vf --env.taskset.task.tier cycle --env.taskset.num-tasks 50

The tool server keeps no state of its own. Every rollout's state is the log of tool
calls; each call replays that log on a fresh session (the env is deterministic and a
replay costs microseconds), runs the new call, then appends it. The reward replays the
same log host-side and grades the result, so tools and reward can never disagree.
"""

import json
from typing import Literal

from pydantic import Field

import verifiers.v1 as vf

from gst_env import DIFFICULTIES, FilingSession, generate_task
from gst_env.cycle import CycleSession, generate_cycle_task

Tier = Literal["gstr3b", "cycle"]


def make_session(info: dict):
    if info["tier"] == "cycle":
        return CycleSession(generate_cycle_task(info["seed"]))
    return FilingSession(generate_task(info["seed"], info["difficulty"]))


def replay(info: dict, calls: list[dict]):
    session = make_session(info)
    for c in calls:
        session.call(c["name"], c["args"])
    return session


class GstState(vf.State):
    calls: list[dict] = Field(default_factory=list)


class GstData(vf.TaskData):
    info: dict
    """tier, seed and (gstr3b tier) difficulty: everything needed to rebuild the task."""


class _Portal(vf.Toolset[vf.ToolsetConfig, GstState]):
    TOOL_PREFIX = None  # advertise tools by their own names

    async def setup_task(self, task) -> None:
        self.info = task.data.info

    def _do(self, name: str, **args) -> str:
        calls = self.state.calls
        out = replay(self.info, calls).call(name, args)
        calls.append({"name": name, "args": args})
        return json.dumps(out)


class Gstr3bTools(_Portal):
    @vf.tool
    def list_invoices(self, kind: Literal["sale", "purchase"], page: int = 1) -> str:
        """List invoice IDs of one kind for the filing period, 10 per page."""
        return self._do("list_invoices", kind=kind, page=page)

    @vf.tool
    def get_invoice(self, invoice_id: str) -> str:
        """Get full details of one invoice: counterparty state code, taxable value, rate, purchase category and reverse-charge flag."""
        return self._do("get_invoice", invoice_id=invoice_id)

    @vf.tool
    def get_credit_ledger(self) -> str:
        """Get the electronic credit ledger balance (IGST, CGST, SGST) brought forward into this period."""
        return self._do("get_credit_ledger")

    @vf.tool
    def describe_category(self, category: str) -> str:
        """Describe what a purchase category code means."""
        return self._do("describe_category", category=category)

    @vf.tool
    def submit_return(self, gstr3b: dict) -> str:
        """Submit the completed GSTR-3B: outward_taxable_value, nil_exempt_value, outward_tax{igst,cgst,sgst},
        rcm_tax{...}, itc_available{...}, blocked_itc, total_cash_payable, in whole rupees. Once only; ends the task."""
        return self._do("submit_return", gstr3b=gstr3b)


class CycleTools(_Portal):
    @vf.tool
    def get_sales_register(self) -> str:
        """The client's sales register for the period."""
        return self._do("get_sales_register")

    @vf.tool
    def get_purchase_register(self) -> str:
        """The client's purchase register (their books) for the period, with any remarks."""
        return self._do("get_purchase_register")

    @vf.tool
    def read_messages(self) -> str:
        """Messages from the client. New messages can arrive while you work."""
        return self._do("read_messages")

    @vf.tool
    def get_credit_ledger(self) -> str:
        """Electronic credit ledger balance (IGST, CGST, SGST) brought forward."""
        return self._do("get_credit_ledger")

    @vf.tool
    def ask_client(self, invoice_id: str, question: Literal["goods_received", "business_use"]) -> str:
        """Ask the client one question about one purchase. They answer only a few questions, so ask only when the books leave it open."""
        return self._do("ask_client", invoice_id=invoice_id, question=question)

    @vf.tool
    def gstr1_file(self, entries: list[dict]) -> str:
        """File GSTR-1 once: one entry per sale invoice to report, each with invoice_id, customer_gstin (null for B2C),
        place_of_supply (two-digit state code), taxable_value and rate."""
        return self._do("gstr1_file", entries=entries)

    @vf.tool
    def gstr1a_amend(self, entries: list[dict]) -> str:
        """Amend filed GSTR-1 entries before GSTR-3B is prepared. Each entry replaces the one with the same invoice_id; taxable_value 0 removes it."""
        return self._do("gstr1a_amend", entries=entries)

    @vf.tool
    def ims_list(self) -> str:
        """Supplier-uploaded invoices in the Invoice Management System (IMS) and your current action on each."""
        return self._do("ims_list")

    @vf.tool
    def ims_action(self, doc_id: str, action: Literal["accept", "reject", "pending"]) -> str:
        """Accept, reject or keep pending one IMS document. Documents with no action are deemed accepted when GSTR-2B is generated."""
        return self._do("ims_action", doc_id=doc_id, action=action)

    @vf.tool
    def gstr3b_prepare(self) -> str:
        """Generate GSTR-2B from IMS and open GSTR-3B. Locks GSTR-1/1A and IMS. Returns the auto-filled, non-editable tables 3.1 and 4A(5)."""
        return self._do("gstr3b_prepare")

    @vf.tool
    def gstr3b_file(self, gstr3b: dict) -> str:
        """File GSTR-3B with rcm_tax_3_1_d, itc_rcm_4a_3, itc_reversed_blocked_4b_1, itc_reversed_other_4b_2 and
        cash_payable, each {igst,cgst,sgst} in whole rupees. Ends the task."""
        return self._do("gstr3b_file", gstr3b=gstr3b)


class GstTaskConfig(vf.TaskConfig):
    tier: Tier = "gstr3b"
    tools: vf.ToolsetConfig = vf.ToolsetConfig()


class GstTask(vf.Task[GstData, GstState, GstTaskConfig]):
    @classmethod
    def toolsets(cls, config: GstTaskConfig) -> list[vf.Toolset]:
        return [CycleTools(config.tools) if config.tier == "cycle" else Gstr3bTools(config.tools)]

    def _session(self, trace: vf.Trace):
        return replay(self.data.info, trace.state.calls)

    @vf.stop
    async def filed(self, trace: vf.Trace) -> bool:
        return self._session(trace).done

    @vf.reward
    async def outcome(self, trace: vf.Trace) -> float:
        result = self._session(trace).result
        return result["reward"] if result else 0.0

    @vf.metric
    async def passed(self, trace: vf.Trace) -> float:
        result = self._session(trace).result
        return float(bool(result and result["passed"]))


class GstTasksetConfig(vf.TasksetConfig):
    task: GstTaskConfig = GstTaskConfig()
    num_tasks: int = 100
    """Tasks per difficulty (gstr3b tier) or in total (cycle tier)."""
    difficulties: list[str] = Field(default_factory=lambda: list(DIFFICULTIES))
    seed_offset: int = 0
    """Keep training and evaluation seeds apart, e.g. 0 for training and 1_000_000 for eval."""


class GstTaskset(vf.Taskset[GstTask, GstTasksetConfig]):
    def load(self) -> list[GstTask]:
        tier = self.config.task.tier
        if tier == "cycle":
            infos = [{"tier": tier, "seed": self.config.seed_offset + i} for i in range(self.config.num_tasks)]
        else:
            infos = [{"tier": tier, "difficulty": d, "seed": self.config.seed_offset + i}
                     for d in self.config.difficulties for i in range(self.config.num_tasks)]
        tasks = []
        for info in infos:
            session = make_session(info)
            name = session.task.task_id
            tasks.append(GstTask(GstData(name=name, id=name, prompt=session.instructions(), info=info), self.config.task))
        return tasks


__all__ = ["GstTask", "GstTaskConfig", "GstTaskset", "GstTasksetConfig", "make_session", "replay"]
