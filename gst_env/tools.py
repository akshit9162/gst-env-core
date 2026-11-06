"""Tool-use interface: the agent sees only invoice data through these calls,
never the ground truth. Submitting ends the episode."""

from dataclasses import asdict

from .generator import Task
from .grader import grade
from .rules import BLOCKED_CATEGORIES, ELIGIBLE_CATEGORIES, HOME_STATE, RCM_CATEGORIES, STATES

PAGE_SIZE = 10

INSTRUCTIONS = """You are filing GSTR-3B for {gstin} ({state}) for period {period}.
Use the tools to read every sale and purchase invoice and the electronic credit ledger balance
brought forward, then call submit_return once with:
outward_taxable_value, nil_exempt_value, outward_tax{{igst,cgst,sgst}}, rcm_tax{{igst,cgst,sgst}},
itc_available{{igst,cgst,sgst}}, blocked_itc, total_cash_payable (all in whole rupees).
itc_available is this period's eligible ITC only. Set off liability using this period's ITC plus
the ledger balance, as per Rule 88A, minimising cash outflow. Tax under reverse charge must be
paid in cash. You can only submit once."""


class FilingSession:
    def __init__(self, task: Task):
        self.task = task
        self.result: dict | None = None

    @property
    def done(self) -> bool:
        return self.result is not None

    def instructions(self) -> str:
        return INSTRUCTIONS.format(
            gstin=self.task.filer_gstin, state=STATES[HOME_STATE], period=self.task.period
        )

    def list_invoices(self, kind: str, page: int = 1) -> dict:
        """List invoice IDs of one kind ("sale" or "purchase"), 10 per page."""
        ids = [i.invoice_id for i in self.task.invoices if i.kind == kind]
        start = (page - 1) * PAGE_SIZE
        return {
            "invoice_ids": ids[start:start + PAGE_SIZE],
            "page": page,
            "total_pages": max(1, -(-len(ids) // PAGE_SIZE)),
        }

    def get_invoice(self, invoice_id: str) -> dict:
        """Return full details of one invoice."""
        for inv in self.task.invoices:
            if inv.invoice_id == invoice_id:
                return asdict(inv)
        return {"error": f"invoice {invoice_id} not found"}

    def get_credit_ledger(self) -> dict:
        """Return the electronic credit ledger balance brought forward into this period."""
        return {"opening_balance": dict(self.task.opening_credit)}

    def describe_category(self, category: str) -> dict:
        """Describe a purchase category code."""
        for table in (ELIGIBLE_CATEGORIES, BLOCKED_CATEGORIES):
            if category in table:
                return {"category": category, "description": table[category]}
        if category in RCM_CATEGORIES:
            return {"category": category, "description": RCM_CATEGORIES[category][0]}
        return {"error": f"unknown category {category}"}

    def submit_return(self, gstr3b: dict) -> dict:
        """Submit the completed GSTR-3B. Can be called once."""
        if self.done:
            return {"error": "return already submitted"}
        self.result = grade(self.task, gstr3b)
        return {"status": "submitted"}

    def call(self, name: str, args: dict | None) -> dict:
        """Dispatch one tool call by name; bad input comes back as {"error": ...}."""
        if not isinstance(args, dict):
            return {"error": "arguments were not valid JSON"}
        try:
            match name:
                case "list_invoices":
                    return self.list_invoices(args["kind"], args.get("page", 1))
                case "get_invoice":
                    return self.get_invoice(args["invoice_id"])
                case "describe_category":
                    return self.describe_category(args["category"])
                case "get_credit_ledger":
                    return self.get_credit_ledger()
                case "submit_return":
                    return self.submit_return(args["gstr3b"])
        except (KeyError, TypeError) as e:
            return {"error": f"bad arguments: {e}"}
        return {"error": f"unknown tool {name}"}
