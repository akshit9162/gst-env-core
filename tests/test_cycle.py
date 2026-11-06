import pytest

from gst_env.cycle import (
    BLOCKED_CATEGORIES,
    HEADS,
    CycleSession,
    generate_cycle_task,
    solve_cycle,
    tax_heads,
)

SEEDS = range(40)


def entries(task, corrected: bool = True, skip_cancelled: bool = True) -> list[dict]:
    out = []
    for s in task.sales:
        if skip_cancelled and s.invoice_id in task.cancelled:
            continue
        value = task.corrections.get(s.invoice_id, s.taxable_value) if corrected else s.taxable_value
        out.append({"invoice_id": s.invoice_id, "customer_gstin": s.customer_gstin,
                    "place_of_supply": s.customer_state, "taxable_value": value, "rate": s.rate})
    return out


def expert(task, *, amend=True, ask=True, ims=True, skip_cancelled=True, reverse_blocked=True):
    """Plays the cycle the way a careful CA would; each flag switches one habit off."""
    s = CycleSession(task)
    s.call("read_messages", {})
    s.call("gstr1_file", {"entries": entries(task, corrected=False, skip_cancelled=skip_cancelled)})
    late = s.call("read_messages", {})["messages"]
    if amend and task.late_message and task.late_message in late:
        s.call("gstr1a_amend", {"entries": [e for e in entries(task) if e["invoice_id"] in task.corrections]})

    facts = {}
    for p in task.purchases:
        if p.remark and ask:
            q = "business_use" if p.category == "miscellaneous" else "goods_received"
            facts[p.invoice_id] = {q: s.call("ask_client", {"invoice_id": p.invoice_id, "question": q})["answer"] == "yes"}
    blocked = {h: 0 for h in HEADS}
    for d in task.ims:
        p = next((p for p in task.purchases if p.invoice_id == d.invoice_no), None)
        action = "accept"
        if ims:
            if p is None or p.taxable_value != d.taxable_value:
                action = "reject"
            elif not facts.get(p.invoice_id, {}).get("goods_received", True):
                action = "pending"
        s.call("ims_action", {"doc_id": d.doc_id, "action": action})
        if action == "accept" and p and (p.category in BLOCKED_CATEGORIES or not facts.get(p.invoice_id, {}).get("business_use", True)):
            for h, v in tax_heads(p.taxable_value, p.rate, p.supplier_state).items():
                blocked[h] += v
    s.call("gstr3b_prepare", {})

    truth = solve_cycle(task)
    s.call("gstr3b_file", {"gstr3b": {
        "rcm_tax_3_1_d": truth["rcm_tax"], "itc_rcm_4a_3": truth["rcm_tax"],
        "itc_reversed_blocked_4b_1": blocked if reverse_blocked else {h: 0 for h in HEADS},
        "itc_reversed_other_4b_2": {h: 0 for h in HEADS}, "cash_payable": truth["cash_payable"]}})
    return s


@pytest.mark.parametrize("seed", SEEDS)
def test_expert_play_scores_full_marks(seed):
    task = generate_cycle_task(seed)
    s = expert(task)
    assert s.done and s.result["passed"] and s.result["reward"] == 1.0, s.result


def test_generator_is_seeded_and_varied():
    assert generate_cycle_task(3).to_dict() == generate_cycle_task(3).to_dict()
    tasks = [generate_cycle_task(i) for i in SEEDS]
    assert any(t.late_message for t in tasks) and any(not t.late_message for t in tasks)
    assert any(t.cancelled for t in tasks)
    assert any(not f["goods_received"] for t in tasks for f in t.client_facts.values())
    assert any(not f["business_use"] for t in tasks for f in t.client_facts.values())


@pytest.mark.parametrize("habit", ["amend", "ask", "ims", "skip_cancelled", "reverse_blocked"])
def test_each_skipped_habit_fails_some_tasks(habit):
    failed = 0
    for seed in SEEDS:
        task = generate_cycle_task(seed)
        # Keep the cash the expert would pay, so only the skipped habit changes the outcome.
        failed += not expert(task, **{habit: False}).result["passed"]
    assert failed > 0


def test_doing_nothing_scores_zero():
    for seed in range(10):
        task = generate_cycle_task(seed)
        s = CycleSession(task)
        s.call("gstr1_file", {"entries": []})
        s.call("gstr3b_prepare", {})
        zero = {h: 0 for h in HEADS}
        s.call("gstr3b_file", {"gstr3b": {k: zero for k in (
            "rcm_tax_3_1_d", "itc_rcm_4a_3", "itc_reversed_blocked_4b_1", "itc_reversed_other_4b_2", "cash_payable")}})
        assert s.result["reward"] == 0.0


def test_late_message_only_after_gstr1_and_locks_hold():
    task = next(generate_cycle_task(i) for i in SEEDS if generate_cycle_task(i).late_message)
    s = CycleSession(task)
    assert task.late_message not in s.call("read_messages", {})["messages"]
    assert "error" in s.call("gstr1a_amend", {"entries": []})
    assert "error" in s.call("gstr3b_prepare", {})
    s.call("gstr1_file", {"entries": entries(task)})
    assert "error" in s.call("gstr1_file", {"entries": []})
    assert task.late_message in s.call("read_messages", {})["messages"]
    s.call("gstr3b_prepare", {})
    assert "error" in s.call("gstr1a_amend", {"entries": entries(task)})
    assert "error" in s.call("ims_action", {"doc_id": task.ims[0].doc_id, "action": "reject"})


def test_client_stops_answering_after_spare_questions():
    task = generate_cycle_task(1)
    s = CycleSession(task)
    pid = next(iter(task.client_facts))
    limit = sum(bool(p.remark) for p in task.purchases) + 2
    answers = [s.call("ask_client", {"invoice_id": pid, "question": "goods_received"})["answer"] for _ in range(limit + 1)]
    assert answers[-1] is None and all(a in ("yes", "no") for a in answers[:-1])


def test_bad_calls_return_errors():
    s = CycleSession(generate_cycle_task(2))
    assert "error" in s.call("nope", {})
    assert "error" in s.call("ims_action", None)
    assert "error" in s.call("ims_action", {"doc_id": "IMS-999", "action": "accept"})
    assert "error" in s.call("gstr1_file", {"entries": [{"invoice_id": "S-001"}]})
    assert "error" in s.call("gstr1_file", {"entries": [{"invoice_id": "S-001", "customer_gstin": None,
                                                         "place_of_supply": "99", "taxable_value": 1, "rate": 5}]})
