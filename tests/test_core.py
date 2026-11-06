import random

import pytest

from gst_env import FilingSession, generate_task, grade, solve
from gst_env.generator import Invoice, Task
from gst_env.solver import _cash_after_credit, minimum_cash


def _task(*invoices: Invoice) -> Task:
    return Task("t", "easy", "2026-09", "36AAAAA0000A1Z5", list(invoices))


def sale(state: str, value: int, rate: int, n: int = 1) -> Invoice:
    return Invoice(f"S-{n}", "sale", None, state, value, rate)


def purchase(state: str, value: int, rate: int, category: str = "raw_material", rcm: bool = False, n: int = 1) -> Invoice:
    return Invoice(f"P-{n}", "purchase", "x", state, value, rate, category, rcm)


def test_intra_state_splits_cgst_sgst():
    truth = solve(_task(sale("36", 10_000, 18)))
    assert truth["outward_tax"] == {"igst": 0, "cgst": 900, "sgst": 900}
    assert truth["total_cash_payable"] == 1_800


def test_inter_state_is_igst():
    truth = solve(_task(sale("29", 10_000, 18)))
    assert truth["outward_tax"] == {"igst": 1_800, "cgst": 0, "sgst": 0}


def test_nil_rated_reported_separately():
    truth = solve(_task(sale("36", 10_000, 0), sale("36", 20_000, 5, n=2)))
    assert truth["nil_exempt_value"] == 10_000
    assert truth["outward_taxable_value"] == 20_000


def test_blocked_itc_not_claimed():
    truth = solve(_task(sale("36", 10_000, 18), purchase("36", 5_000, 18, "food_catering")))
    assert truth["itc_available"] == {"igst": 0, "cgst": 0, "sgst": 0}
    assert truth["blocked_itc"] == 900
    assert truth["total_cash_payable"] == 1_800


def test_rcm_paid_in_cash_even_with_excess_credit():
    truth = solve(_task(
        sale("36", 10_000, 18),
        purchase("36", 100_000, 18, n=1),
        purchase("36", 10_000, 18, "legal_services", rcm=True, n=2),
    ))
    assert truth["rcm_tax"] == {"igst": 0, "cgst": 900, "sgst": 900}
    assert truth["total_cash_payable"] == 1_800


def test_cgst_credit_cannot_pay_sgst():
    liab = {"igst": 0, "cgst": 0, "sgst": 1_000}
    itc = {"igst": 0, "cgst": 5_000, "sgst": 0}
    assert minimum_cash(liab, itc)["sgst"] == 1_000


def test_igst_credit_spills_to_cgst_and_sgst():
    liab = {"igst": 100, "cgst": 500, "sgst": 500}
    itc = {"igst": 1_000, "cgst": 0, "sgst": 0}
    assert sum(minimum_cash(liab, itc).values()) == 100


@pytest.mark.parametrize("seed", range(300))
def test_minimum_cash_matches_brute_force(seed):
    rng = random.Random(seed)
    liab = {h: rng.randint(0, 60) for h in ("igst", "cgst", "sgst")}
    itc = {h: rng.randint(0, 60) for h in ("igst", "cgst", "sgst")}
    spare = max(itc["igst"] - liab["igst"], 0)
    brute = min(sum(_cash_after_credit(liab, itc, x).values()) for x in range(spare + 1))
    assert sum(minimum_cash(liab, itc).values()) == brute


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_generation_is_deterministic(difficulty):
    assert generate_task(7, difficulty).to_dict() == generate_task(7, difficulty).to_dict()


def test_perfect_submission_scores_one_and_wrong_total_fails():
    task = generate_task(1, "hard")
    truth = solve(task)
    assert grade(task, truth) == {**grade(task, truth), "reward": 1.0, "passed": True}
    wrong = {**truth, "total_cash_payable": truth["total_cash_payable"] + 500}
    result = grade(task, wrong)
    assert not result["passed"] and result["reward"] < 0.8


def test_garbage_submission_scores_zero():
    assert grade(generate_task(2, "medium"), {"total_cash_payable": "lots"})["reward"] == 0


def test_session_reads_invoices_and_submits_once():
    task = generate_task(3, "hard")
    s = FilingSession(task)
    ids = s.list_invoices("sale")["invoice_ids"]
    assert s.get_invoice(ids[0])["kind"] == "sale"
    s.submit_return(solve(task))
    assert s.result["passed"]
    assert "error" in s.submit_return({})


def test_ledger_balance_reduces_cash_but_never_pays_rcm():
    task = _task(sale("36", 10_000, 18), purchase("36", 10_000, 18, "legal_services", rcm=True, n=2))
    task.opening_credit = {"igst": 0, "cgst": 50_000, "sgst": 50_000}
    truth = solve(task)
    assert truth["cash_payable"] == {"igst": 0, "cgst": 900, "sgst": 900}  # only the RCM tax
    assert truth["itc_available"] == {"igst": 0, "cgst": 900, "sgst": 900}  # this month's ITC only


def test_uneven_ledger_cgst_cannot_cover_sgst():
    task = _task(sale("36", 100_000, 18))
    task.opening_credit = {"igst": 0, "cgst": 100_000, "sgst": 0}
    assert solve(task)["cash_payable"] == {"igst": 0, "cgst": 0, "sgst": 9_000}


def test_hard_tasks_always_carry_a_ledger_balance():
    assert all(sum(generate_task(s, "hard").opening_credit.values()) > 0 for s in range(50))
    assert all(sum(generate_task(s, "easy").opening_credit.values()) == 0 for s in range(50))


def test_session_exposes_ledger():
    task = generate_task(4, "hard")
    assert FilingSession(task).get_credit_ledger() == {"opening_balance": task.opening_credit}


def test_wrong_value_in_a_zero_field_costs_reward():
    task = generate_task(11, "easy")  # easy tasks have no reverse charge, so rcm_tax is all zero
    truth = solve(task)
    assert truth["rcm_tax"] == {"igst": 0, "cgst": 0, "sgst": 0}
    padded = {**truth, "rcm_tax": {"igst": 0, "cgst": 100, "sgst": 0}}
    result = grade(task, padded)
    assert not result["passed"] and result["reward"] < 1.0


def test_every_invoice_is_reachable_through_pagination():
    task = _task(*[sale("36", 10_000, 18, n=i) for i in range(11)])
    s = FilingSession(task)
    assert s.list_invoices("sale")["total_pages"] == 2
    assert len(s.list_invoices("sale", page=2)["invoice_ids"]) == 1
    for seed in range(30):
        s = FilingSession(generate_task(seed, "hard"))
        for kind in ("sale", "purchase"):
            pages = s.list_invoices(kind)["total_pages"]
            seen = [i for p in range(1, pages + 1) for i in s.list_invoices(kind, p)["invoice_ids"]]
            assert seen == [i.invoice_id for i in s.task.invoices if i.kind == kind]


def test_each_level_generates_the_cases_it_claims_to_cover():
    from gst_env.rules import BLOCKED_CATEGORIES, HOME_STATE

    def covers(difficulty):
        invs = [i for s in range(60) for i in generate_task(s, difficulty).invoices]
        return {
            "inter_state": any(i.counterparty_state != HOME_STATE for i in invs),
            "nil_rated": any(i.kind == "sale" and i.rate == 0 for i in invs),
            "blocked": any(i.category in BLOCKED_CATEGORIES for i in invs),
            "reverse_charge": any(i.reverse_charge for i in invs),
        }

    assert covers("easy") == dict.fromkeys(("inter_state", "nil_rated", "blocked", "reverse_charge"), False)
    assert covers("medium") == {"inter_state": True, "nil_rated": True, "blocked": False, "reverse_charge": False}
    assert covers("hard") == dict.fromkeys(("inter_state", "nil_rated", "blocked", "reverse_charge"), True)
