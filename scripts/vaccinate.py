"""Qualify the test suite: plant one known defect at a time in a copy of the
package and check that the suite fails. A clean copy must still pass.

    python scripts/vaccinate.py                 # all seeds, writes evals/qualification/LEDGER.md
    python scripts/vaccinate.py --only m03 m09

Each seed is first checked to change behaviour (solver output, grades or
tool responses on generated tasks). A seed that changes nothing is reported as
equivalent and left out of recall, since a passing suite is then correct.
The baseline checker is one smoke test (perfect answer scores 1, wrong total
fails); the suite has to beat it to earn its cost.
"""

import argparse
import datetime
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SMOKE = "tests/test_core.py::test_perfect_submission_scores_one_and_wrong_total_fails"
LEDGER = ROOT / "evals" / "qualification" / "LEDGER.md"


@dataclass(frozen=True)
class Seed:
    id: str
    defect: str  # the class of error a CA would recognise
    file: str
    old: str
    new: str


SEEDS = [
    Seed("m01", "CGST credit used to pay SGST (cross-utilisation)", "gst_env/solver.py",
         "        itc[head] -= used\n    return liab",
         "        itc[head] -= used\n    x = min(itc['cgst'], liab['sgst'])\n    liab['sgst'] -= x\n    itc['cgst'] -= x\n    return liab"),
    Seed("m02", "IGST credit not applied to IGST first (Rule 88A order)", "gst_env/solver.py",
         'used = min(itc["igst"], liab["igst"])\n    liab["igst"] -= used\n    itc["igst"] -= used',
         'used = 0'),
    Seed("m03", "Reverse-charge tax paid from ITC instead of cash", "gst_env/solver.py",
         "    cash = minimum_cash(outward, credit)\n    for h in HEADS:\n        cash[h] += rcm[h]",
         "    cash = minimum_cash({h: outward[h] + rcm[h] for h in HEADS}, credit)"),
    Seed("m04", "Section 17(5) blocked credit claimed as ITC", "gst_env/solver.py",
         "elif inv.category in BLOCKED_CATEGORIES:", "elif False:"),
    Seed("m05", "Nil-rated sales counted as taxable value", "gst_env/solver.py",
         "nil_value += inv.taxable_value", "taxable_value += inv.taxable_value"),
    Seed("m06", "Inter-state supply split as CGST+SGST (place of supply)", "gst_env/solver.py",
         "if inv.counterparty_state == HOME_STATE:", "if True:"),
    Seed("m07", "Credit ledger balance brought forward ignored", "gst_env/solver.py",
         "itc[h] + task.opening_credit[h]", "itc[h]"),
    Seed("m08", "IGST credit split not optimised (always all to SGST)", "gst_env/solver.py",
         "    return best\n", "    return _cash_after_credit(liab, itc, 0)\n"),
    Seed("m09", "Grader tolerance loosened from Rs 1 to Rs 1,000", "gst_env/grader.py",
         "TOLERANCE = 1  #", "TOLERANCE = 1000  #"),
    Seed("m10", "Zero-value fields no longer penalised", "gst_env/grader.py",
         "penalty = sum(weights[p] for p in zero if not fields[p])", "penalty = 0"),
    Seed("m11", "Headline cash figure loses its extra weight", "gst_env/grader.py",
         '"total_cash_payable": 5,', '"total_cash_payable": 1,'),
    Seed("m12", "Pass decided by the total alone", "gst_env/grader.py",
         '"passed": all(fields.values()),', '"passed": fields["total_cash_payable"],'),
    Seed("m13", "Last partial page of invoices never reported", "gst_env/tools.py",
         '"total_pages": max(1, -(-len(ids) // PAGE_SIZE)),', '"total_pages": max(1, len(ids) // PAGE_SIZE),'),
    Seed("m14", "Return can be submitted more than once", "gst_env/tools.py",
         "        if self.done:\n", "        if False:\n"),
    Seed("m15", "Ledger tool hides the balance brought forward", "gst_env/tools.py",
         '{"opening_balance": dict(self.task.opening_credit)}', '{"opening_balance": {h: 0 for h in self.task.opening_credit}}'),
    Seed("m16", "Hard tasks stop generating reverse-charge purchases", "gst_env/generator.py",
         'if cfg["rcm"] and rng.random() < 0.15:', "if False:"),
    Seed("m17", "Hard tasks stop generating blocked purchases", "gst_env/generator.py",
         'if cfg["blocked"] and rng.random() < 0.2:', "if False:"),
    Seed("m18", "Medium tasks stop generating nil-rated sales", "gst_env/generator.py",
         'rate = 0 if cfg["nil"] and rng.random() < 0.15', 'rate = 0 if False'),
]

# Prints one JSON line describing behaviour; a seed that leaves it unchanged is equivalent.
PROBE = r"""
import json
from gst_env import DIFFICULTIES, FilingSession, generate_task, grade, solve
truths = json.load(open("truths.json"))
out = []
for d in DIFFICULTIES:
    for i in range(150):
        t = generate_task(7000 + i, d)
        ref = truths[t.task_id]
        zero = {h: 0 for h in ("igst", "cgst", "sgst")}
        subs = [ref, {**ref, "total_cash_payable": ref["total_cash_payable"] + 500},
                {**ref, "blocked_itc": ref["blocked_itc"] + 7}, {"total_cash_payable": ref["total_cash_payable"]},
                {"outward_taxable_value": 0, "nil_exempt_value": 0, "outward_tax": zero, "rcm_tax": zero,
                 "itc_available": zero, "blocked_itc": 0, "total_cash_payable": 0}]
        s = FilingSession(t)
        tools = [s.list_invoices(k) for k in ("sale", "purchase")] + [s.get_credit_ledger()]
        s.submit_return(ref)
        tools.append(s.submit_return(ref))
        out.append([t.to_dict(), solve(t), [grade(t, x) for x in subs], tools])
print(json.dumps(out, sort_keys=True))
"""

TRUTHS = r"""
import json
from gst_env import DIFFICULTIES, generate_task, solve
print(json.dumps({t.task_id: solve(t) for d in DIFFICULTIES for t in (generate_task(7000 + i, d) for i in range(150))}))
"""


def copy_repo(dest: Path) -> None:
    for name in ("gst_env", "tests", "scripts"):
        shutil.copytree(ROOT / name, dest / name, ignore=shutil.ignore_patterns("__pycache__"))


def py(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    # PYTHONPATH puts the copy ahead of the editable install of the real package.
    env = {"PYTHONPATH": str(cwd), "PATH": "/usr/bin:/bin"}
    return subprocess.run([sys.executable, *args], cwd=cwd, env=env, capture_output=True, text=True)


def suite_fails(cwd: Path, target: str | None = None) -> bool:
    r = py(cwd, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *([target] if target else []))
    if r.returncode not in (0, 1):
        raise RuntimeError(f"pytest did not run cleanly:\n{r.stdout[-2000:]}{r.stderr[-2000:]}")
    return r.returncode == 1


def run(seeds: list[Seed]) -> list[dict]:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        control = work / "control"
        copy_repo(control)
        r = py(control, "-c", TRUTHS)
        (work / "truths.json").write_text(r.stdout)
        shutil.copy(work / "truths.json", control / "truths.json")
        control_probe = py(control, "-c", PROBE).stdout
        control_fails = suite_fails(control)
        rows = [{"id": "control", "defect": "none (clean copy)", "equivalent": False,
                 "suite": "FALSE ALARM" if control_fails else "pass", "smoke": "pass" if not suite_fails(control, SMOKE) else "FALSE ALARM"}]

        for seed in seeds:
            d = work / seed.id
            copy_repo(d)
            shutil.copy(work / "truths.json", d / "truths.json")
            f = d / seed.file
            text = f.read_text()
            if text.count(seed.old) != 1:
                raise RuntimeError(f"{seed.id}: seed text must occur exactly once in {seed.file}")
            f.write_text(text.replace(seed.old, seed.new))
            probe = py(d, "-c", PROBE)
            equivalent = probe.returncode == 0 and probe.stdout == control_probe
            rows.append({
                "id": seed.id, "defect": seed.defect, "file": seed.file, "equivalent": equivalent,
                "suite": "caught" if suite_fails(d) else "MISSED",
                "smoke": "caught" if suite_fails(d, SMOKE) else "missed",
            })
            shutil.rmtree(d)
    return rows


def write_ledger(rows: list[dict]) -> str:
    live = [r for r in rows if r["id"] != "control" and not r["equivalent"]]
    caught = sum(r["suite"] == "caught" for r in live)
    smoke = sum(r["smoke"] == "caught" for r in live)
    control = rows[0]
    verdict = "PASS" if caught == len(live) and control["suite"] == "pass" else "FAIL"
    today = datetime.date.today().isoformat()
    lines = [
        "# Qualification ledger",
        "",
        "Generated by `scripts/vaccinate.py`. Checker under test: the full pytest suite. "
        f"Baseline: the single smoke test `{SMOKE.split('::')[1]}`.",
        "",
        "| date | target | artifact | defect classes | N | recall | FPR | baseline | verdict |",
        "|---|---|---|---|---|---|---|---|---|",
        f"| {today} | pytest suite | gst_env (solver, grader, tools, generator) | {len(live)} seeded classes"
        f" | {len(live)} | {caught}/{len(live)} | {'0/1' if control['suite'] == 'pass' else '1/1'} (clean control)"
        f" | smoke test {smoke}/{len(live)} | {verdict} |",
        "",
        "## Seeds",
        "",
        "| id | defect | file | suite | smoke test |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        if r["id"] == "control":
            lines.append(f"| control | {r['defect']} | - | {r['suite']} | {r['smoke']} |")
        elif r["equivalent"]:
            lines.append(f"| {r['id']} | {r['defect']} | `{r['file']}` | equivalent (no behaviour change; excluded) | - |")
        else:
            lines.append(f"| {r['id']} | {r['defect']} | `{r['file']}` | {r['suite']} | {r['smoke']} |")
    lines += ["", "The suite is deterministic, so one run per seed is a full measurement; replicates would repeat it.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", help="seed ids to run")
    parser.add_argument("--no-ledger", action="store_true")
    args = parser.parse_args()
    seeds = [s for s in SEEDS if not args.only or s.id in args.only]
    rows = run(seeds)
    for r in rows:
        tag = "equivalent" if r["equivalent"] else f"suite={r['suite']:<11} smoke={r['smoke']}"
        print(f"{r['id']:<8} {tag:<36} {r['defect']}")
    if not args.no_ledger:
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        LEDGER.write_text(write_ledger(rows))
        print(f"\nledger: {LEDGER.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
