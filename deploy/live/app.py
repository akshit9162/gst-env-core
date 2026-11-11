"""Live demo: file one GSTR-3B through Otari on request.

A visitor presses a button on https://akshit9162.github.io/gst-env-core/otari/ and this server runs the
real pipeline on a fresh task: the guard model and Otari's guardrail screen every invoice note, Qwen files
the return, GLM reviews it, gpt-oss-20b explains it to the owner. The result comes back with the grade
against the deterministic solver and every Otari request ID.

Spending is capped here (one run at a time, RUNS_PER_DAY a day) and by the key's own Otari budget.

    OTARI_API_KEY=... uvicorn app:app --app-dir deploy/live --port 8000
"""
import datetime
import os
import random
import sys
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace as NS

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import pipeline  # noqa: E402

from gst_env import FilingSession, generate_task, solve  # noqa: E402

RUNS_PER_DAY = int(os.environ.get("RUNS_PER_DAY", "40"))
# Off by default: on 12 Nov a turn using Otari's sandbox took ~7 minutes, too long for a visitor to wait.
CODE_EXEC = os.environ.get("LIVE_CODE_EXEC") == "1"
ROLES = dict(filer="otari:alibaba/qwen-3.5-35b-a3b", reviewer="otari:z-ai/glm-5.3-flash",
             explainer="otari:openai/gpt-oss-20b", prescreen="otari:alibaba/qwen3guard-gen-8b")

app = FastAPI(title="gst-env live")
app.add_middleware(CORSMiddleware, allow_origins=["https://akshit9162.github.io", "http://localhost:8765"],
                   allow_methods=["GET", "POST"], allow_headers=["*"])

jobs: dict[str, dict] = {}
busy = threading.Lock()
day = {"date": None, "runs": 0}


class RunRequest(BaseModel):
    difficulty: str = "easy"
    poison: bool = True


def invoices_view(task, session) -> list[dict]:
    return [session.get_invoice(i.invoice_id) for i in task.invoices]


def run(job_id: str, req: RunRequest, seed: int) -> None:
    job = jobs[job_id]
    try:
        task = generate_task(seed, req.difficulty)
        job["task"] = {"task_id": f"gstr3b-{req.difficulty}-{seed:05d}", "gstin": task.filer_gstin,
                       "period": task.period, "invoices": invoices_view(task, FilingSession(task))}
        args = NS(**ROLES, guardrail=None, code_exec=CODE_EXEC, web_fetch=False, users=False, max_turns=24,
                  benign_notes=True)
        row = pipeline.run_item(args, task, inject=req.poison)
        # what the visitor sees: the notes as the models saw them (flagged ones withheld)
        shown = pipeline.InjectedSession(task, None, benign_notes=True)
        notes = {s["invoice_id"]: s for s in row.get("prescreen") or []}
        invoices = []
        for inv in invoices_view(task, FilingSession(task)):
            note = shown.notes.get(inv["invoice_id"])
            screen = notes.get(inv["invoice_id"], {})
            if screen.get("injected"):
                note = pipeline.INJECTION
            invoices.append({**inv, "note": note, "flagged_by": screen.get("flagged_by"),
                             "guard": screen.get("guard"), "screen_error": screen.get("error"),
                             "screen_request_id": screen.get("request_id")})
        o = row.get("otari") or {}
        job.update(status="done", result={
            "task_id": row["task_id"], "seed": seed, "difficulty": req.difficulty, "gstin": task.filer_gstin,
            "period": task.period, "invoices": invoices, "poisoned": req.poison,
            "passed": row["passed"], "reward": row["reward"], "failed_fields": row.get("failed_fields", []),
            "filed": row.get("filed_return"), "correct": solve(task),
            "first_draft_passed": row.get("first_draft_passed"), "review": row.get("review"),
            "explanation": row.get("explanation"), "injection_withheld": row.get("injection_withheld"),
            "obeyed_injection": row.get("obeyed_injection"), "stop_reason": row.get("stop_reason"),
            "roles": row.get("role_models"), "cost_usd": row.get("total_cost_usd"),
            "filer": {"served": o.get("served_models"), "fallback_turns": o.get("fallback_turns"),
                      "code_executions": o.get("code_executions"), "request_ids": o.get("request_ids")},
            "reviewer": [{"served": r.get("served_models"), "request_ids": r.get("request_ids")}
                         for r in row.get("reviewer_otari") or [] if r],
            "seconds": round(time.time() - job["started"], 1),
        })
    except Exception as e:  # the visitor sees the failure, never a hung spinner
        job.update(status="error", error=f"{type(e).__name__}: {str(e)[:300]}")
    finally:
        busy.release()


@app.get("/")
def health() -> dict:
    return {"ok": True, "runs_today": day["runs"], "runs_per_day": RUNS_PER_DAY, "roles": ROLES}


@app.post("/api/run")
def start(req: RunRequest) -> dict:
    if req.difficulty not in ("easy", "medium"):
        raise HTTPException(400, "difficulty is easy or medium")
    today = datetime.datetime.now(datetime.timezone.utc).date()
    if day["date"] != today:
        day.update(date=today, runs=0)
    if day["runs"] >= RUNS_PER_DAY:
        raise HTTPException(429, f"today's {RUNS_PER_DAY} live runs are used up; the recorded runs are below")
    if not busy.acquire(blocking=False):
        raise HTTPException(409, "a run is in progress; try again in a minute")
    day["runs"] += 1
    seed = random.randint(90_000, 99_999)  # never one of the published tasks
    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {"status": "running", "started": time.time(), "seed": seed}
    threading.Thread(target=run, args=(job_id, req, seed), daemon=True).start()
    return {"job": job_id, "seed": seed}


@app.get("/api/jobs/{job_id}")
def status(job_id: str) -> dict:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    return {**{k: v for k, v in job.items() if k != "started"}, "elapsed": round(time.time() - job["started"], 1)}
