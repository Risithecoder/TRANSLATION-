import os
import re
import uuid
import json
import time
import asyncio
from threading import Thread
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# ── Project modules ──────────────────────────────────────────────────────────
import text_extractor

import docx_builder
# import cache_manager
import batch_translator
import email_sender

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI()



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Folders ──────────────────────────────────────────────────────────────────
JOBS_FOLDER = os.path.join(os.getcwd(), "jobs")
os.makedirs(JOBS_FOLDER, exist_ok=True)

# Default batch size for auto-batching questions before translation
DEFAULT_BATCH_SIZE = 3

# ── In-memory thread result store ────────────────────────────────────────────
thread_results: dict = {}

# ── Helpers ──────────────────────────────────────────────────────────────────
def job_path(job_id: str, *parts) -> str:
    return os.path.join(JOBS_FOLDER, job_id, *parts)

def save_json(job_id: str, filename: str, data):
    with open(job_path(job_id, filename), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(job_id: str, filename: str):
    with open(job_path(job_id, filename), "r", encoding="utf-8") as f:
        return json.load(f)

# ═════════════════════════════════════════════════════════════════════════════
# ROUTE 1 — Upload file, kick off extraction stream
# POST /api/upload
# Body: file (.docx/.html), language (str)
# Returns: { job_id }
# ═════════════════════════════════════════════════════════════════════════════
@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    language: str = Form(...),
):
    ext = os.path.splitext(file.filename or "")[-1].lower()
    if ext not in (".docx", ".html", ".htm"):
        raise HTTPException(400, "Only .docx and .html files are supported.")

    job_id = str(uuid.uuid4())
    os.makedirs(job_path(job_id), exist_ok=True)

    # Save uploaded file
    upload_path = job_path(job_id, f"source{ext}")
    content = await file.read()
    with open(upload_path, "wb") as f:
        f.write(content)

    # Save metadata
    save_json(job_id, "meta.json", {
        "language": language,
        "original_filename": file.filename,
        "ext": ext,
        "status": "uploaded",
    })

    return {"job_id": job_id}


# ═════════════════════════════════════════════════════════════════════════════
# ROUTE 2 — SSE stream: convert → images → vision extraction
# GET /api/extract-stream/{job_id}
# Streams progress events, saves extracted questions to questions.json
# ═════════════════════════════════════════════════════════════════════════════
def _sse(event: str = "message", data: str = "") -> str:
    if event == "message":
        return f"data: {data}\n\n"
    return f"event: {event}\ndata: {data}\n\n"

def _auto_batch_breaks(questions: list, batch_size: int = DEFAULT_BATCH_SIZE) -> list:
    """
    Generate batch break positions using absolute 0-based indices.
    Returns a list of indices — each index is the LAST question in a batch.
    E.g. for 10 questions with batch_size=3: [2, 5, 8]  (last batch has q[9])
    """
    breaks = []
    for i in range(batch_size - 1, len(questions) - 1, batch_size):
        breaks.append(i)
    return breaks



def extract_stream_generator(job_id: str):
    try:
        meta = load_json(job_id, "meta.json")
        ext = meta["ext"]
        upload_path = job_path(job_id, f"source{ext}")

        yield _sse(data="Extracting questions from document...")

        key = f"{job_id}_extract"
        thread_results[key] = None

        def _run_extraction(path=upload_path, k=key):
            try:
                print(f"[extraction] Starting for job {job_id}")
                qs = text_extractor.extract_questions(path)
                print(f"[extraction] Done — {len(qs)} questions")
                thread_results[k] = {"result": qs}
            except BaseException as ex:
                import traceback
                tb = traceback.format_exc()
                print(f"[extraction] ERROR: {ex}")
                print(tb)
                thread_results[k] = {"error": str(ex), "traceback": tb}

        t = Thread(target=_run_extraction, daemon=True)
        t.start()
        while t.is_alive():
            yield ":\n\n"   # SSE heartbeat
            time.sleep(1)
        t.join()
        time.sleep(0.1)  # brief pause to ensure thread_results write is visible

        res = thread_results.pop(key, None)
        if res is None:
            raise Exception(
                "Extraction thread returned no result. "
                "Check server logs for details — the thread may have crashed silently."
            )
        if "error" in res:
            tb_hint = f" | Traceback: {res.get('traceback', '')[-300:]}" if res.get('traceback') else ""
            raise Exception(f"{res['error']}{tb_hint}")

        all_questions = res["result"]
        if not all_questions:
            raise Exception("No questions found in the document.")

        # Save extracted questions
        save_json(job_id, "questions.json", all_questions)

        # Auto-batch: generate default batch breaks every DEFAULT_BATCH_SIZE questions
        batch_breaks = _auto_batch_breaks(all_questions)
        save_json(job_id, "batch_breaks.json", batch_breaks)
        num_batches = len(batch_breaks) + 1

        # Update status
        meta["status"] = "extracted"
        meta["total_questions"] = len(all_questions)
        meta["batch_breaks"] = batch_breaks
        save_json(job_id, "meta.json", meta)

        yield _sse(data=f"Extraction complete — {len(all_questions)} question(s) found, auto-batched into {num_batches} batch(es) of {DEFAULT_BATCH_SIZE}.")
        yield _sse(event="done", data=json.dumps({
            "total_questions": len(all_questions),
            "batch_breaks": batch_breaks,
            "batch_size": DEFAULT_BATCH_SIZE
        }))

    except Exception as e:
        yield _sse(event="error", data=str(e))


@app.get("/api/extract-stream/{job_id}")
def extract_stream(job_id: str):
    return StreamingResponse(
        extract_stream_generator(job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


# ═════════════════════════════════════════════════════════════════════════════
# ROUTE 3 — Get extracted questions for review
# GET /api/review/{job_id}
# Returns: { questions: [...], language: str, total: int }
# ═════════════════════════════════════════════════════════════════════════════
@app.get("/api/review/{job_id}")
def get_review(job_id: str):
    questions_file = job_path(job_id, "questions.json")
    if not os.path.exists(questions_file):
        raise HTTPException(404, "Questions not found. Extraction may still be running.")
    questions = load_json(job_id, "questions.json")
    meta = load_json(job_id, "meta.json")
    return {
        "questions": questions,
        "language": meta["language"],
        "total": len(questions)
    }


# ═════════════════════════════════════════════════════════════════════════════
# ROUTE 4 — Save reviewed + batch-broken questions
# POST /api/save-review/{job_id}
# Body: { questions: [...], batch_breaks: [2, 5, 8, ...] }
#   batch_breaks: list of 0-based indices — each is the LAST question in a batch
# ═════════════════════════════════════════════════════════════════════════════
@app.post("/api/save-review/{job_id}")
async def save_review(job_id: str, request: Request):
    body = await request.json()
    batch_breaks = body.get("batch_breaks", [])

    # Use the original extracted questions.json — no re-parsing needed
    questions_file = job_path(job_id, "questions.json")
    if not os.path.exists(questions_file):
        raise HTTPException(400, "Extraction not complete. questions.json not found. "
                                 "Please re-run extraction.")
    questions = load_json(job_id, "questions.json")

    save_json(job_id, "batch_breaks.json", batch_breaks)

    meta = load_json(job_id, "meta.json")
    meta["status"] = "reviewed"
    meta["batch_breaks"] = batch_breaks
    save_json(job_id, "meta.json", meta)

    return {"ok": True, "total_questions": len(questions), "batch_breaks": batch_breaks}


# ═════════════════════════════════════════════════════════════════════════════
# ROUTE 5 — SSE stream: translate batches → assemble docx
# GET /api/translate-stream/{job_id}
# ═════════════════════════════════════════════════════════════════════════════
def split_into_batches(questions: list, batch_breaks: list) -> list:
    """
    Split questions list into batches using batch_breaks.
    batch_breaks is a sorted list of 0-based indices — each index is the LAST
    question in a batch.  E.g. [2, 5, 8] means:
      batch 1 = questions[0:3], batch 2 = questions[3:6], batch 3 = questions[6:9], batch 4 = rest
    If breaks is empty, falls back to fixed batches of DEFAULT_BATCH_SIZE (3).
    """
    if not batch_breaks:
        return [questions[i:i + DEFAULT_BATCH_SIZE] for i in range(0, len(questions), DEFAULT_BATCH_SIZE)]

    # Sort and deduplicate, clamp to valid range
    sorted_breaks = sorted(set(b for b in batch_breaks if 0 <= b < len(questions)))

    if not sorted_breaks:
        return [questions[i:i + DEFAULT_BATCH_SIZE] for i in range(0, len(questions), DEFAULT_BATCH_SIZE)]

    batches = []
    start = 0
    for break_idx in sorted_breaks:
        end = break_idx + 1  # inclusive → exclusive
        if start < end <= len(questions):
            batches.append(questions[start:end])
            start = end

    # Remaining questions after the last break
    if start < len(questions):
        batches.append(questions[start:])

    return batches




# ═════════════════════════════════════════════════════════════════════════════
# ROUTE 6 — Batch API: Submit batch translation job
# POST /api/batch-translate/{job_id}
# ═════════════════════════════════════════════════════════════════════════════
@app.post("/api/batch-translate/{job_id}")
async def batch_translate(job_id: str):
    questions_file = job_path(job_id, "questions.json")
    if not os.path.exists(questions_file):
        raise HTTPException(400, "Questions not found. Run extraction first.")

    questions = load_json(job_id, "questions.json")
    batch_breaks = (
        load_json(job_id, "batch_breaks.json")
        if os.path.exists(job_path(job_id, "batch_breaks.json"))
        else []
    )
    meta = load_json(job_id, "meta.json")
    language = meta["language"]

    batches = split_into_batches(questions, batch_breaks)

    try:
        batch_job_name = batch_translator.submit_batch_job(batches, language)
    except Exception as e:
        raise HTTPException(500, f"Failed to submit batch job: {e}")

    # Save batch job info into meta
    meta["status"] = "batch_translating"
    meta["batch_job_name"] = batch_job_name
    meta["batch_count"] = len(batches)
    save_json(job_id, "meta.json", meta)

    return {
        "ok": True,
        "batch_job_name": batch_job_name,
        "batch_count": len(batches),
    }


# ═════════════════════════════════════════════════════════════════════════════
# ROUTE 7 — Batch API: Poll batch job status
# GET /api/batch-status/{job_id}
# ═════════════════════════════════════════════════════════════════════════════
@app.get("/api/batch-status/{job_id}")
async def batch_status(job_id: str):
    meta_file = job_path(job_id, "meta.json")
    if not os.path.exists(meta_file):
        raise HTTPException(404, "Job not found.")

    meta = load_json(job_id, "meta.json")
    batch_job_name = meta.get("batch_job_name")
    if not batch_job_name:
        raise HTTPException(400, "No batch job found for this job. Use real-time mode or submit a batch first.")

    try:
        status = batch_translator.poll_batch_status(batch_job_name)
    except Exception as e:
        raise HTTPException(500, f"Failed to poll batch status: {e}")

    return status


# ═════════════════════════════════════════════════════════════════════════════
# ROUTE 8 — Batch API: Collect results and build DOCX
# POST /api/batch-collect/{job_id}
# ═════════════════════════════════════════════════════════════════════════════
def _run_collect_and_build(job_id: str, batch_job_name: str, batches: list, language: str, meta: dict):
    try:
        batch_outputs = batch_translator.collect_batch_results(batch_job_name, batches, language)
        
        # Build DOCX
        original_name = os.path.splitext(meta["original_filename"])[0]
        output_filename = f"{language.lower()}_{original_name}.docx"
        output_path = job_path(job_id, output_filename)

        docx_builder.build(batch_outputs, output_path, language)

        meta["status"] = "done"
        meta["output_filename"] = output_filename
        save_json(job_id, "meta.json", meta)
    except Exception as e:
        print(f"[batch-collect] Error in background task: {e}")
        meta["status"] = "collect_failed"
        meta["error"] = str(e)
        save_json(job_id, "meta.json", meta)

@app.post("/api/batch-collect/{job_id}")
async def batch_collect(job_id: str, background_tasks: BackgroundTasks):
    meta_file = job_path(job_id, "meta.json")
    if not os.path.exists(meta_file):
        raise HTTPException(404, "Job not found.")

    meta = load_json(job_id, "meta.json")
    batch_job_name = meta.get("batch_job_name")
    if not batch_job_name:
        raise HTTPException(400, "No batch job found for this job.")

    questions = load_json(job_id, "questions.json")
    batch_breaks = (
        load_json(job_id, "batch_breaks.json")
        if os.path.exists(job_path(job_id, "batch_breaks.json"))
        else []
    )
    language = meta["language"]
    batches = split_into_batches(questions, batch_breaks)

    meta["status"] = "collecting"
    save_json(job_id, "meta.json", meta)

    background_tasks.add_task(_run_collect_and_build, job_id, batch_job_name, batches, language, meta)

    return {"ok": True, "status": "collecting"}

@app.get("/api/collect-status/{job_id}")
def collect_status(job_id: str):
    meta_file = job_path(job_id, "meta.json")
    if not os.path.exists(meta_file):
        raise HTTPException(404, "Job not found.")
    meta = load_json(job_id, "meta.json")
    status = meta.get("status")
    
    if status == "collect_failed":
        raise HTTPException(500, meta.get("error", "Collection failed."))
        
    return {
        "status": status,
        "filename": meta.get("output_filename") if status == "done" else None
    }




# ═════════════════════════════════════════════════════════════════════════════
# ROUTE 10 — Send translated DOCX via email
# POST /api/send-email/{job_id}
# ═════════════════════════════════════════════════════════════════════════════
@app.post("/api/send-email/{job_id}")
async def send_email(job_id: str, request: Request):
    meta_file = job_path(job_id, "meta.json")
    if not os.path.exists(meta_file):
        raise HTTPException(404, "Job not found.")
    meta = load_json(job_id, "meta.json")
    if meta.get("status") != "done":
        raise HTTPException(400, "Translation not complete yet.")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid request body.")

    username = body.get("username", "").strip()
    if not username:
        raise HTTPException(400, "Username is required.")
    # Basic validation: alphanumeric, dots, underscores, hyphens
    if not re.match(r'^[a-zA-Z0-9._-]+$', username):
        raise HTTPException(400, "Invalid username. Use only letters, numbers, dots, underscores, or hyphens.")

    output_filename = meta.get("output_filename")
    if not output_filename:
        raise HTTPException(404, "Output filename not found in job metadata.")
    output_path = job_path(job_id, output_filename)
    if not os.path.exists(output_path):
        raise HTTPException(404, "Output file not found.")

    # Count questions for email summary
    question_count = 0
    questions_file = job_path(job_id, "questions.json")
    if os.path.exists(questions_file):
        questions = load_json(job_id, "questions.json")
        question_count = len(questions)

    try:
        result = await asyncio.to_thread(
            email_sender.send_translation_email,
            username,
            output_path,
            meta["language"],
            question_count,
            meta.get("original_filename", "document"),
        )
        return result
    except RuntimeError as e:
        raise HTTPException(500, str(e))


# ═════════════════════════════════════════════════════════════════════════════
# ROUTE 11 — Serve frontend
# ═════════════════════════════════════════════════════════════════════════════
@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html"))




# ═════════════════════════════════════════════════════════════════════════════
# ROUTE 12 — Health check
# ═════════════════════════════════════════════════════════════════════════════
@app.get("/health")
def health():
    return {"status": "ok"}


# ── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5016)