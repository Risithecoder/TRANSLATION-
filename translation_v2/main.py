import os
import uuid
import glob
import json
import time
from threading import Thread
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# ── Project modules ──────────────────────────────────────────────────────────
import text_extractor
import translator
import docx_builder

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
DEFAULT_BATCH_SIZE = 5

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
    language: str = Form(...)
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
        "status": "uploaded"
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

def _auto_batch_breaks(total_questions: int, batch_size: int = DEFAULT_BATCH_SIZE) -> list:
    """
    Generate batch break positions for auto-batching.
    Returns a list of question_no values after which a batch break is placed.
    E.g., with batch_size=5 and 12 questions: [5, 10] → batches [1-5], [6-10], [11-12]
    """
    breaks = []
    for i in range(batch_size, total_questions, batch_size):
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

        def _run_extraction(path=upload_path, jdir=job_path(job_id), k=key):
            try:
                print(f"[extraction] Starting for job {job_id}")
                qs = text_extractor.extract_questions(path, images_folder=jdir)
                print(f"[extraction] Done — {len(qs)} questions")
                thread_results[k] = {"result": qs}
            except Exception as ex:
                import traceback
                print(f"[extraction] ERROR: {ex}")
                print(traceback.format_exc())
                thread_results[k] = {"error": str(ex)}

        t = Thread(target=_run_extraction)
        t.start()
        while t.is_alive():
            yield ":\n\n"   # SSE heartbeat
            time.sleep(10)
        t.join()

        res = thread_results.pop(key, None)
        if res is None:
            raise Exception("Extraction thread returned no result.")
        if "error" in res:
            raise Exception(res["error"])

        all_questions = res["result"]
        if not all_questions:
            raise Exception("No questions found in the document.")

        # Re-number questions sequentially
        for idx, q in enumerate(all_questions, start=1):
            q["question_no"] = idx

        # Save extracted questions
        save_json(job_id, "questions.json", all_questions)

        # Auto-batch: generate default batch breaks every DEFAULT_BATCH_SIZE questions
        batch_breaks = _auto_batch_breaks(len(all_questions))
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
# Body: { questions: [...], batch_breaks: [15, 30, ...] }
#   batch_breaks: list of question_no values AFTER which a batch break is placed
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
    batch_breaks is a list of question_no values after which a new batch starts.
    E.g. breaks=[20, 40] with 60 questions → [Q1–Q20], [Q21–Q40], [Q41–Q60]
    Falls back to single batch if no breaks.
    """
    if not batch_breaks:
        return [questions]

    breaks_set = set(batch_breaks)
    batches = []
    current_batch = []

    for q in questions:
        current_batch.append(q)
        if q["question_no"] in breaks_set:
            batches.append(current_batch)
            current_batch = []

    if current_batch:
        batches.append(current_batch)

    return batches


def translate_stream_generator(job_id: str):
    try:
        questions_file = job_path(job_id, "questions.json")
        if not os.path.exists(questions_file):
            raise Exception("Questions not found. Run extraction first.")

        questions = load_json(job_id, "questions.json")
        batch_breaks = load_json(job_id, "batch_breaks.json") if os.path.exists(job_path(job_id, "batch_breaks.json")) else []
        meta = load_json(job_id, "meta.json")
        language = meta["language"]

        batches = split_into_batches(questions, batch_breaks)
        yield _sse(data=f"Starting translation into {language} — {len(batches)} batch(es), {len(questions)} question(s) total.")

        batch_outputs = [None] * len(batches)
        import concurrent.futures
        
        yield _sse(data="Submitting batches for parallel translation...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = set()
            for idx, batch in enumerate(batches):
                future = executor.submit(translator.translate_batch, batch, language)
                future.batch_idx = idx  # store index on future to maintain order
                futures.add(future)
                
            completed = 0
            while futures:
                # Wait for up to 10 seconds for any future to complete
                done, not_done = concurrent.futures.wait(
                    futures, 
                    timeout=10, 
                    return_when=concurrent.futures.FIRST_COMPLETED
                )
                
                # If nothing completed in 10s, send a heartbeat to keep SSE alive
                if not done:
                    yield ":\n\n"
                    continue
                    
                # Process completed futures
                for future in done:
                    try:
                        res = future.result()
                        idx = future.batch_idx
                        batch_outputs[idx] = res
                        completed += 1
                        yield _sse(data=f"Translated batch {idx + 1}/{len(batches)} (Completed: {completed}/{len(batches)}).")
                    except Exception as e:
                        raise Exception(f"Translation failed on batch {future.batch_idx + 1}: {e}")
                
                futures = not_done

        yield _sse(data="All batches translated. Assembling document...")

        # ── Step 6: Build DOCX ────────────────────────────────────────────
        original_name = os.path.splitext(meta["original_filename"])[0]
        output_filename = f"{language.lower()}_{original_name}.docx"
        output_path = job_path(job_id, output_filename)

        docx_builder.build(batch_outputs, output_path, language)

        meta["status"] = "done"
        meta["output_filename"] = output_filename
        save_json(job_id, "meta.json", meta)

        yield _sse(data="Document ready.")
        yield _sse(event="done", data=json.dumps({"filename": output_filename}))

    except Exception as e:
        yield _sse(event="error", data=str(e))


@app.get("/api/translate-stream/{job_id}")
def translate_stream(job_id: str):
    return StreamingResponse(
        translate_stream_generator(job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


# ═════════════════════════════════════════════════════════════════════════════
# ROUTE 6 — Download translated DOCX
# GET /api/download/{job_id}
# ═════════════════════════════════════════════════════════════════════════════
@app.get("/api/download/{job_id}")
def download(job_id: str):
    meta_file = job_path(job_id, "meta.json")
    if not os.path.exists(meta_file):
        raise HTTPException(404, "Job not found.")
    meta = load_json(job_id, "meta.json")
    if meta.get("status") != "done":
        raise HTTPException(400, "Translation not complete yet.")
    output_path = job_path(job_id, meta["output_filename"])
    if not os.path.exists(output_path):
        raise HTTPException(404, "Output file not found.")
    return FileResponse(
        output_path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=meta["output_filename"]
    )


# ═════════════════════════════════════════════════════════════════════════════
# ROUTE 7 — Serve frontend
# ═════════════════════════════════════════════════════════════════════════════
@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html"))


# ═════════════════════════════════════════════════════════════════════════════
# ROUTE 8 — Health check
# ═════════════════════════════════════════════════════════════════════════════
@app.get("/health")
def health():
    return {"status": "ok"}


# ── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5016)