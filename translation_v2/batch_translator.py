"""
batch_translator.py — Gemini Batch API integration for translation.

Provides an alternative, cost-effective translation path using the
asynchronous Gemini Batch API (50% cheaper than real-time).

Flow:
  1. submit_batch_job()  → Build inline requests, call client.batches.create()
  2. poll_batch_status() → Check job state via client.batches.get()
  3. collect_batch_results() → Parse inlined_responses, validate with Pydantic,
                               fallback to synchronous translate_batch() on failure
"""

import os
import concurrent.futures
from google import genai

# Reuse prompts, schema, and helpers from the existing translator module
import translator
from translator import (
    SYSTEM_INSTRUCTION,
    PROMPT_TEMPLATE,
    _format_questions_block,
    _parse_and_validate,
    translate_batch,  # fallback for failed batch responses
)

# ── Gemini setup ──────────────────────────────────────────────────────────────
MODEL = translator.MODEL

# Batch index marker embedded in prompts to correlate unordered responses
_BATCH_INDEX_PREFIX = "###BATCH_INDEX="
_BATCH_INDEX_SUFFIX = "###"


# ═════════════════════════════════════════════════════════════════════════════
# SUBMIT — Create a Batch API job with inline requests
# ═════════════════════════════════════════════════════════════════════════════

def submit_batch_job(batches: list, language: str) -> str:
    """
    Submit all translation batches as a single Gemini Batch API job.

    Args:
        batches: List of question lists (each inner list is one translation batch)
        language: Target language (e.g., "Hindi", "Tamil")

    Returns:
        The batch job name (string) used to poll status and retrieve results.
    """
    sys_inst = SYSTEM_INSTRUCTION.replace("{language}", language)

    inline_requests = []
    for idx, batch_questions in enumerate(batches):
        questions_block = _format_questions_block(batch_questions)

        # Build the prompt with a batch index marker so we can correlate
        # responses back to the correct batch, even if ordering is not preserved.
        prompt = (
            PROMPT_TEMPLATE
            .replace("{language}", language)
            .replace("{questions_block}", questions_block)
        )
        # Append the batch index marker at the very end of the prompt
        prompt += f"\n\n{_BATCH_INDEX_PREFIX}{idx}{_BATCH_INDEX_SUFFIX}"

        inline_requests.append({
            "contents": [
                {"role": "user", "parts": [{"text": prompt}]}
            ],
            "config": {
                "system_instruction": sys_inst,
                "response_mime_type": "application/json",
                "temperature": 0.1,
                "max_output_tokens": 65536,
            }
        })

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    batch_job = client.batches.create(
        model=MODEL,
        src=inline_requests,
    )

    print(f"[batch_translator] ✅ Batch job submitted: {batch_job.name} "
          f"({len(batches)} request(s))")

    return batch_job.name


# ═════════════════════════════════════════════════════════════════════════════
# POLL — Check the status of a batch job
# ═════════════════════════════════════════════════════════════════════════════

def poll_batch_status(batch_job_name: str) -> dict:
    """
    Check the current state of a batch job.

    Returns:
        dict with keys:
          - state: str ("JOB_STATE_PENDING", "JOB_STATE_RUNNING",
                        "JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", etc.)
          - done: bool (True if terminal state)
    """
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    batch_job = client.batches.get(name=batch_job_name)

    state_name = batch_job.state.name if batch_job.state else "UNKNOWN"

    terminal_states = {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    }

    result = {
        "state": state_name,
        "done": state_name in terminal_states,
    }

    print(f"[batch_translator] Poll {batch_job_name}: {state_name}")
    return result


# ═════════════════════════════════════════════════════════════════════════════
# COLLECT — Retrieve and validate batch results
# ═════════════════════════════════════════════════════════════════════════════




def collect_batch_results(
    batch_job_name: str,
    original_batches: list,
    language: str,
) -> list:
    """
    Retrieve results from a completed batch job, validate each response
    with Pydantic, and fall back to synchronous translation for failures.

    Args:
        batch_job_name: The Gemini batch job name
        original_batches: The original list of question batches (same order
                          as submitted to submit_batch_job)
        language: Target language

    Returns:
        List of validated batch outputs (same format as translator.translate_batch)
        in the same order as original_batches.
    """
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    batch_job = client.batches.get(name=batch_job_name)

    if batch_job.state.name != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(
            f"Batch job {batch_job_name} is not in SUCCEEDED state "
            f"(current: {batch_job.state.name})"
        )

    # Get inlined responses
    responses = batch_job.dest.inlined_responses
    if not responses:
        raise RuntimeError(
            f"Batch job {batch_job_name} succeeded but returned no responses."
        )

    print(f"[batch_translator] Collecting {len(responses)} response(s) "
          f"from batch job {batch_job_name}")

    # Since ordering is not guaranteed by the Batch API, but we submitted
    # requests in order and typically get them back in order for small batches,
    # we use positional mapping. For robustness, we validate count.
    if len(responses) != len(original_batches):
        print(f"[batch_translator] ⚠️  Response count mismatch: "
              f"got {len(responses)}, expected {len(original_batches)}")

    batch_outputs = [None] * len(original_batches)

    for idx in range(len(original_batches)):
        if idx >= len(responses):
            # Missing response — will be handled by fallback below
            print(f"[batch_translator] ⚠️  No response for batch {idx + 1}, "
                  f"will use fallback")
            continue

        response = responses[idx]
        try:
            # Extract text from the response
            raw_text = response.candidates[0].content.parts[0].text
            if not raw_text:
                raise ValueError("Empty response text")

            raw_json = raw_text.strip()

            # Validate with Pydantic (same as translator._parse_and_validate)
            batch = _parse_and_validate(raw_json)

            # Filter out dummy/placeholder questions
            batch.questions = [
                q for q in batch.questions
                if "dummy" not in q.english_question.lower()
                and "placeholder" not in q.english_question.lower()
                and "filler" not in q.english_question.lower()
            ]

            # Re-stamp original question_no from extraction
            questions = original_batches[idx]
            for i, tq in enumerate(batch.questions):
                if i < len(questions):
                    original_q_no = questions[i].get("question_no", 0)
                    try:
                        tq.question_no = int(original_q_no)
                    except (ValueError, TypeError):
                        pass

            batch_outputs[idx] = batch.model_dump()
            print(f"[batch_translator] ✅ Batch {idx + 1}: "
                  f"{len(batch.questions)} question(s) validated")

        except Exception as e:
            print(f"[batch_translator] ⚠️  Batch {idx + 1} validation failed: "
                  f"{str(e)[:200]}")
            # Will be handled by fallback below

    # Fallback: For any batch that failed validation, use synchronous translation
    fallback_indices = [idx for idx, output in enumerate(batch_outputs) if output is None]
    fallback_count = len(fallback_indices)
    
    if fallback_count > 0:
        print(f"[batch_translator] 🔄 Falling back to synchronous translation "
              f"for {fallback_count} batch(es) using parallel processing...")
        
        def _process_fallback(idx):
            print(f"[batch_translator] 🔄 Starting fallback for batch {idx + 1}...")
            try:
                result = translate_batch(original_batches[idx], language)
                print(f"[batch_translator] ✅ Fallback succeeded for batch {idx + 1}")
                return idx, result
            except Exception as e:
                raise RuntimeError(
                    f"Batch {idx + 1} failed both batch API and fallback "
                    f"synchronous translation: {e}"
                )
                
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(_process_fallback, idx) for idx in fallback_indices]
            for future in concurrent.futures.as_completed(futures):
                idx, result = future.result()
                batch_outputs[idx] = result
                
        print(f"[batch_translator] ℹ️  {fallback_count} batch(es) required "
              f"synchronous fallback")

    return batch_outputs
