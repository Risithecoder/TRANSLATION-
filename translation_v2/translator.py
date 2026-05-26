import os
import re
import json
import time
from typing import List
from pydantic import BaseModel, field_validator, ValidationError
from google import genai
from google.genai import types

# ── Gemini setup ──────────────────────────────────────────────────────────────
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
MODEL  = "gemini-2.5-pro"


# ═════════════════════════════════════════════════════════════════════════════
# PYDANTIC SCHEMA — Every translated question must conform to this shape.
# Gemini's JSON mode + response_schema enforces the structure at the API level.
# Pydantic then validates field values and triggers retry on mismatch.
# ═════════════════════════════════════════════════════════════════════════════

class TranslatedQuestion(BaseModel):
    question_no: int
    english_question: str          # Full English question body (multi-line OK)
    translated_question: str       # Full translated question body
    english_options: List[str]     # Exactly 4 items: ["(1) ...", "(2) ...", ...]
    translated_options: List[str]  # Exactly 4 items: ["(5) ...", "(6) ...", ...]
    answer_key: str                # Just the number, e.g. "2"
    english_solution: str          # Full English solution / explanation
    translated_solution: str       # Full translated solution / explanation

    @field_validator("english_options", "translated_options")
    @classmethod
    def must_have_four_options(cls, v: List[str]) -> List[str]:
        if len(v) != 4:
            raise ValueError(f"Expected exactly 4 options, got {len(v)}")
        return v

    @field_validator("answer_key")
    @classmethod
    def answer_key_must_be_digit(cls, v: str) -> str:
        cleaned = v.strip()
        if not re.match(r"^\d+$", cleaned):
            raise ValueError(f"answer_key must be a digit string, got: {repr(v)}")
        return cleaned

    @field_validator("english_question", "translated_question",
                     "english_solution", "translated_solution")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field must not be empty")
        return v.strip()


class TranslationBatch(BaseModel):
    questions: List[TranslatedQuestion]

    @field_validator("questions")
    @classmethod
    def must_have_questions(cls, v: List[TranslatedQuestion]) -> List[TranslatedQuestion]:
        if not v:
            raise ValueError("questions list must not be empty")
        return v


# ═════════════════════════════════════════════════════════════════════════════
# PROMPTS
# ═════════════════════════════════════════════════════════════════════════════

SYSTEM_INSTRUCTION = """You are a high-precision examination translator with deep expertise in Indian government competitive exams. You translate structured English question papers with absolute fidelity to structure, formatting, and meaning.

OUTPUT FORMAT: You MUST return a valid JSON object matching exactly this schema:
{
  "questions": [
    {
      "question_no": <integer — the question number>,
      "english_question": "<Full question text in English including question number, e.g. '1. What is...' — preserve ALL internal items A./B./C./I./II. verbatim>",
      "translated_question": "<Full translated question text in {language} — preserve ALL internal labels verbatim>",
      "english_options": ["(1) <option1>", "(2) <option2>", "(3) <option3>", "(4) <option4>"],
      "translated_options": ["(5) <trans1>", "(6) <trans2>", "(7) <trans3>", "(8) <trans4>"],
      "answer_key": "<digit only, e.g. 2>",
      "english_solution": "<Full solution/explanation in English>",
      "translated_solution": "<Full solution/explanation translated into {language}>"
    }
  ]
}

ABSOLUTE RULES:
1. Every question in the batch MUST appear in the output JSON. Do NOT skip any.
2. english_options must always contain exactly 4 strings labelled (1)-(4).
3. translated_options must always contain exactly 4 strings labelled (5)-(8).
4. answer_key must be only a number string — no extra text.
5. english_solution and translated_solution must NEVER be empty strings.
6. Translate or transliterate ALL English words into {language} script in translated fields. No English letters except preserved labels (A., B., Roman numerals, symbols, units, dates, codes).
7. Do NOT alter equations, formulas, symbols, units, dates, codes, or placeholders.
"""

PROMPT_TEMPLATE = """Translate the following batch of examination questions into {language}.

CRITICAL RULE: QUESTION BODY STRUCTURE DETECTION
Many government exam questions contain internal statements labelled with letters or Roman numerals:
  A., B., C., D.  or  I., II., III.  or  1., 2., 3.  (internal items in question body)
These are PART OF THE QUESTION BODY — they are NOT answer options. Preserve them verbatim.

MANDATORY: Return a JSON object with a "questions" array. Each element must have ALL of these keys:
  - question_no (integer)
  - english_question (string)
  - translated_question (string)
  - english_options (array of exactly 4 strings: "(1) ...", "(2) ...", "(3) ...", "(4) ...")
  - translated_options (array of exactly 4 strings: "(5) ...", "(6) ...", "(7) ...", "(8) ...")
  - answer_key (string, digit only)
  - english_solution (string — MUST NOT be empty)
  - translated_solution (string — MUST NOT be empty)

────────────────────────────────────────────────────────────────────────────────
EXAMPLE INPUT:
1. What is the capital of India?
(1) Mumbai
(2) New Delhi
(3) Chennai
(4) Kolkata
Answer Key: 2
Solution:
New Delhi is the capital of India since 1911.

EXAMPLE OUTPUT (for {language}=Hindi):
{{
  "questions": [
    {{
      "question_no": 1,
      "english_question": "1. What is the capital of India?",
      "translated_question": "1. भारत की राजधानी क्या है?",
      "english_options": ["(1) Mumbai", "(2) New Delhi", "(3) Chennai", "(4) Kolkata"],
      "translated_options": ["(5) मुंबई", "(6) नई दिल्ली", "(7) चेन्नई", "(8) कोलकाता"],
      "answer_key": "2",
      "english_solution": "New Delhi is the capital of India since 1911.",
      "translated_solution": "नई दिल्ली 1911 से भारत की राजधानी है।"
    }}
  ]
}}
────────────────────────────────────────────────────────────────────────────────
QUESTIONS TO TRANSLATE:

{questions_block}"""

REPAIR_PROMPT_TEMPLATE = """The previous JSON output failed Pydantic validation with the following error:

VALIDATION ERROR:
{validation_error}

BROKEN JSON (returned by you):
{broken_json}

Please fix the JSON so it strictly conforms to the schema. Return ONLY the corrected JSON object — no markdown, no explanation.

Rules reminder:
- "questions" must be a non-empty array.
- Each question must have ALL keys: question_no, english_question, translated_question, english_options (exactly 4), translated_options (exactly 4), answer_key (digit only), english_solution (non-empty), translated_solution (non-empty).
- translated_solution MUST contain the actual translated explanation text — it must NOT be empty."""


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _format_questions_block(questions: list) -> str:
    blocks = []
    for q in questions:
        raw = q.get("raw_text", "").strip()
        if raw:
            blocks.append(raw)
    return "\n\n".join(blocks)


def _render_batch_to_text(batch: TranslationBatch, language: str) -> str:
    """
    Convert a validated TranslationBatch into the flat text format that
    docx_builder.py already knows how to render. This replaces the old
    brittle state-machine post-processor.
    """
    blocks = []
    for q in batch.questions:
        lines = []
        lines.append(q.english_question)
        lines.append(f"{language}:")
        lines.append(q.translated_question)
        lines.extend(q.english_options)
        lines.extend(q.translated_options)
        lines.append(f"Answer Key: {q.answer_key}")
        lines.append("Solution:")
        lines.append(q.english_solution)
        lines.append(f"{language}:")
        lines.append(q.translated_solution)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _call_gemini_json(prompt: str, system_instruction: str,
                      max_output_tokens: int = 65536) -> str:
    """
    Call Gemini with JSON mode enabled. Returns the raw response text.
    Raises ValueError if the model returned no text.
    """
    response = client.models.generate_content(
        model=MODEL,
        contents=[types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)]
        )],
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=max_output_tokens,
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=TranslationBatch,
        )
    )
    if response.text is None:
        finish_reason = "unknown"
        try:
            finish_reason = str(response.candidates[0].finish_reason)
        except Exception:
            pass
        raise ValueError(f"Gemini returned no text (finish_reason: {finish_reason}).")
    return response.text.strip()


def _parse_and_validate(raw_json: str) -> TranslationBatch:
    """
    Parse JSON string and validate with Pydantic.
    Raises json.JSONDecodeError or pydantic.ValidationError on failure.
    """
    data = json.loads(raw_json)
    return TranslationBatch.model_validate(data)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def translate_batch(questions: list, language: str, max_retries: int = 3) -> str:
    """
    Translate a batch of questions into the target language.

    Flow:
      1. Call Gemini with JSON mode + response_schema (structural enforcement).
      2. Parse response with Pydantic (field-level validation).
      3. On ValidationError → send a repair prompt with the broken JSON + error.
      4. Repeat up to max_retries times total.
      5. On success → render to flat text for docx_builder.

    Returns:
        Flat text string ready for docx_builder.build().
    Raises:
        RuntimeError if all attempts fail.
    """
    if not questions:
        return ""

    questions_block = _format_questions_block(questions)
    sys_inst = SYSTEM_INSTRUCTION.replace("{language}", language)

    main_prompt = (PROMPT_TEMPLATE
                   .replace("{language}", language)
                   .replace("{questions_block}", questions_block))

    last_raw   = ""
    last_error = ""

    for attempt in range(1, max_retries + 1):
        try:
            print(f"[translator] Attempt {attempt}/{max_retries} — "
                  f"{len(questions)} question(s) → {language}")

            # ── Choose prompt: initial or repair ─────────────────────────────
            if attempt == 1:
                prompt = main_prompt
            else:
                # Repair mode: give Gemini the broken JSON + validation error
                prompt = (REPAIR_PROMPT_TEMPLATE
                          .replace("{validation_error}", last_error)
                          .replace("{broken_json}", last_raw))

            # ── Step 1: Call Gemini in JSON mode ─────────────────────────────
            raw_json = _call_gemini_json(prompt, sys_inst)
            last_raw = raw_json

            # ── Step 2: Pydantic validation ───────────────────────────────────
            batch = _parse_and_validate(raw_json)

            # ── Step 3: Validate question count ───────────────────────────────
            if len(batch.questions) < len(questions):
                raise ValueError(
                    f"Expected {len(questions)} questions in JSON output, "
                    f"got {len(batch.questions)}. Some questions are missing."
                )

            # ── Step 4: Render to flat text ───────────────────────────────────
            print(f"[translator] ✅ Batch validated — "
                  f"{len(batch.questions)} question(s) OK")
            return _render_batch_to_text(batch, language)

        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            last_error = str(e)
            print(f"[translator] ⚠️  Attempt {attempt} validation failed: {last_error[:200]}")
            if attempt < max_retries:
                time.sleep(3 * attempt)
            else:
                raise RuntimeError(
                    f"Translation batch failed after {max_retries} attempts. "
                    f"Last error: {last_error}"
                )

        except Exception as e:
            last_error = str(e)
            print(f"[translator] ❌ Attempt {attempt} API error: {last_error[:200]}")
            if attempt < max_retries:
                time.sleep(3 * attempt)
            else:
                raise RuntimeError(
                    f"Translation batch failed after {max_retries} attempts: {last_error}"
                )

    return ""