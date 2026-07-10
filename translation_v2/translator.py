import os
import re
import json
import time
from typing import List
from pydantic import BaseModel, field_validator, ValidationError
from google import genai
from google.genai import types

# ── Gemini setup ──────────────────────────────────────────────────────────────
MODEL  = "gemini-3.1-pro-preview"


# ═════════════════════════════════════════════════════════════════════════════
# PYDANTIC SCHEMA — Every translated question must conform to this shape.
# Gemini's JSON mode + response_schema enforces the structure at the API level.
# Pydantic then validates field values and triggers retry on mismatch.
# ═════════════════════════════════════════════════════════════════════════════

class TranslatedQuestion(BaseModel):
    question_no: int
    english_passage: str = ""      # Passage/directions text (e.g. reading comprehension) — empty if none
    translated_passage: str = ""   # Translated passage/directions text — empty if none
    english_question: str          # Full English question body (multi-line OK)
    translated_question: str       # Full translated question body
    english_options: List[str]     # Array of options, e.g. 4 or 5 items
    translated_options: List[str]  # Array of translated options
    answer_key: str                # Just the number, e.g. "2"
    english_solution: str          # Full English solution / explanation
    translated_solution: str       # Full translated solution / explanation

    @field_validator("english_options", "translated_options")
    @classmethod
    def must_have_valid_options(cls, v: List[str]) -> List[str]:
        if len(v) != 0 and (len(v) < 2 or len(v) > 8):
            raise ValueError(f"Expected between 2 and 8 options (or 0 for passages), got {len(v)}")
        return v

    @field_validator("answer_key")
    @classmethod
    def answer_key_must_be_digit(cls, v: str) -> str:
        cleaned = v.strip()
        if cleaned == "":
            return cleaned
        if not re.match(r"^\d+$", cleaned):
            raise ValueError(f"answer_key must be a digit string or empty, got: {repr(v)}")
        return cleaned

    @field_validator("translated_question")
    @classmethod
    def remove_leading_number(cls, v: str) -> str:
        cleaned = v.strip()
        # Remove leading number like "1. ", "23.", etc.
        return re.sub(r"^\d+\.\s*", "", cleaned)

    @field_validator("english_question", "translated_question")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field must not be empty")
        return v.strip()

    @field_validator("english_passage", "translated_passage")
    @classmethod
    def strip_passage(cls, v: str) -> str:
        return v.strip() if v else ""


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
      "question_no": <integer — the question number. Use 0 if it is a standalone passage/directions block>,
      "english_passage": "",
      "translated_passage": "",
      "english_question": "<The question stem. If this is a passage/directions block, put the FULL passage here.>",
      "translated_question": "<The translated question stem or passage.>",
      "english_options": ["(1) <opt1>", "(2) <opt2>", "..."],
      "translated_options": ["(number) <trans1>", "(number) <trans2>", "..."],
      "answer_key": "<digit only, or empty string \"\" for passages>",
      "english_solution": "<Full solution in English, or empty string \"\" for passages>",
      "translated_solution": "<Full solution translated into {language}, or empty string \"\" for passages>"
    }
  ]
}

ABSOLUTE RULES:
1. Every item in the batch MUST appear in the output JSON. Do NOT skip any.
2. For normal questions, english_options must contain all options present in the input (e.g., 4 or 5 strings). For passages/directions/other text, it must be an empty list [].
3. For normal questions, translated_options must contain all corresponding translated options. For passages/directions/other text, it must be an empty list [].
4. Translate or transliterate ALL English words into {language} script in translated fields. No English letters except preserved labels.
5. Do NOT alter equations, formulas, symbols, units, dates, codes, or placeholders.
6. PASSAGE/DIRECTIONS RULE: If you encounter an item that is a standalone passage, directions block, or general context, you MUST output it as its own item in the JSON array. Set `question_no` to 0. Put ALL of the original text in `english_question` and its translation in `translated_question`. Leave options as `[]` and answer_key/solutions as `""`.
7. DO NOT OMIT ANY TEXT. If the input contains extra sentences, context, or notes, you MUST include them in the translated output.
8. MARKDOWN TABLES: If the input contains a Markdown table (e.g. | Col1 | Col2 |), you MUST preserve the exact Markdown table format in the output. Do NOT remove the | characters and do NOT break rows across multiple lines.
9. NEVER create dummy, placeholder, or filler questions. Only translate questions that actually exist in the input. If fewer questions exist than expected, translate ONLY what is present — do NOT invent extra questions.
10. PRESERVE FORMATTING: If the input contains `**bold**`, `__underline__`, `<sup>` or `<sub>` markers, you MUST preserve them exactly around the corresponding translated text. Also preserve any bullet points (`- `) or numbering (`1. `, `2. `) exactly as they appear.
11. PRESERVE NEWLINES AND PARAGRAPHS: You MUST preserve all newlines (`\n`) and paragraph breaks from the original text. If a sentence starts on a new line (like a bullet point, a numbered item, or a section heading like 'Further Insights:'), the translated sentence MUST also start on a new line. Do NOT combine multiple lines into a single paragraph.
12. IMPORTANT: The translated output must keep every original newline, bullet, and numbering exactly as in the source. Do NOT collapse multiple lines into a paragraph.
"""

PROMPT_TEMPLATE = """Translate the following batch of examination items into {language}.

CRITICAL RULE: PASSAGE/DIRECTIONS HANDLING
Some items in the input are not multiple-choice questions, but rather shared passages, directions, context, or notes.
- You MUST translate ALL of this text exactly as it is. DO NOT OMIT ANYTHING.
- For these items (usually where question_no is 0): put the full text in `english_question` and `translated_question`. Set `english_options` and `translated_options` to `[]`. Set `answer_key`, `english_solution`, and `translated_solution` to `""`.
- If an item contains BOTH a passage and a question, translate EVERYTHING. Do not drop the passage.

MANDATORY: Return a JSON object with a "questions" array. Each element must have ALL of these keys:
  - question_no (integer — 0 if passage)
  - english_passage (always "")
  - translated_passage (always "")
  - english_question (string, the question stem OR the full passage text)
  - translated_question (string, the translated stem OR translated passage)
  - english_options (array of strings for all options, or [] for passages/notes)
  - translated_options (array of strings for all options, or [] for passages/notes)
  - answer_key (string digit, or "" for passages/notes)
  - english_solution (string, or "" for passages/notes)
  - translated_solution (string, or "" for passages/notes)

────────────────────────────────────────────────────────────────────────────────
EXAMPLE INPUT (Batch containing a passage and a question):
DIRECTIONS: SET OF 5 QUESTIONS (Q.46-Q.50):
Read the following passage and answer the questions:
Organized endeavours directed by people responsible for planning... [passage text]

46. According to the passage, the ancient equivalent of the assembly line was used in:
Egypt
China
Venice
Giza
Answer Key: 3
Solution:
VENICE is the correct answer because...

EXAMPLE OUTPUT (for {{language}}=Hindi):
{{
  "questions": [
    {{
      "question_no": 0,
      "english_passage": "",
      "translated_passage": "",
      "english_question": "DIRECTIONS: SET OF 5 QUESTIONS (Q.46-Q.50):\nRead the following passage and answer the questions:\nOrganized endeavours directed by people responsible for planning... [full passage text]",
      "translated_question": "निर्देश: 5 प्रश्नों का सेट (प्र.46-प्र.50):\nनिम्नलिखित गद्यांश पढ़ें और प्रश्नों के उत्तर दें:\nनियोजन के लिए जिम्मेदार लोगों द्वारा निर्देशित संगठित प्रयास... [पूर्ण गद्यांश पाठ]",
      "english_options": [],
      "translated_options": [],
      "answer_key": "",
      "english_solution": "",
      "translated_solution": ""
    }},
    {{
      "question_no": 46,
      "english_passage": "",
      "translated_passage": "",
      "english_question": "46. According to the passage, the ancient equivalent of the assembly line was used in:",
      "translated_question": "गद्यांश के अनुसार, असेंबली लाइन के प्राचीन समतुल्य का उपयोग कहाँ किया गया था:",
      "english_options": ["(1) Egypt", "(2) China", "(3) Venice", "(4) Giza"],
      "translated_options": ["(5) मिस्र", "(6) चीन", "(7) वेनिस", "(8) गीज़ा"],
      "answer_key": "3",
      "english_solution": "VENICE is the correct answer because...",
      "translated_solution": "वेनिस (VENICE) सही उत्तर है क्योंकि..."
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

The ORIGINAL QUESTIONS that you were supposed to translate are provided below. Your output should contain at least {expected_count} items matching the original block.

ORIGINAL QUESTIONS TO TRANSLATE:

{questions_block}

Please fix the JSON so it strictly conforms to the schema. Return ONLY the corrected JSON object — no markdown, no explanation.

Rules reminder:
- "questions" must be a non-empty array.
- Each question must have ALL keys: question_no, english_question, translated_question, english_options (array of all options), translated_options (array of all options), answer_key (digit only), english_solution (non-empty), translated_solution (non-empty).
- translated_solution MUST contain the actual translated explanation text — it must NOT be empty.
- NEVER create dummy, placeholder, or filler questions. Only translate questions that actually exist in the original text.
- If the original block has fewer questions than expected, translate ONLY what exists — do NOT invent extra questions.
- PRESERVE FORMATTING: You must preserve `**bold**`, `__underline__`, `<sup>`, and `<sub>` markers exactly as they appear in the original text. Also preserve any bullet points (`- `) or numbering (`1. `, `2. `) exactly.
- PRESERVE NEWLINES AND PARAGRAPHS: You MUST preserve all newlines (`\n`) and paragraph breaks from the original text. Do NOT combine multiple lines into a single paragraph."""


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



def _call_gemini_json(prompt: str, system_instruction: str,
                      max_output_tokens: int = 65536) -> str:
    """
    Call Gemini with JSON mode enabled. Returns the raw response text.
    Raises ValueError if the model returned no text.
    """
    # Instantiate client per-call to prevent stale connection pool (Broken pipe) issues on long batches
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
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
                # Repair mode: give Gemini the broken JSON + validation error + original text
                prompt = (REPAIR_PROMPT_TEMPLATE
                          .replace("{validation_error}", last_error)
                          .replace("{broken_json}", last_raw)
                          .replace("{questions_block}", questions_block)
                          .replace("{expected_count}", str(len(questions))))

            # ── Step 1: Call Gemini in JSON mode ─────────────────────────────
            raw_json = _call_gemini_json(prompt, sys_inst)
            last_raw = raw_json

            # ── Step 2: Pydantic validation ───────────────────────────────────
            batch = _parse_and_validate(raw_json)

            # ── Step 3: Filter out dummy/placeholder questions ─────────────
            batch.questions = [
                q for q in batch.questions
                if "dummy" not in q.english_question.lower()
                and "placeholder" not in q.english_question.lower()
                and "filler" not in q.english_question.lower()
            ]

            # ── Step 4: Validate question count (lenient) ─────────────────────
            returned = len(batch.questions)
            expected = len(questions)
            if returned < expected:
                # Allow if at least 75% of expected questions are present
                if returned >= max(1, int(expected * 0.75)):
                    print(f"[translator] ⚠️  Got {returned}/{expected} questions "
                          f"(within tolerance) — accepting batch")
                else:
                    raise ValueError(
                        f"Expected at least {expected} questions in JSON output, "
                        f"got {returned}. Too many questions are missing."
                    )

            # ── Step 5: Re-stamp original question_no from extraction ─────────
            # The translator LLM may return different question_no values than
            # what the extractor assigned. Override them positionally so the
            # original document numbering is always preserved.
            for i, tq in enumerate(batch.questions):
                if i < len(questions):
                    original_q_no = questions[i].get("question_no", 0)
                    try:
                        tq.question_no = int(original_q_no)
                    except (ValueError, TypeError):
                        pass  # keep whatever the LLM returned

            # ── Step 6: Return structured data ────────────────────────────────
            print(f"[translator] ✅ Batch validated — "
                  f"{len(batch.questions)} question(s) OK")

            return batch.model_dump()

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

    # Unreachable in practice (max_retries >= 1 always), but satisfies the type contract.
    # docx_builder.build() expects {"questions": []} — never a bare empty dict.
    return {"questions": []}