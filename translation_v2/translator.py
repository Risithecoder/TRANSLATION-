import os
import time
from google import genai
from google.genai import types

# ── Gemini setup ──────────────────────────────────────────────────────────────
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
MODEL  = "gemini-2.5-pro"

# ── Cache store — one cache per language, TTL 24h ─────────────────────────────
_cache_store: dict = {}

SYSTEM_INSTRUCTION = """You are a high-precision examination translator with deep expertise in Indian government competitive exams. You translate structured English question papers with absolute fidelity to structure, formatting, and meaning.

CRITICAL NON-NEGOTIABLE FORMATTING RULES:
1. Every single question in the batch MUST follow this exact output structure:
<Full question text in English exactly as provided>
{language}:
<Translated question text>
(1) <MCQ Option 1 in English>
(2) <MCQ Option 2 in English>
(3) <MCQ Option 3 in English>
(4) <MCQ Option 4 in English>
(5) <MCQ Option 1 translated>
(6) <MCQ Option 2 translated>
(7) <MCQ Option 3 translated>
(8) <MCQ Option 4 translated>
Answer Key: <number only, e.g., 2>
Solution:
<Full solution in English>
{language}:
<Translated solution>

2. The English text must ALWAYS come first, followed by the `{language}:` label, followed by the translation. This applies to the question body and the solution.
3. For options, you MUST output 2N options: original English options (1-4) followed by translated options (5-8).
4. Do NOT skip any questions in the batch. Do NOT change the formatting midway. Maintain absolute consistency for EVERY question.
"""

PROMPT_TEMPLATE = """OBJECTIVE
Translate the following batch of questions into {language}.
Each question is provided as complete raw text exactly as it appears in the exam paper.

CRITICAL RULE: QUESTION BODY STRUCTURE DETECTION
Many government exam questions contain internal statements or items listed inside the question_body itself, labelled with letters or Roman numerals such as:
  A., B., C., D. or I., II., III. or 1., 2., 3. (internal items in question body)
These are PART OF THE QUESTION BODY — not answer options.

MANDATORY DETECTION STEP
Before translating each question, scan for statements labelled A./B./C. inside the question stem.
If detected:
  - Preserve all internal labels exactly.
  - Each labelled statement on its own line.
  - The options field is separate — internal labels are not options.

NON-NEGOTIABLE RULES
1. Translate only the content provided. Do not add, remove, or reinterpret.
2. Preserve line breaks, numbering, bullets, labels, and punctuation exactly.
3. Do not alter equations, formulas, symbols, units, dates, codes, or placeholders.
4. Translate or transliterate ALL English words into {language} script. No English letters except preserved labels.

────────────────────────────────────────────────────────────────────────────────
EXAMPLE INPUT:
1. What is the capital of India?
(1) Mumbai
(2) New Delhi
(3) Chennai
(4) Kolkata
Answer Key: 2
Solution:
New Delhi is the capital of India.

EXAMPLE OUTPUT (for {language}=Hindi):
1. What is the capital of India?
Hindi:
भारत की राजधानी क्या है?
(1) Mumbai
(2) New Delhi
(3) Chennai
(4) Kolkata
(5) मुंबई
(6) नई दिल्ली
(7) चेन्नई
(8) कोलकाता
Answer Key: 2
Solution:
New Delhi is the capital of India.
Hindi:
नई दिल्ली भारत की राजधानी है।
────────────────────────────────────────────────────────────────────────────────
QUESTIONS TO TRANSLATE:

{questions_block}"""


def _format_questions_block(questions: list) -> str:
    blocks = []
    for q in questions:
        raw = q.get("raw_text", "").strip()
        if raw:
            blocks.append(raw)
    return "\n\n".join(blocks)


def _count_answer_keys(text: str) -> int:
    """Count Answer Key lines in the translated output."""
    import re
    return len(re.findall(r'Answer\s*Key\s*:', text, re.IGNORECASE))


def translate_batch(questions: list, language: str, max_retries: int = 3) -> str:
    if not questions:
        return ""
    questions_block = _format_questions_block(questions)
    full_prompt = (PROMPT_TEMPLATE
        .replace("{language}", language)
        .replace("{questions_block}", questions_block))
    
    sys_inst = SYSTEM_INSTRUCTION.replace("{language}", language)
    expected_count = len(questions)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=[types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=full_prompt)]
                )],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=65536,
                    system_instruction=sys_inst,
                )
            )
            if response.text is None:
                finish_reason = "unknown"
                try:
                    finish_reason = str(response.candidates[0].finish_reason)
                except Exception:
                    pass
                raise ValueError(f"Gemini returned no text (finish_reason: {finish_reason}).")

            result = response.text.strip()

            # Validate: check that the output contains the expected number of questions
            ak_count = _count_answer_keys(result)
            if ak_count < expected_count:
                print(f"[translator] WARNING: Expected {expected_count} questions but output has {ak_count} Answer Keys. "
                      f"Attempt {attempt}/{max_retries}.")
                if attempt < max_retries:
                    time.sleep(3 * attempt)
                    continue  # Retry
                else:
                    print(f"[translator] Proceeding with partial output ({ak_count}/{expected_count}).")

            return result
        except Exception as e:
            print(f"[translator] Batch attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time.sleep(3 * attempt)
            else:
                raise RuntimeError(f"Batch translation failed after {max_retries} attempts: {e}")
    return ""