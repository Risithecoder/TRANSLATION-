import os
import re
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

ABSOLUTE STRUCTURE RULE — FOLLOW THIS EXACTLY FOR EVERY QUESTION:
The output for EACH question must have exactly this block order:
  BLOCK 1: Question number + English question body (may be multi-line)
  BLOCK 2: "{language}:" label on its own line
  BLOCK 3: Translated question body
  BLOCK 4: (1) through (4) — original English options
  BLOCK 5: (5) through (8) — translated options
  BLOCK 6: "Answer Key: X" — appears exactly ONCE
  BLOCK 7: "Solution:" label
  BLOCK 8: English solution text
  BLOCK 9: "{language}:" label on its own line
  BLOCK 10: Translated solution text

Do NOT put options before the "{language}:" translated question body.
Do NOT duplicate "Answer Key:" lines.
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
{language}:
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
{language}:
नई दिल्ली भारत की राजधानी है。
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
    return len(re.findall(r'Answer\s*Key\s*:', text, re.IGNORECASE))


# ═════════════════════════════════════════════════════════════════════════════
# POST-PROCESSOR — Enforce the correct structure deterministically
# ═════════════════════════════════════════════════════════════════════════════

def _post_process_output(raw_output: str, language: str) -> str:
    """
    Parse the raw LLM translation output and enforce the correct structure.
    
    Handles the common LLM drift where it outputs:
      English question → options 1-4 → Answer Key → Solution →
      Language: → translated question → options 1-4 + 5-8 → Answer Key → Solution
    
    And restructures into the correct format:
      English question → Language: → translated question →
      options 1-4 → options 5-8 → Answer Key → Solution English →
      Language: → Solution translated
    """
    lines = raw_output.split("\n")
    
    # Split output into individual question blocks
    q_start_indices = []
    for i, line in enumerate(lines):
        if re.match(r'^\d+\.\s+', line.strip()):
            q_start_indices.append(i)
    
    if not q_start_indices:
        return raw_output
    
    question_blocks = []
    for idx, start in enumerate(q_start_indices):
        end = q_start_indices[idx + 1] if idx + 1 < len(q_start_indices) else len(lines)
        block_lines = lines[start:end]
        question_blocks.append(block_lines)
    
    lang_pattern = re.compile(rf'^{re.escape(language)}\s*:\s*$', re.IGNORECASE)
    
    restructured_blocks = []
    
    for block_lines in question_blocks:
        try:
            restructured = _restructure_question_block(block_lines, language, lang_pattern)
            restructured_blocks.append(restructured)
        except Exception as e:
            print(f"[translator] Post-process warning: {e}")
            restructured_blocks.append("\n".join(block_lines))
    
    return "\n\n".join(restructured_blocks)


def _classify_line(line: str, lang_pattern: re.Pattern) -> str:
    """Classify a line into a category."""
    stripped = line.strip()
    if not stripped:
        return "empty"
    if lang_pattern.match(stripped):
        return "lang_label"
    if re.match(r'^Answer\s*Key\s*:', stripped, re.IGNORECASE):
        return "answer_key"
    # Match 'Solution:' or 'Solution' as a standalone label (not 'Solution: some text')
    if re.match(r'^Solution\s*:\s*$', stripped, re.IGNORECASE) or stripped.lower() == 'solution':
        return "solution_label"
    if re.match(r'^\(\d+\)\s', stripped):
        return "option"
    return "text"


def _restructure_question_block(block_lines: list, language: str, 
                                 lang_pattern: re.Pattern) -> str:
    """
    Restructure a single question block into the correct format.
    
    Uses a state-machine approach to scan all lines and collect components,
    then reassembles them in the correct order.
    """
    # Remove trailing empty lines
    while block_lines and not block_lines[-1].strip():
        block_lines = block_lines[:-1]
    
    if not block_lines:
        return ""
    
    # Classify every line
    classifications = [_classify_line(line, lang_pattern) for line in block_lines]
    
    # Count language labels and answer keys to detect drift
    lang_label_count = classifications.count("lang_label")
    ak_count = classifications.count("answer_key")
    
    # Find positions of lang labels
    lang_positions = [i for i, c in enumerate(classifications) if c == "lang_label"]
    ak_positions = [i for i, c in enumerate(classifications) if c == "answer_key"]
    sol_positions = [i for i, c in enumerate(classifications) if c == "solution_label"]
    
    # ── DETECT FORMAT: Is this already correct or does it need restructuring? ──
    # 
    # CORRECT FORMAT: lang_label comes BEFORE first option
    # DRIFTED FORMAT: options come BEFORE first lang_label
    
    first_option_pos = None
    for i, c in enumerate(classifications):
        if c == "option":
            first_option_pos = i
            break
    
    # If no lang labels at all, return as-is
    if not lang_positions:
        return "\n".join(block_lines)
    
    first_lang_pos = lang_positions[0]
    
    # Check if already correct: lang label before first option, and only 1 answer key
    if (first_option_pos is not None and first_lang_pos < first_option_pos 
        and ak_count <= 1 and lang_label_count == 2):
        # Already in correct format — return as-is
        return "\n".join(block_lines)
    
    # ── RESTRUCTURING NEEDED ──
    # The LLM output is in drifted format. Parse into components.
    
    # Components to extract:
    eng_question_body = []    # English question text (no options)
    trans_question_body = []  # Translated question text (no options)
    eng_options = []          # Options (1)-(4)
    trans_options = []        # Options (5)-(8)
    answer_key = ""           # Answer Key line
    eng_solution = []         # English solution lines
    trans_solution = []       # Translated solution lines
    
    # Strategy: Walk through all lines with a state machine
    # States:
    #   eng_q       — English question body
    #   eng_opts    — English options (1-4)
    #   eng_ak      — After first Answer Key (still in English section)
    #   eng_sol     — English solution text (first section)
    #   trans_q     — Translated question body
    #   trans_opts  — Translated options section
    #   trans_ak    — After second (duplicate) Answer Key
    #   trans_sol_pending — Saw second Solution: label, waiting for lang_label or text
    #   trans_sol   — Translated solution text
    
    state = "eng_q"
    
    for i, line in enumerate(block_lines):
        stripped = line.strip()
        cls = classifications[i]
        
        if cls == "empty":
            continue
        
        if state == "eng_q":
            if cls == "option":
                state = "eng_opts"
                opt_m = re.match(r'^\((\d+)\)', stripped)
                if opt_m:
                    opt_num = int(opt_m.group(1))
                    if opt_num <= 4:
                        eng_options.append(stripped)
                    else:
                        trans_options.append(stripped)
            elif cls == "lang_label":
                state = "trans_q"
            elif cls == "answer_key":
                answer_key = stripped
                state = "eng_ak"
            elif cls == "solution_label":
                state = "eng_sol"
            else:
                eng_question_body.append(stripped)

        elif state == "eng_opts":
            if cls == "option":
                opt_m = re.match(r'^\((\d+)\)', stripped)
                if opt_m:
                    opt_num = int(opt_m.group(1))
                    if opt_num <= 4:
                        eng_options.append(stripped)
                    else:
                        trans_options.append(stripped)
            elif cls == "answer_key":
                answer_key = stripped
                state = "eng_ak"
            elif cls == "lang_label":
                state = "trans_q"
            elif cls == "solution_label":
                state = "eng_sol"
            else:
                eng_question_body.append(stripped)

        elif state == "eng_ak":
            if cls == "solution_label":
                state = "eng_sol"
            elif cls == "lang_label":
                state = "trans_q"
            elif cls == "option":
                opt_m = re.match(r'^\((\d+)\)', stripped)
                if opt_m:
                    opt_num = int(opt_m.group(1))
                    if opt_num <= 4:
                        eng_options.append(stripped)
                    else:
                        trans_options.append(stripped)
            else:
                eng_solution.append(stripped)

        elif state == "eng_sol":
            if cls == "lang_label":
                state = "trans_q"
            elif cls == "answer_key":
                pass  # duplicate answer key — ignore
            elif cls == "solution_label":
                pass  # duplicate solution label — ignore
            else:
                eng_solution.append(stripped)

        elif state == "trans_q":
            if cls == "option":
                state = "trans_opts"
                opt_m = re.match(r'^\((\d+)\)', stripped)
                if opt_m:
                    opt_num = int(opt_m.group(1))
                    if opt_num <= 4:
                        # Duplicate (1-4) in translated section — skip if already collected
                        if not eng_options:
                            eng_options.append(stripped)
                    else:
                        trans_options.append(stripped)
            elif cls == "answer_key":
                # Duplicate answer key in translated section — skip
                state = "trans_ak"
            elif cls == "solution_label":
                # 'Solution:' encountered while in trans_q:
                # This is the SECOND Solution: label (translated section).
                # The translated solution text comes next.
                state = "trans_sol_pending"
            elif cls == "lang_label":
                # A second language label while in trans_q — 
                # means the LLM put Language: before the translated solution.
                state = "trans_sol"
            else:
                trans_question_body.append(stripped)

        elif state == "trans_opts":
            if cls == "option":
                opt_m = re.match(r'^\((\d+)\)', stripped)
                if opt_m:
                    opt_num = int(opt_m.group(1))
                    if opt_num <= 4:
                        if not eng_options:
                            eng_options.append(stripped)
                    else:
                        trans_options.append(stripped)
            elif cls == "answer_key":
                state = "trans_ak"
            elif cls == "solution_label":
                # Second Solution: label in translated section
                state = "trans_sol_pending"
            elif cls == "lang_label":
                state = "trans_sol"
            else:
                trans_question_body.append(stripped)

        elif state == "trans_ak":
            if cls == "solution_label":
                # Second Solution: label after duplicate answer key
                state = "trans_sol_pending"
            elif cls == "lang_label":
                state = "trans_sol"
            else:
                # Some LLMs place translated solution directly here without a label
                trans_solution.append(stripped)

        elif state == "trans_sol_pending":
            # We saw the second 'Solution:' label.
            # Now we wait for either:
            #   - a lang_label (Language:) → THEN collect translated solution
            #   - plain text → this IS the translated solution already (no lang label separator)
            if cls == "lang_label":
                # e.g. "Solution:\nLanguage:\n<translated>" — now collect trans solution
                state = "trans_sol"
            elif cls == "answer_key":
                pass  # skip duplicate
            elif cls == "solution_label":
                pass  # skip duplicate
            else:
                # The solution text came directly after Solution: without a lang label
                # Treat as translated solution
                trans_solution.append(stripped)
                state = "trans_sol"

        elif state == "trans_sol":
            if cls == "lang_label":
                # Skip — we are already in translated solution
                pass
            elif cls == "answer_key":
                pass  # skip duplicate
            elif cls == "solution_label":
                pass  # skip duplicate
            else:
                trans_solution.append(stripped)
    
    # ── Handle case where trans options are (1)-(4) renumbered ──
    # If we have no (5)-(8) options but the translated section had (1)-(4),
    # AND those differ from English options, renumber them as (5)-(8)
    if not trans_options and eng_options:
        # Check if there were translated-language (1)-(4) we skipped
        # Re-scan the trans section for options
        trans_section_start = first_lang_pos + 1 if lang_positions else 0
        second_opts_1_4 = []
        for i in range(trans_section_start, len(block_lines)):
            stripped = block_lines[i].strip()
            opt_m = re.match(r'^\((\d+)\)\s+(.*)', stripped)
            if opt_m:
                opt_num = int(opt_m.group(1))
                if opt_num <= 4:
                    second_opts_1_4.append(stripped)
        
        # If these are different from eng_options, they're translated options
        if second_opts_1_4 and len(second_opts_1_4) == len(eng_options):
            different = any(a != b for a, b in zip(eng_options, second_opts_1_4))
            if different:
                for opt in second_opts_1_4:
                    m = re.match(r'^\((\d+)\)(.*)', opt)
                    if m:
                        new_num = int(m.group(1)) + 4
                        trans_options.append(f"({new_num}){m.group(2)}")
    
    # ── Reassemble in correct order ──
    result_lines = []
    
    # Block 1: English question body
    result_lines.extend(eng_question_body)
    
    # Block 2: Language label
    result_lines.append(f"{language}:")
    
    # Block 3: Translated question body
    result_lines.extend(trans_question_body)
    
    # Block 4: English options (1-4)
    result_lines.extend(eng_options)
    
    # Block 5: Translated options (5-8)
    result_lines.extend(trans_options)
    
    # Block 6: Answer Key (once)
    if answer_key:
        result_lines.append(answer_key)
    
    # Block 7-8: Solution + English solution
    if eng_solution:
        result_lines.append("Solution:")
        result_lines.extend(eng_solution)
    
    # Block 9-10: Language label + Translated solution
    if trans_solution:
        result_lines.append(f"{language}:")
        result_lines.extend(trans_solution)
    
    return "\n".join(result_lines)


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

            # ── POST-PROCESS: Enforce correct structure ──
            result = _post_process_output(result, language)

            return result
        except Exception as e:
            print(f"[translator] Batch attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time.sleep(3 * attempt)
            else:
                raise RuntimeError(f"Batch translation failed after {max_retries} attempts: {e}")
    return ""