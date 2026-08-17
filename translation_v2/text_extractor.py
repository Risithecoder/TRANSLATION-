"""
text_extractor.py — Universal question extraction from .docx/.html files.

Approach:
1. Convert .docx → HTML via mammoth (pure Python, no LibreOffice needed)
2. Pre-process HTML → clean plain text (tables → markdown, tags stripped)
3. Smart chunking — split at 'Answer Key' boundaries (never cuts a question)
4. GPT-4o-mini extracts structured questions from each chunk
   (handles ANY question format — numbered, lettered, unnumbered, etc.)

No regex-based question detection. The LLM handles all format variations.
"""

import os
import re
import json
import time
from google import genai
from google.genai import types

# ── Gemini setup ─────────────────────────────────────────────────────────────
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
MODEL = "gemini-2.5-pro"


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1: HTML → Clean Plain Text
# ═════════════════════════════════════════════════════════════════════════════

UNICODE_SUP = {
    '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4', '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9',
    '⁺': '+', '⁻': '-', '⁼': '=', '⁽': '(', '⁾': ')', 'ⁿ': 'n', 'ⁱ': 'i', 'ᵃ': 'a', 'ᵇ': 'b', 'ᶜ': 'c',
    'ᵈ': 'd', 'ᵉ': 'e', 'ᶠ': 'f', 'ᵍ': 'g', 'ʰ': 'h', 'ʲ': 'j', 'ᵏ': 'k', 'ˡ': 'l', 'ᵐ': 'm', 'ᵒ': 'o',
    'ᵖ': 'p', 'ʳ': 'r', 'ˢ': 's', 'ᵗ': 't', 'ᵘ': 'u', 'ᵛ': 'v', 'ʷ': 'w', 'ˣ': 'x', 'ʸ': 'y', 'ᶻ': 'z',
    'ᴬ': 'A', 'ᴮ': 'B', 'ᴰ': 'D', 'ᴱ': 'E', 'ᴳ': 'G', 'ᴴ': 'H', 'ᴵ': 'I', 'ᴶ': 'J', 'ᴷ': 'K', 'ᴸ': 'L',
    'ᴹ': 'M', 'ᴺ': 'N', 'ᴼ': 'O', 'ᴾ': 'P', 'ᴿ': 'R', 'ᵀ': 'T', 'ᵁ': 'U', 'ⱽ': 'V', 'ᵂ': 'W'
}

UNICODE_SUB = {
    '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4', '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9',
    '₊': '+', '₋': '-', '₌': '=', '₍': '(', '₎': ')', 'ₐ': 'a', 'ₑ': 'e', 'ₕ': 'h', 'ᵢ': 'i', 'ⱼ': 'j',
    'ₖ': 'k', 'ₗ': 'l', 'ₘ': 'm', 'ₙ': 'n', 'ₒ': 'o', 'ₚ': 'p', 'ᵣ': 'r', 'ₛ': 's', 'ₜ': 't', 'ᵤ': 'u',
    'ᵥ': 'v', 'ₓ': 'x'
}


def _html_to_plain_text(html_content: str) -> str:
    """
    Convert raw HTML to clean plain text.
    Tables → markdown tables. Lists → bullet items. Tags stripped.
    Preserves internal line breaks from DOCX (soft returns / <br> tags).
    Also splits labeled items (A. B. C.) that appear on a single line.
    """
    from bs4 import BeautifulSoup

    # Convert Unicode superscript/subscript characters to HTML tags
    for char, replacement in UNICODE_SUP.items():
        html_content = html_content.replace(char, f"<sup>{replacement}</sup>")
    for char, replacement in UNICODE_SUB.items():
        html_content = html_content.replace(char, f"<sub>{replacement}</sub>")

    # ── Step 1: Pre-process raw HTML to preserve <br> line breaks ──────────
    # Replace ALL <br> variants at string level before BeautifulSoup parsing.
    # This is more reliable than el.find_all("br") which can miss self-closing
    # tag variants depending on the parser.
    html_content = re.sub(r'<br\s*/?\s*>', '\n', html_content, flags=re.IGNORECASE)

    soup = BeautifulSoup(html_content, "html.parser")
    
    # ── Replace bold/strong and u tags with markdown-style markers ─────────
    for tag in soup.find_all(['strong', 'b']):
        tag.insert_before("**")
        tag.insert_after("**")
        tag.unwrap()
        
    for tag in soup.find_all('u'):
        tag.insert_before("__")
        tag.insert_after("__")
        tag.unwrap()

    for tag in soup.find_all('sup'):
        tag.insert_before("<sup>")
        tag.insert_after("</sup>")
        tag.unwrap()

    for tag in soup.find_all('sub'):
        tag.insert_before("<sub>")
        tag.insert_after("</sub>")
        tag.unwrap()

    parts = []

    for el in soup.children:
        if el.name is None:
            continue
        if el.name == "table":
            rows = el.find_all("tr")
            md_rows = []
            # Track max columns so separator always matches
            max_cols = 0
            raw_row_cells = []
            for row in rows:
                cells = row.find_all(["td", "th"])
                expanded = []
                for c in cells:
                    text = c.get_text(strip=True)
                    try:
                        span = int(c.get("colspan", 1))
                    except (ValueError, TypeError):
                        span = 1
                    expanded.append(text)
                    # Fill extra columns caused by colspan with empty strings
                    for _ in range(span - 1):
                        expanded.append("")
                raw_row_cells.append(expanded)
                if len(expanded) > max_cols:
                    max_cols = len(expanded)
            for row_idx, cols in enumerate(raw_row_cells):
                # Pad rows that are shorter than max_cols
                while len(cols) < max_cols:
                    cols.append("")
                md_rows.append("| " + " | ".join(cols) + " |")
                if row_idx == 0:
                    md_rows.append("|" + "|".join(["---"] * max_cols) + "|")
            parts.append("\n".join(md_rows))
        elif el.name in ("ol", "ul"):
            is_ordered = (el.name == "ol")
            for idx, li in enumerate(el.find_all("li", recursive=False), start=1):
                t = li.get_text(strip=True)
                if t:
                    prefix = f"{idx}. " if is_ordered else "- "
                    parts.append(prefix + t)
        else:
            text = el.get_text(strip=False).strip()
            if text:
                parts.append(text)

    plain = "\n".join(parts)

    # ── Step 2: Split internal labeled items crammed onto one line ─────────
    # Many DOCXs put "A. Item1 B. Item2 C. Item3" on one line with no breaks.
    # Detect lines with 2+ single-letter labels and split them.
    plain = _split_internal_labels(plain)

    return plain


def _split_internal_labels(text: str) -> str:
    """
    Split lines that contain multiple internal labels (A. B. C. etc.)
    onto separate lines. Only activates when 2+ labels are found on the
    same line, so normal text like 'Section A. Introduction' is untouched.
    """
    lines = text.split('\n')
    result = []
    for line in lines:
        # Count single-letter labels A-Z followed by ". " in this line
        labels = re.findall(r'\b[A-Z]\.\s', line)
        if len(labels) >= 2:
            # Insert a newline before each label that appears mid-line
            # (i.e., preceded by non-whitespace text)
            line = re.sub(r'(?<=\S)\s+([A-Z]\.\s)', r'\n\1', line)
        result.append(line)
    return '\n'.join(result)


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2: Smart Chunking — split at Answer Key boundaries
# ═════════════════════════════════════════════════════════════════════════════

def _smart_chunk(plain_text: str, max_chunk_chars: int = 30000) -> list:
    """
    Split plain text into chunks at 'Answer Key' boundaries.
    
    Why: Naive character-based splitting can slice a question in half.
    By splitting only at 'Answer Key' + solution-end boundaries, we 
    guarantee each chunk contains only COMPLETE questions.
    
    If the text is small enough, returns a single chunk.
    """
    if len(plain_text) <= max_chunk_chars:
        return [plain_text]

    lines = plain_text.split("\n")

    # Find all line indices that contain 'Answer Key'
    ak_line_indices = []
    for idx, line in enumerate(lines):
        if re.search(r'answer\s*key', line, re.IGNORECASE):
            ak_line_indices.append(idx)

    if not ak_line_indices:
        # No answer keys found — fall back to simple character-based splitting
        chunks = []
        start = 0
        while start < len(plain_text):
            end = min(start + max_chunk_chars, len(plain_text))
            chunks.append(plain_text[start:end])
            if end >= len(plain_text):
                break
            start = end - 5000  # small overlap for safety
        return chunks

    # Build chunks by grouping consecutive Answer Key blocks
    chunks = []
    current_start_line = 0

    for i, ak_idx in enumerate(ak_line_indices):
        # Check if adding the next AK block would exceed the chunk limit
        # We split AFTER the solution content that follows this Answer Key
        # Find the next AK or end of document
        if i + 1 < len(ak_line_indices):
            next_ak_line = ak_line_indices[i + 1]
        else:
            next_ak_line = len(lines)

        # Calculate current chunk size if we include up to next_ak_line
        candidate_text = "\n".join(lines[current_start_line:next_ak_line])

        if len(candidate_text) > max_chunk_chars and current_start_line < ak_idx:
            # Current chunk is too big — split here (after this AK's solution)
            # Find a good split point: a few lines after this Answer Key
            # (to include the solution that follows it)
            split_line = min(ak_idx + 15, next_ak_line)  # include ~15 lines of solution
            chunk_text = "\n".join(lines[current_start_line:split_line])
            chunks.append(chunk_text)
            current_start_line = split_line

    # Don't forget the remaining content
    if current_start_line < len(lines):
        remaining = "\n".join(lines[current_start_line:])
        if remaining.strip():
            chunks.append(remaining)

    return chunks if chunks else [plain_text]


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3: LLM Extraction — GPT-4o-mini
# ═════════════════════════════════════════════════════════════════════════════

EXTRACTION_PROMPT = """You are an expert document parser. Your ONLY task is to segment the provided text into logical items (questions, passages, directions, context).

ABSOLUTE RULES:
1. DO NOT OMIT, DROP, OR IGNORE ANY TEXT. Every single word from the input MUST appear in one of the extracted items.
2. If the text contains a passage, directions, or notes before the questions, you MUST create a separate item for it.
3. DO NOT SUMMARIZE. Copy the text exactly.

For each item, return a JSON object with:
1. question_no: The question number as an integer. If it is a passage, directions, or unnumbered text, use 0.
2. raw_text: The complete text of the item, including all options, answer keys, and solutions. Use \\n for line breaks.

EXCEPTION FOR MARKDOWN TABLES:
If the text contains a Markdown table (e.g., `| Col1 | Col2 |`), you MUST preserve it exactly as a single line per row. Do NOT break table rows across multiple lines.

TEXT TO SEGMENT:
{chunk}

Return a JSON object strictly in this format: {"questions": [{"question_no": 0, "raw_text": "..."}, ...]}"""


def _sanitize_json_string(raw: str) -> str:
    """
    Fix unescaped control characters inside JSON string values.

    The LLM sometimes outputs literal newlines, tabs, or other control
    characters inside JSON strings (e.g. in raw_text fields).  These are
    invalid JSON and cause 'Unterminated string' errors in json.loads().

    This function walks through the raw JSON character-by-character,
    tracks whether we're inside a string literal, and escapes any
    unescaped control characters found within strings.
    """
    result = []
    in_string = False
    i = 0
    n = len(raw)

    while i < n:
        ch = raw[i]

        if in_string:
            if ch == '\\':
                # Escaped character — take it and the next char as-is
                result.append(ch)
                if i + 1 < n:
                    i += 1
                    result.append(raw[i])
            elif ch == '"':
                # End of string
                result.append(ch)
                in_string = False
            elif ch == '\n':
                result.append('\\n')
            elif ch == '\r':
                result.append('\\r')
            elif ch == '\t':
                result.append('\\t')
            elif ord(ch) < 0x20:
                # Other control characters — escape as unicode
                result.append(f'\\u{ord(ch):04x}')
            else:
                result.append(ch)
        else:
            result.append(ch)
            if ch == '"':
                in_string = True

        i += 1

    return ''.join(result)


def _extract_chunk(chunk: str, chunk_idx: int, total_chunks: int,
                   max_retries: int = 3) -> list:
    """Send one chunk to Gemini and parse the response."""
    prompt = EXTRACTION_PROMPT.replace("{chunk}", chunk)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                    max_output_tokens=65536,
                )
            )
            raw = response.text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            # Sanitize: fix unescaped newlines/control chars inside JSON strings
            # The LLM sometimes outputs literal newlines inside JSON string values
            # which causes "Unterminated string" errors in json.loads()
            raw = _sanitize_json_string(raw)

            data = json.loads(raw)
            if isinstance(data, list):
                qs = data
            elif isinstance(data, dict):
                qs = data.get("questions", [])
                if not isinstance(qs, list):
                    qs = [qs]
            else:
                qs = []

            extracted = []
            for q in qs:
                if not isinstance(q, dict):
                    continue
                raw_text = q.get("raw_text")
                if raw_text is None:
                    raw_text = ""
                raw_text = str(raw_text).strip()
                if not raw_text:
                    continue
                
                try:
                    q_no = int(q.get("question_no", 0))
                except (ValueError, TypeError):
                    q_no = 0


                
                
                extracted.append({
                    "question_no": q_no,
                    "raw_text": raw_text
                })

            print(f"[text_extractor] Chunk {chunk_idx}/{total_chunks}: "
                  f"{len(extracted)} question(s) extracted")

            return extracted
        except Exception as e:
            print(f"[text_extractor] Chunk {chunk_idx} attempt "
                  f"{attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time.sleep(2 * attempt)
            else:
                raise
    return []


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4: Deduplication
# ═════════════════════════════════════════════════════════════════════════════

def _deduplicate(questions: list) -> list:
    """Remove duplicate questions based on normalized text content.
    Preserves the original document question numbers extracted by the LLM.
    Also merges orphaned Answer Key / Solution fragments back into the
    preceding question so they don't appear as standalone q_no=0 blocks.
    """

    # ── Pass 0: Merge orphaned Answer Key / Solution fragments ────────────
    # Sometimes the LLM extraction separates the Answer Key / Solution from
    # its parent question into a standalone q_no=0 item.  Detect these and
    # append their text back into the previous question's raw_text.
    merged = []
    for q in questions:
        raw = q.get("raw_text", "").strip()
        q_no = q.get("question_no", 0)

        # An orphan is a q_no==0 item that starts with "Answer Key" or "Solution"
        is_orphan = (
            q_no == 0
            and merged  # there is a preceding question to merge into
            and re.match(r'^(Answer\s*Key|Solution)\s*:', raw, re.IGNORECASE)
        )

        if is_orphan:
            # Append to the previous question's raw_text
            merged[-1]["raw_text"] = merged[-1]["raw_text"].rstrip() + "\n" + raw
        else:
            merged.append(q)

    # ── Pass 1: Deduplicate ───────────────────────────────────────────────
    seen = set()
    unique = []
    last_q_no = 0
    for q in merged:
            # Normalize: strip whitespace for comparison
            norm = re.sub(r"\s+", "", q.get("raw_text", ""))
            if not norm or len(norm) < 5:
                continue
            if norm not in seen:
                seen.add(norm)
                
                q_no = q.get("question_no")
                
                # Default to 0 (which means unnumbered passage/directions)
                if not q_no:
                    q_no = 0
                    q["question_no"] = 0
                
                # Ensure it's an int and update our running counter
                try:
                    q_no_int = int(q_no)
                    if q_no_int > last_q_no:
                        last_q_no = q_no_int
                except (ValueError, TypeError):
                    pass
                    
                # Enforce the question number prefix on the raw_text so it appears in the UI
                # ONLY if it's an actual numbered question (>0)
                if q["question_no"] > 0:
                    raw = str(q.get("raw_text", "")).strip()
                    if not re.match(r'^\d+\.', raw):
                        # E.g. "What is..." -> "46. What is..."
                        q["raw_text"] = f"{q['question_no']}. {raw}"
                    else:
                        # Force the prefix to exactly match the extracted/assigned question_no
                        q["raw_text"] = re.sub(r'^\d+\.', f"{q['question_no']}.", raw, count=1)
                    
                unique.append(q)
    return unique


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC API — main entry point
# ═════════════════════════════════════════════════════════════════════════════

def extract_questions(file_path: str) -> list:
    """
    Extract questions from .docx or .html files.

    Pipeline:
    1. Convert to HTML via mammoth (for .docx) or read directly (.html)
    2. Pre-process HTML → clean plain text (strips tags, tables → markdown)
    3. Smart-chunk at Answer Key boundaries (never splits a question)
    4. GPT-4o-mini extracts structured questions from each chunk
    5. Deduplicate and re-number

    Works with ANY question format — numbered, lettered, unnumbered, etc.

    Args:
        file_path: Path to the .docx or .html file

    Returns:
        List of dicts: [{"question_no": int, "raw_text": str}, ...]
    """
    ext = os.path.splitext(file_path)[-1].lower()

    # Step 1: Get HTML content
    if ext == ".docx":
        import mammoth
        print(f"[text_extractor] Converting .docx → HTML via mammoth...")
        with open(file_path, "rb") as f:
            html_content = mammoth.convert_to_html(f, style_map="u => u").value
    elif ext in (".html", ".htm"):
        print(f"[text_extractor] Reading HTML file...")
        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    if not html_content.strip():
        raise ValueError("Empty document — no content to extract.")

    # Step 2: HTML → clean plain text
    print(f"[text_extractor] Pre-processing HTML → plain text...")
    plain_text = _html_to_plain_text(html_content)
    print(f"[text_extractor] Plain text: {len(plain_text)} chars "
          f"(reduced from {len(html_content)} HTML chars)")

    # Step 3: Smart chunking
    chunks = _smart_chunk(plain_text)
    print(f"[text_extractor] Split into {len(chunks)} chunk(s)")

    # Step 4: Extract questions from each chunk via LLM (PARALLEL)
    import concurrent.futures
    
    all_questions = []
    chunk_results = {}
    
    # Use max_workers=10 so we can process up to 10 chunks simultaneously
    last_error = None
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, max(1, len(chunks)))) as executor:
        futures = {}
        for idx, chunk in enumerate(chunks, 1):
            future = executor.submit(_extract_chunk, chunk, idx, len(chunks))
            futures[future] = idx
            time.sleep(0.7)
        
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                qs = future.result()
                chunk_results[idx] = qs
            except Exception as e:
                print(f"[text_extractor] Chunk {idx} extraction failed completely: {e}")
                last_error = e
                chunk_results[idx] = []
                
    # Combine results in the correct original order
    for idx in sorted(chunk_results.keys()):
        all_questions.extend(chunk_results[idx])

    # Step 5: Deduplicate
    questions = _deduplicate(all_questions)

    if not questions:
        if last_error:
            raise RuntimeError(f"No questions could be extracted from the document. Reason: {last_error}")
        raise RuntimeError("No questions could be extracted from the document.")

    print(f"[text_extractor] ✅ Total: {len(questions)} unique question(s)")
    return questions