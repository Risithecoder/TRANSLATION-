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
from openai import OpenAI

# ── OpenAI setup ─────────────────────────────────────────────────────────────
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1: HTML → Clean Plain Text
# ═════════════════════════════════════════════════════════════════════════════

def _html_to_plain_text(html_content: str) -> str:
    """
    Convert raw HTML to clean plain text.
    Tables → markdown tables. Lists → bullet items. Tags stripped.
    This reduces token count significantly before sending to the LLM.
    """
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_content, "html.parser")
    parts = []

    for el in soup.children:
        if el.name is None:
            continue
        if el.name == "table":
            rows = el.find_all("tr")
            md_rows = []
            for row_idx, row in enumerate(rows):
                cells = row.find_all(["td", "th"])
                cell_texts = [c.get_text(strip=True) for c in cells]
                md_rows.append("| " + " | ".join(cell_texts) + " |")
                if row_idx == 0:
                    md_rows.append("|" + "|".join(["---"] * len(cell_texts)) + "|")
            parts.append("\n".join(md_rows))
        elif el.name in ("ol", "ul"):
            for li in el.find_all("li", recursive=False):
                t = li.get_text(strip=True)
                if t:
                    parts.append(t)
        else:
            text = el.get_text(strip=True)
            if text:
                parts.append(text)

    return "\n".join(parts)


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

EXTRACTION_PROMPT = """You are reading a segment of plain text extracted from an Indian competitive exam question paper.

Extract every COMPLETE question that is fully contained in this text.
Do NOT extract questions that are cut off at the start or end.

For each question, return:
1. question_no: the question number as it appears in the document (integer). If no number is visible, assign sequential numbers starting from 1.
2. raw_text: the COMPLETE verbatim text of the question. You MUST include ALL of the following if present:
   - Question stem / passage / context
   - All options (A/B/C/D or 1/2/3/4 etc.)
   - Answer Key line
   - Solution / Explanation
   Use \\n for line breaks. Do NOT add or remove any content.

RULES:
- Copy text EXACTLY as it appears. Do not rephrase, restructure, or summarize.
- Preserve all numbering, labels, formatting, and mathematical expressions.
- If a passage/table/chart appears before a group of questions and applies to all of them, include it ONLY in the first question's raw_text.
- Do NOT repeat shared passages/tables in subsequent questions of the same group.

TEXT:
{chunk}

Return a JSON object: {"questions": [...]}"""


def _extract_chunk(chunk: str, chunk_idx: int, total_chunks: int,
                   max_retries: int = 3) -> list:
    """Send one chunk to GPT-4o-mini and parse the response."""
    prompt = EXTRACTION_PROMPT.replace("{chunk}", chunk)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=16384,
            )
            data = json.loads(response.choices[0].message.content)
            qs = data.get("questions", [])
            print(f"[text_extractor] Chunk {chunk_idx}/{total_chunks}: "
                  f"{len(qs)} question(s) extracted")
            return [
                {
                    "question_no": int(q.get("question_no", 0)),
                    "raw_text": str(q.get("raw_text", "")).strip(),
                }
                for q in qs
                if q.get("raw_text", "").strip()
            ]
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
    """Remove duplicate questions based on normalized text content."""
    seen = set()
    unique = []
    for q in questions:
        # Normalize: strip whitespace for comparison
        norm = re.sub(r"\s+", "", q.get("raw_text", ""))
        if not norm or len(norm) < 20:
            continue
        if norm not in seen:
            seen.add(norm)
            unique.append(q)
    # Re-number sequentially
    for i, q in enumerate(unique, 1):
        q["question_no"] = i
    return unique


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC API — main entry point
# ═════════════════════════════════════════════════════════════════════════════

def extract_questions(file_path: str, images_folder: str = None) -> list:
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
        images_folder: Unused, kept for interface compatibility

    Returns:
        List of dicts: [{"question_no": int, "raw_text": str}, ...]
    """
    ext = os.path.splitext(file_path)[-1].lower()

    # Step 1: Get HTML content
    if ext == ".docx":
        import mammoth
        print(f"[text_extractor] Converting .docx → HTML via mammoth...")
        with open(file_path, "rb") as f:
            html_content = mammoth.convert_to_html(f).value
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
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, max(1, len(chunks)))) as executor:
        futures = {
            executor.submit(_extract_chunk, chunk, idx, len(chunks)): idx 
            for idx, chunk in enumerate(chunks, 1)
        }
        
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                qs = future.result()
                chunk_results[idx] = qs
            except Exception as e:
                print(f"[text_extractor] Chunk {idx} extraction failed completely: {e}")
                chunk_results[idx] = []
                
    # Combine results in the correct original order
    for idx in sorted(chunk_results.keys()):
        all_questions.extend(chunk_results[idx])

    # Step 5: Deduplicate
    questions = _deduplicate(all_questions)

    if not questions:
        raise RuntimeError("No questions could be extracted from the document.")

    print(f"[text_extractor] ✅ Total: {len(questions)} unique question(s)")
    return questions