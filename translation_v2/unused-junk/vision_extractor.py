import os
import json
import base64
import time
from google import genai
from google.genai import types

# ── Gemini setup ─────────────────────────────────────────────────────────────
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
MODEL = "gemini-2.5-pro"

# ── Response schema ───────────────────────────────────────────────────────────
# Gemini will fill this structure for every batch of images
RESPONSE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "question_no":  {"type": "integer"},
            "question":     {"type": "string"},
            "options": {
                "type": "object",
                "properties": {
                    "A": {"type": "string"},
                    "B": {"type": "string"},
                    "C": {"type": "string"},
                    "D": {"type": "string"},
                    "E": {"type": "string"}
                }
            },
            "answer": {"type": "string"},
            "solution": {"type": "string"}
        },
        "required": ["question_no", "question"]
    }
}

EXTRACTION_PROMPT = """You are an expert at reading Indian competitive exam question papers.

These images are consecutive pages from an exam paper. Extract every MCQ question you can see.

Rules:
- Extract question number, full question text, all options (A/B/C/D/E if present), answer key, and solution if visible.
- If the answer key is not shown on these pages, leave "answer" as an empty string "".
- If no solution is present, leave "solution" as an empty string "".
- Keep all text exactly as written — do not paraphrase or correct anything.
- Extract the FULL solution text including all option explanations and any "Further Insights" section.
- If a question or solution spans across two pages, combine it fully into one entry.
- Ignore headers, footers, watermarks, and page numbers.
- CRITICAL: Do NOT ignore DIRECTIONS or reading comprehension passages. 
- Instead of attaching the passage to a question, extract it as its OWN STANDALONE ITEM in the JSON array.
- For these standalone passage items, set `question_no` to 0, put the passage text in `question`, and leave `options`, `answer`, and `solution` empty.
- Ignore any marking scheme symbols (X marks, tick marks, etc.).
- If options are labelled 1/2/3/4 instead of A/B/C/D, map them: 1→A, 2→B, 3→C, 4→D.
- For the answer field: if the answer is given as a letter (A/B/C/D) use that. If given as a number (1/2/3/4), convert to letter.
- Options may go up to E — extract all that are present.

Return a JSON array of all questions found across all provided images in order."""


def _encode_image(image_path: str) -> str:
    """Read an image file and return base64 encoded string."""
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def extract_questions(image_paths: list, max_retries: int = 3) -> list:
    """
    Send a batch of page images to Gemini and extract structured questions.

    Args:
        image_paths (list): List of JPEG image file paths (one per page).
        max_retries (int): Number of retry attempts on failure.

    Returns:
        list: List of question dicts with keys: question_no, question, options, answer.
    """
    # Build the content parts — images first, then the prompt
    parts = []
    for img_path in image_paths:
        encoded = _encode_image(img_path)
        parts.append(
            types.Part.from_bytes(
                data=base64.b64decode(encoded),
                mime_type="image/jpeg"
            )
        )
    parts.append(types.Part.from_text(text=EXTRACTION_PROMPT))

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=[types.Content(role="user", parts=parts)],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=RESPONSE_SCHEMA,
                    temperature=0.1,    # Low temperature for consistent extraction
                )
            )

            raw = response.text.strip()

            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            questions = json.loads(raw)

            # Validate it's a list
            if not isinstance(questions, list):
                raise ValueError("Response is not a JSON array.")

            # Sanitise each question
            cleaned = []
            for q in questions:
                cleaned.append({
                    "question_no": int(q.get("question_no", 0)),
                    "question":    str(q.get("question", "")).strip(),
                    "options": {
                        "A": str(q.get("options", {}).get("A", "")).strip(),
                        "B": str(q.get("options", {}).get("B", "")).strip(),
                        "C": str(q.get("options", {}).get("C", "")).strip(),
                        "D": str(q.get("options", {}).get("D", "")).strip(),
                        "E": str(q.get("options", {}).get("E", "")).strip(),
                    },
                    "answer":   str(q.get("answer", "")).strip().upper(),
                    "solution": str(q.get("solution", "")).strip(),
                })

            return cleaned

        except Exception as e:
            print(f"[vision_extractor] Attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time.sleep(3 * attempt)   # backoff: 3s, 6s
            else:
                raise RuntimeError(
                    f"Vision extraction failed after {max_retries} attempts: {e}"
                )

    return []