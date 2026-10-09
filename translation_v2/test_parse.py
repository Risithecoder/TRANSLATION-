import json
from google import genai
from google.genai import types
from pydantic import BaseModel
import os

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
MODEL = "gemini-2.5-flash"

class ExtractedQuestion(BaseModel):
    question_no: int
    raw_text: str

class ExtractionResult(BaseModel):
    questions: list[ExtractedQuestion]

prompt = "Extract 2 dummy questions"
response = client.models.generate_content(
    model=MODEL,
    contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=ExtractionResult,
        temperature=0.0,
    )
)
print("Raw:")
print(response.text)
try:
    print("Parsed JSON:")
    print(json.loads(response.text))
except Exception as e:
    print("JSON Error:", e)

