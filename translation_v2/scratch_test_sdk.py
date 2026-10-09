import os
from dotenv import load_dotenv
load_dotenv()
from google import genai
from google.genai import types
from pydantic import BaseModel

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

class ExtractedQuestion(BaseModel):
    question_no: int
    raw_text: str

class ExtractionResult(BaseModel):
    questions: list[ExtractedQuestion]

def test():
    print("Testing generate_content...")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[types.Content(role="user", parts=[types.Part.from_text(text="Extract question 1: What is 2+2? Question 2: What is 3+3?")])],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ExtractionResult,
            temperature=0.0,
        )
    )
    print("Has parsed:", hasattr(response, "parsed"))
    if hasattr(response, "parsed"):
        print("Parsed type:", type(response.parsed))
        print("Parsed:", response.parsed)

if __name__ == "__main__":
    test()
