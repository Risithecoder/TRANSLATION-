import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

try:
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[types.Content(role="user", parts=[types.Part.from_text(text="Say 'API is working'")])]
    )
    print("SUCCESS: The API key is working. Response:", response.text)
except Exception as e:
    print("FAILED: API call failed.")
    print("Error:", e)
