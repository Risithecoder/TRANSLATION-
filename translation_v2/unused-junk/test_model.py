import os
from dotenv import load_dotenv
from google import genai

# Load env variables from the current directory (.env)
load_dotenv()
try:
    print("Testing Gemini API with key:", os.environ.get('GEMINI_API_KEY')[:10] + "...")
    client = genai.Client(api_key=os.environ.get('GEMINI_API_KEY'))
    # Use gemini-2.5-pro just like text_extractor.py does
    print("\nAttempting to generate content using gemini-2.5-pro...\n")
    client.models.generate_content(model='gemini-2.5-pro', contents='hello')
    print("MODEL OK: Generation successful!")
except Exception as e:
    import traceback
    print("ERROR CAUGHT DURING EXTRACTION TEST:")
    traceback.print_exc()
