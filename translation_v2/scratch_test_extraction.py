import os
import sys
from dotenv import load_dotenv
load_dotenv()

# Add translation_v2 to path so we can import text_extractor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import text_extractor

def test():
    job_id = "e659c20d-e077-490a-9577-e96d83d8c468"
    doc_path = os.path.join("jobs", job_id, "source.docx")
    print(f"Testing extraction on {doc_path}...")
    try:
        qs, imgs = text_extractor.extract_questions(doc_path)
        print(f"Extracted {len(qs)} questions and {len(imgs)} images.")
        for q in qs:
            print(f"Q {q.get('question_no')}: {q.get('raw_text')[:50]}...")
    except Exception as e:
        print(f"Extraction failed: {e}")

if __name__ == "__main__":
    test()
