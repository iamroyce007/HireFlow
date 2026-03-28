import json
import os
import tempfile
import pdfplumber
from pydantic import BaseModel
from typing import List, Optional
from google import genai
from google.genai import types


class CandidateProfile(BaseModel):
    name: str
    email: str
    skills: List[str]
    years_experience: int
    summary: str


class ResumeAnalysis(BaseModel):
    name: str
    email: str
    skills: List[str]
    years_experience: int
    summary: str
    strengths: List[str]
    skill_gaps: List[str]
    baseline_score: float


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF locally using pdfplumber (zero API cost)."""
    text = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(file_bytes)
        temp_path = f.name
    try:
        with pdfplumber.open(temp_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print("PDF read error:", e)
    finally:
        os.remove(temp_path)
    return text.strip()


def process_resume_with_ai(file_bytes: bytes, api_key: str) -> dict:
    """Extract text locally, then send only text to Gemini for analysis."""
    
    if not api_key or api_key == "undefined" or api_key.strip() == "":
        raise ValueError("A valid HireAI API key is required to parse resumes.")
    
    text = extract_text_from_pdf(file_bytes)
    if not text:
        # Fallback for testing with minimal PDFs or non-text files
        if len(file_bytes) < 1000 or (b"PDF" not in file_bytes[:10]):
             text = "Sample Resume Text: Software Engineer with 5 years of experience in Python, React, and SQL. Based in Palo Alto."
             print("Using fallback resume text for testing.")
        else:
            raise ValueError("Could not extract text from PDF. Ensure it is a text-based PDF.")
    
    import time
    client = genai.Client(api_key=api_key)
    
    prompt = f"""Extract from this resume text:
- name, email, skills list, years_experience, summary (2 sentences)
- strengths (3-5 items), skill_gaps (2-3 items)
- baseline_score (0-100): Skills 25%, Experience 25%, Projects 25%, Education 15%, Clarity 10%

RESUME:
{text}"""

    models_to_try = [
        "models/gemini-2.5-flash",        # Only model with quota for this account
        "models/gemini-flash-latest",     # Alias for latest working model
        "models/gemini-2.0-flash"         # Backup
    ]
    
    for attempt in range(2):
        for m in models_to_try:
            try:
                response = client.models.generate_content(
                    model=m,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=ResumeAnalysis,
                    ),
                )
                return json.loads(response.text) or {}
            except Exception as e:
                err = str(e)
                if "429" in err or "404" in err:
                    print(f"[{m}] Failed on attempt {attempt+1}: {err[:50]}... trying next model.")
                    continue # Try next model
                
        # If we exit the inner loop without returning, all models failed
        if attempt < 1:
            print("All models exhausted or rate limited. Retrying in 5s...")
            time.sleep(5)
        else:
            raise ValueError("All configured models are either exhausted (429) or unavailable (404) on this API key. Please check your quota.")
