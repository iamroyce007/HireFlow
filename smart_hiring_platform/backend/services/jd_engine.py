import json
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import List


class JDAnalysis(BaseModel):
    skills: List[str]
    seniority: str
    responsibilities: List[str]
    nice_to_have: List[str]
    min_experience_years: int


def parse_jd(description: str, api_key: str) -> dict:
    """Parse a job description using HireAI to extract structured requirements."""
    
    if not api_key or api_key.strip() == "":
        raise ValueError("API key required for JD parsing")
    
    if not description or description.strip() == "":
        raise ValueError("Job description text is required")
    
    try:
        client = genai.Client(api_key=api_key)
        
        prompt = f"""Extract from this job description: skills, seniority (Junior/Mid/Senior/Lead/Principal), responsibilities (max 6), nice_to_have, min_experience_years.

JD: {description}"""
        
        models_to_try = [
            "models/gemini-2.5-flash",
            "models/gemini-flash-latest",
            "models/gemini-2.0-flash"
        ]
        
        import time
        for attempt in range(2):
            for m in models_to_try:
                try:
                    response = client.models.generate_content(
                        model=m,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=JDAnalysis,
                        ),
                    )
                    return json.loads(response.text) or {}
                except Exception as e:
                    if "429" in str(e) or "404" in str(e):
                        continue
            
            if attempt < 1:
                time.sleep(5)
            else:
                raise ValueError("All configured models are either exhausted (429) or unavailable (404) on this API key.")
        
    except Exception as e:
        print(f"JD Parsing Error: {e}")
        raise ValueError(f"Failed to parse job description: {str(e)}")
