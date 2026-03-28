import json
from pydantic import BaseModel
from typing import List, Optional
from services.llm_client import LLMClient

class InterviewEvaluation(BaseModel):
    interview_qa_score: float
    communication_clarity: float
    confidence: float
    technical_explanation: float
    professionalism: float
    consistency_score: float
    summary: str
    red_flags: str
    key_observations: List[str]


def evaluate_candidate_with_ai(
    api_key: str,
    candidate_name: str,
    skills: list,
    summary: str,
    years_experience: int,
    github_metrics: str = "",
    linkedin_metrics: str = ""
) -> dict:
    """
    Uses DeepSeek-V3.2 to evaluate a candidate based on their complete profile.
    """
    
    skills_str = ", ".join(skills) if skills else "Not specified"
    
    prompt = f"""You are a senior technical interviewer. Evaluate this candidate based on their profile.

CANDIDATE: {candidate_name}
SKILLS: {skills_str}
EXPERIENCE: {years_experience} years
SUMMARY: {summary}
GITHUB: {github_metrics}
LINKEDIN: {linkedin_metrics}

Return JSON with 0-100 scores:
- interview_qa_score
- communication_clarity
- confidence
- technical_explanation
- professionalism
- consistency_score
- summary (2-3 sentences)
- red_flags (string)
- key_observations (list of 3-4 strings)"""

    try:
        client = LLMClient(model_type="deepseek")
        
        response_text = client.generate_content(
            prompt=prompt,
            system_prompt="You are a senior technical interviewer. Return ONLY valid JSON.",
            json_mode=True
        )
        
        eval_data = json.loads(response_text)
        
        comm = eval_data.get("communication_clarity", 75)
        conf = eval_data.get("confidence", 75)
        tech = eval_data.get("technical_explanation", 75)
        prof = eval_data.get("professionalism", 80)
        
        def to_stars(score: float) -> str:
            if score >= 90: return "⭐⭐⭐⭐⭐"
            elif score >= 80: return "⭐⭐⭐⭐☆"
            elif score >= 70: return "⭐⭐⭐☆☆"
            elif score >= 60: return "⭐⭐☆☆☆"
            return "⭐☆☆☆☆"
        
        return {
            "status": "Evaluated",
            "scores_breakdown": {
                "interview_qa": eval_data.get("interview_qa_score", 75),
                "communication_clarity": comm,
                "confidence": conf,
                "technical_explanation": tech,
                "professionalism": prof,
                "consistency_score": eval_data.get("consistency_score", 80),
            },
            "behavioral_insights": {
                "communication_stars": to_stars(comm),
                "confidence_stars": to_stars(conf),
                "technical_clarity_stars": to_stars(tech),
                "communication_score": comm,
                "confidence_score": conf,
                "technical_score": tech,
                "professionalism_score": prof,
                "summary": eval_data.get("summary", ""),
                "red_flags": eval_data.get("red_flags", ""),
                "key_observations": eval_data.get("key_observations", [])
            }
        }
        
    except Exception as e:
        print(f"DeepSeek Evaluation Error: {e}")
        raise ValueError(f"AI evaluation failed: {str(e)}")
