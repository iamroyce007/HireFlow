import os
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
import uvicorn
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import json
import numpy as np

from database import SessionLocal, CandidateDB, JobDescriptionDB
from services.resume_engine import process_resume_with_ai
from services.interview_agents import evaluate_candidate_with_ai
from services.scoring_engine import calculate_final_score, calculate_behavior_score
from services.enrichment_engine import analyze_github_profile, analyze_linkedin_profile
from services.matching_engine import match_candidates_to_job
from services.jd_engine import parse_jd

app = FastAPI(title="HireFlow — AI Recruitment Platform", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_api_key_store = {"key": "AIzaSyBzW5r7hk_yTKchbET5FDk4MJ2_UHKBHxk"}


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_api_key():
    """Get the stored API key."""
    return _api_key_store.get("key", "")


# ─── SETTINGS ───

class APIKeyRequest(BaseModel):
    api_key: str

@app.post("/api/settings/api-key")
def set_api_key(req: APIKeyRequest):
    """Store HireAI API key for the session."""
    _api_key_store["key"] = req.api_key.strip()
    return {"message": "API key saved successfully", "configured": True}

@app.get("/api/settings/status")
def get_settings_status():
    """Check if API key is configured."""
    key = get_api_key()
    return {"configured": bool(key and len(key) > 10)}


# ─── DASHBOARD STATS ───

@app.get("/api/stats")
def get_dashboard_stats(company: Optional[str] = None, db: Session = Depends(get_db)):
    """Return aggregate statistics for the dashboard, filtered by company."""
    query = db.query(CandidateDB)
    if company:
        query = query.filter(CandidateDB.company == company)
    
    total = query.count()
    evaluated = query.filter(CandidateDB.status == "Deep Evaluated").count()
    shortlisted = query.filter(CandidateDB.status == "Shortlisted").count()
    
    candidates = query.all()
    avg_score = 0
    if candidates:
        scores = [c.baseline_score for c in candidates if c.baseline_score and c.baseline_score > 0]
        avg_score = round(sum(scores) / len(scores), 1) if scores else 0
    
    avg_composite = 0
    evaluated_candidates = [c for c in candidates if c.composite_score and c.composite_score > 0]
    if evaluated_candidates:
        avg_composite = round(sum(c.composite_score for c in evaluated_candidates) / len(evaluated_candidates), 1)
    
    return {
        "total_candidates": total,
        "evaluated": evaluated,
        "shortlisted": shortlisted,
        "avg_resume_score": avg_score,
        "avg_composite_score": avg_composite,
        "pipeline_health": "Active" if total > 0 else "Empty"
    }


# ─── CANDIDATES ───

@app.get("/api/candidates/verify-github")
def verify_github_profile(url: str):
    """Real-time GitHub profile verification for candidates."""
    return analyze_github_profile(url)


@app.post("/api/candidates/extract-links")
async def extract_links_endpoint(file: UploadFile = File(...)):
    """Extract GitHub and LinkedIn URLs from uploaded PDF resume."""
    from services.link_extractor import LinkExtractor
    try:
        file_bytes = await file.read()
        links = LinkExtractor.extract_links_from_pdf(file_bytes)
        return links
    except Exception as e:
        print(f"Error parsing PDF for links: {e}")
        return {"github": None, "linkedin": None}

@app.get("/api/candidates")
def get_all_candidates(company: Optional[str] = None, db: Session = Depends(get_db)):
    """Fetch all candidates from database, optionally filtered by company."""
    query = db.query(CandidateDB)
    if company:
        query = query.filter(CandidateDB.company == company)
    
    records = query.order_by(CandidateDB.applied_at.desc()).all()
    out = []
    for r in records:
        candidate_dict = {
            "id": r.id,
            "name": r.name,
            "email": r.email,
            "github_url": r.github_url,
            "linkedin_url": r.linkedin_url,
            "skills": r.skills if r.skills else [],
            "years_experience": r.years_experience or 0,
            "summary": r.summary or "",
            "baselineScore": r.baseline_score,
            "status": r.status,
            "evaluated": r.status in ["Deep Evaluated", "Shortlisted"],
            "strengths": r.strengths if r.strengths else [],
            "skill_gaps": r.skill_gaps if r.skill_gaps else [],
            "ai_summary": r.ai_summary or "",
            "applied_at": r.applied_at.isoformat() if r.applied_at else None,
            "evaluated_at": r.evaluated_at.isoformat() if r.evaluated_at else None,
            "metrics": {
                "behavior_score": r.behavior_score,
                "composite_score": r.composite_score,
                "github_score": r.github_score,
                "github_data": r.github_data,
                "interview_qa": r.interview_qa_score,
                "career_growth": r.career_growth_score,
                "consistency": r.consistency_score,
                "behavioral_insights": r.behavioral_insights
            } if r.status in ["Deep Evaluated", "Shortlisted"] else None
        }
        out.append(candidate_dict)
    return {"candidates": out}


@app.get("/api/candidates/{candidate_id}")
def get_candidate(candidate_id: int, db: Session = Depends(get_db)):
    """Fetch a single candidate by ID."""
    candidate = db.query(CandidateDB).filter(CandidateDB.id == candidate_id).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    
    return {
        "id": candidate.id,
        "name": candidate.name,
        "email": candidate.email,
        "github_url": candidate.github_url,
        "linkedin_url": candidate.linkedin_url,
        "skills": candidate.skills or [],
        "years_experience": candidate.years_experience or 0,
        "summary": candidate.summary or "",
        "baselineScore": candidate.baseline_score,
        "status": candidate.status,
        "strengths": candidate.strengths or [],
        "skill_gaps": candidate.skill_gaps or [],
        "ai_summary": candidate.ai_summary or ""
    }


@app.delete("/api/candidates/{candidate_id}")
def delete_candidate(candidate_id: int, db: Session = Depends(get_db)):
    """Delete a candidate from the pipeline."""
    candidate = db.query(CandidateDB).filter(CandidateDB.id == candidate_id).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    db.delete(candidate)
    db.commit()
    return {"message": f"Candidate {candidate.name} removed from pipeline"}


class StatusUpdate(BaseModel):
    status: str

@app.patch("/api/candidates/{candidate_id}/status")
def update_candidate_status(candidate_id: int, req: StatusUpdate, db: Session = Depends(get_db)):
    """Update candidate status (Shortlisted, Rejected, etc.)."""
    candidate = db.query(CandidateDB).filter(CandidateDB.id == candidate_id).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    
    valid_statuses = ["Applied", "Interviewing", "Deep Evaluated", "Shortlisted", "Rejected"]
    if req.status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of: {valid_statuses}")
    
    candidate.status = req.status
    db.commit()
    return {"message": f"Status updated to {req.status}"}


# ─── EVALUATION ───

class InterviewRequest(BaseModel):
    candidate_id: int

@app.post("/api/interviews/evaluate")
def run_ai_evaluation(req: InterviewRequest, db: Session = Depends(get_db)):
    """Run full AI deep evaluation on a candidate."""
    api_key = get_api_key()
    if not api_key:
        raise HTTPException(status_code=400, detail="HireAI API key not configured. Go to Settings to add your key.")
    
    candidate = db.query(CandidateDB).filter(CandidateDB.id == req.candidate_id).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    
    # Cache check: skip re-evaluation if already deep evaluated
    if candidate.status == "Deep Evaluated" and candidate.composite_score:
        return {
            "message": "Using cached evaluation (no API credits used)",
            "composite_score": candidate.composite_score,
            "cached": True
        }
    
    # GitHub enrichment (real API)
    gh_analysis = analyze_github_profile(candidate.github_url)
    
    # LinkedIn analysis
    li_analysis = analyze_linkedin_profile(candidate.linkedin_url)
    
    # AI-powered interview evaluation
    eval_result = evaluate_candidate_with_ai(
        api_key=api_key,
        candidate_name=candidate.name,
        skills=candidate.skills or [],
        summary=candidate.summary or "",
        years_experience=candidate.years_experience or 0,
        github_metrics=gh_analysis.get("metrics", ""),
        linkedin_metrics=li_analysis.get("metrics", "")
    )
    
    bd = eval_result["scores_breakdown"]
    
    # Behavior Score
    b_score = calculate_behavior_score(
        bd["communication_clarity"], bd["confidence"],
        bd["technical_explanation"], bd["consistency_score"],
        bd["professionalism"]
    )
    
    # Composite Score
    f_score = calculate_final_score(
        resume_score=candidate.baseline_score or 0.0,
        github_score=gh_analysis["score"],
        interview_qa_score=bd["interview_qa"],
        behavior_score=b_score,
        career_growth_score=li_analysis["score"],
        consistency_score=bd["consistency_score"]
    )
    
    # Save everything to DB
    candidate.status = "Deep Evaluated"
    candidate.github_score = gh_analysis["score"]
    candidate.github_data = gh_analysis
    candidate.interview_qa_score = bd["interview_qa"]
    candidate.career_growth_score = li_analysis["score"]
    candidate.consistency_score = bd["consistency_score"]
    candidate.behavior_score = b_score
    candidate.composite_score = f_score
    candidate.behavioral_insights = eval_result["behavioral_insights"]
    candidate.ai_summary = eval_result["behavioral_insights"].get("summary", "")
    candidate.evaluated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(candidate)
    
    return {
        "message": "AI Evaluation Complete",
        "composite_score": f_score,
        "github_analysis": gh_analysis,
        "linkedin_analysis": li_analysis
    }


# ─── CANDIDATE APPLICATION ───

@app.post("/api/candidates/apply")
async def candidate_apply(
    name: str = Form(""),
    email: str = Form(""),
    github_url: str = Form(""),
    linkedin_url: str = Form(""),
    company: str = Form(""),
    resume: UploadFile = File(...)
):
    """Parse resume via HireAI and store candidate in database."""
    api_key = get_api_key()
    if not api_key:
        return {"error": "HireAI API key not configured. Please ask the recruiter to configure it in Settings."}
    
    db = SessionLocal()
    content_bytes = await resume.read()
    
    try:
        parsed_data = process_resume_with_ai(content_bytes, api_key)
        
        final_name = name if name.strip() else parsed_data.get("name", "Unknown")
        final_email = email if email.strip() else parsed_data.get("email", "unknown@example.com")
        skills = parsed_data.get("skills", [])
        experience = parsed_data.get("years_experience", 0)
        summary = parsed_data.get("summary", "")
        strengths = parsed_data.get("strengths", [])
        skill_gaps = parsed_data.get("skill_gaps", [])
        base_score = parsed_data.get("baseline_score", 50.0)
        
        # Check for duplicate email
        existing = db.query(CandidateDB).filter(CandidateDB.email == final_email).first()
        if existing:
            db.close()
            return {"error": f"A candidate with email {final_email} has already applied."}
        
        new_candidate = CandidateDB(
            name=final_name,
            email=final_email,
            github_url=github_url,
            linkedin_url=linkedin_url,
            company=company,
            skills=skills,
            years_experience=experience,
            summary=summary,
            baseline_score=base_score,
            strengths=strengths,
            skill_gaps=skill_gaps,
            status="Applied"
        )
        db.add(new_candidate)
        db.commit()
        db.refresh(new_candidate)
        
        return {
            "message": "Application submitted successfully!",
            "data": {
                "name": final_name,
                "email": final_email,
                "skills": skills,
                "computed_score": base_score,
                "strengths": strengths,
                "experience": experience
            }
        }
    except Exception as e:
        db.close()
        return {"error": str(e)}


# ─── JOB DESCRIPTIONS ───

class JDRequest(BaseModel):
    title: str
    description: str

@app.post("/api/jobs")
def create_job(req: JDRequest, db: Session = Depends(get_db)):
    """Parse and store a job description using AI."""
    api_key = get_api_key()
    if not api_key:
        raise HTTPException(status_code=400, detail="API key not configured")
    
    parsed = parse_jd(req.description, api_key)
    
    job = JobDescriptionDB(
        title=req.title,
        description=req.description,
        required_skills=parsed.get("skills", []),
        seniority=parsed.get("seniority", "Mid"),
        responsibilities=parsed.get("responsibilities", [])
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    
    return {"message": "Job description saved", "id": job.id, "parsed": parsed}

@app.get("/api/jobs")
def get_all_jobs(db: Session = Depends(get_db)):
    """Fetch all job descriptions."""
    jobs = db.query(JobDescriptionDB).order_by(JobDescriptionDB.created_at.desc()).all()
    return {"jobs": [{
        "id": j.id,
        "title": j.title,
        "description": j.description,
        "required_skills": j.required_skills or [],
        "seniority": j.seniority,
        "responsibilities": j.responsibilities or [],
        "created_at": j.created_at.isoformat() if j.created_at else None
    } for j in jobs]}


@app.get("/api/jobs/{job_id}/matches")
def get_job_matches(job_id: int, db: Session = Depends(get_db)):
    """Match candidates to a specific job description."""
    job = db.query(JobDescriptionDB).filter(JobDescriptionDB.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    candidates = db.query(CandidateDB).all()
    candidate_dicts = [{
        "id": c.id,
        "name": c.name,
        "email": c.email,
        "skills": c.skills or [],
        "baselineScore": c.baseline_score,
        "composite_score": c.composite_score,
        "status": c.status,
        "years_experience": c.years_experience or 0
    } for c in candidates]
    
    matched = match_candidates_to_job(candidate_dicts, job.required_skills or [])
    return {"job_title": job.title, "matches": matched}



# ─── STANDALONE PROCTORED INTERVIEW SYSTEM ───
# Self-contained — does NOT require database records.
# Uses: InterviewAgent, STTService, AudioProcessor, EvaluationEngine, ProctoringService
# NOTE: Imports are lazy to avoid crashing server if optional deps (cv2, whisper, mediapipe) are missing.

_proctor_service = None

def _get_proctor():
    global _proctor_service
    if _proctor_service is None:
        from proctoring_v2.proctoring_service import ProctoringServiceV2
        _proctor_service = ProctoringServiceV2()
    return _proctor_service

class InterviewStartRequest(BaseModel):

    candidate_name: str
    job_role: str

@app.post("/api/interview/start")
def start_standalone_interview(req: InterviewStartRequest):
    """Start a new proctored interview session. No DB required."""
    from services.live_interview import InterviewAgent, create_session
    api_key = get_api_key()
    if not api_key:
        raise HTTPException(status_code=400, detail="API key not configured")

    session = create_session(req.candidate_name.strip(), req.job_role.strip())
    agent = InterviewAgent(api_key)

    # Generate first question via Gemini
    first_question = agent.generate_question(session)
    session.current_question = first_question
    session.question_number = 1

    return {
        "session_id": session.session_id,
        "candidate_name": session.candidate_name,
        "job_role": session.job_role,
        "current_question": first_question,
        "question_number": 1,
        "total_questions": session.total_questions,
    }


@app.get("/api/interview/{session_id}/state")
def get_interview_session_state(session_id: str):
    """Get current interview session state."""
    from services.live_interview import get_session
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.to_dict()


class TextAnswerRequest(BaseModel):
    answer: str

@app.post("/api/interview/{session_id}/answer")
def submit_text_answer(session_id: str, req: TextAnswerRequest):
    """Submit a typed text answer. Evaluates and generates next question."""
    from services.live_interview import InterviewAgent, get_session, save_session_state
    from services.evaluation_engine import EvaluationEngine
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.completed:
        raise HTTPException(status_code=400, detail="Interview already completed")

    api_key = get_api_key()
    evaluator = EvaluationEngine(api_key)
    agent = InterviewAgent(api_key)

    # Evaluate the answer with Gemini
    evaluation = evaluator.evaluate_answer(
        question=session.current_question,
        answer=req.answer.strip(),
        job_role=session.job_role,
        candidate_name=session.candidate_name,
        question_number=session.question_number,
        total_questions=session.total_questions,
    )

    # Save to history
    session.history.append({
        "question": session.current_question,
        "answer": req.answer.strip(),
        "evaluation": evaluation,
        "question_number": session.question_number,
        "answer_mode": "text",
    })

    # Check if interview is complete
    is_last = session.question_number >= session.total_questions
    next_question = None

    if not is_last:
        session.question_number += 1
        next_question = agent.generate_question(session)
        session.current_question = next_question
    else:
        session.completed = True

    save_session_state() # PERSIST CHANGE
    return {
        "evaluation": evaluation,
        "question_number": session.question_number,
        "is_complete": is_last,
        "next_question": next_question,
    }


@app.post("/api/interview/{session_id}/audio")
async def submit_audio_answer(session_id: str, audio: UploadFile = File(...)):
    """Submit an audio recording. Transcribes → Analyzes → Evaluates → Next question."""
    from services.live_interview import InterviewAgent, get_session, save_session_state
    from services.stt_service import STTService
    from services.audio_processor import AudioProcessor
    from services.evaluation_engine import EvaluationEngine
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.completed:
        raise HTTPException(status_code=400, detail="Interview already completed")

    api_key = get_api_key()
    audio_bytes = await audio.read()

    # 1. Speech-to-Text (OPTIMIZED: Local Whisper Async)
    stt = STTService()
    stt_result = await stt.transcribe_async(audio_bytes) # Async call

    # 2. Audio Confidence Analysis (real DSP)
    processor = AudioProcessor()
    audio_analysis = processor.analyze(audio_bytes)

    # 3. Evaluate answer with LLM
    evaluator = EvaluationEngine(api_key)
    transcript_text = stt_result["text"] if stt_result["has_speech"] else ""

    evaluation = evaluator.evaluate_answer(
        question=session.current_question,
        answer=transcript_text,
        job_role=session.job_role,
        candidate_name=session.candidate_name,
        question_number=session.question_number,
        total_questions=session.total_questions,
    )

    # 4. Save to history
    session.history.append({
        "question": session.current_question,
        "answer": transcript_text,
        "evaluation": evaluation,
        "audio_analysis": audio_analysis,
        "question_number": session.question_number,
        "answer_mode": "audio",
    })

    # 5. Next question or complete
    agent = InterviewAgent(api_key)
    is_last = session.question_number >= session.total_questions
    next_question = None

    if not is_last:
        session.question_number += 1
        next_question = agent.generate_question(session)
        session.current_question = next_question
    else:
        session.completed = True

    save_session_state() # PERSIST CHANGE
    return {
        "transcript": stt_result,
        "audio_analysis": audio_analysis,
        "evaluation": evaluation,
        "question_number": session.question_number,
        "is_complete": is_last,
        "next_question": next_question,
    }


@app.post("/api/interview/{session_id}/proctor")
async def proctor_check(session_id: str, frame: UploadFile = File(...)):
    """Submit a webcam JPEG frame for proctoring analysis."""
    from services.live_interview import get_session
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    api_key = get_api_key()
    proctor = _get_proctor()
    image_bytes = await frame.read()

    result = proctor.analyze_bytes(image_bytes)

    # Log violations
    if result.get("violation"):
        session.proctor_events.append({
            "type": result.get("violation_type", "UNKNOWN"),
            "reason": result.get("reason", ""),
            "question_number": session.question_number,
            "timestamp": result.get("timestamp", 0),
        })
        from services.live_interview import save_session_state
        save_session_state()

    return result


@app.post("/api/interview/{session_id}/log_event")
def log_proctor_event(session_id: str, event: dict):
    """Log a proctoring violation event from the client (e.g. Tab Switch)."""
    from services.live_interview import get_session
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    event_type = event.get("type", "UNKNOWN_EVENT")
    timestamp = event.get("timestamp", 0)
    
    session.proctor_events.append({
        "type": event_type,
        "reason": f"Client-side violation: {event_type}",
        "question_number": session.question_number,
        "timestamp": timestamp,
    })
    
    from services.live_interview import save_session_state
    save_session_state()
    return {"status": "event_logged"}


@app.post("/api/interview/{session_id}/complete")
def complete_standalone_interview(session_id: str):
    """Complete the interview and generate final AI assessment report."""
    from services.live_interview import get_session
    from services.evaluation_engine import EvaluationEngine
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    api_key = get_api_key()
    evaluator = EvaluationEngine(api_key)

    import time
    duration = int(time.time() - session.start_time)

    report = evaluator.generate_final_report(
        candidate_name=session.candidate_name,
        job_role=session.job_role,
        history=session.history,
        proctor_events=session.proctor_events,
        duration_seconds=duration,
    )

    session.completed = True
    session.report = report

    return report


# Mount frontend static files (AFTER all API routes)
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
