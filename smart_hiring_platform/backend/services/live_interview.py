"""
Live Interview Session Orchestrator
Coordinates all services: question generation, STT, audio analysis, evaluation, proctoring.
Session management with in-memory store.
"""

import os
import json
import uuid
import time
from typing import List, Dict, Optional


# ─── SESSION STORE WITH PERSISTENCE ───
SESSION_FILE = "interviews_sessions.json"

def _load_sessions() -> Dict[str, "InterviewSession"]:
    if not os.path.exists(SESSION_FILE):
        return {}
    try:
        with open(SESSION_FILE, "r") as f:
            data = json.load(f)
            sessions = {}
            for sid, sdata in data.items():
                s = InterviewSession(sid, sdata["candidate_name"], sdata["job_role"])
                s.history = sdata.get("history", [])
                s.current_question = sdata.get("current_question", "")
                s.question_number = sdata.get("question_number", 0)
                s.completed = sdata.get("completed", False)
                s.proctor_events = sdata.get("proctor_events", [])
                s.report = sdata.get("report")
                s.start_time = sdata.get("start_time", time.time())
                sessions[sid] = s
            return sessions
    except:
        return {}

def _save_sessions():
    data = {sid: s.to_dict() for sid, s in _sessions.items()}
    with open(SESSION_FILE, "w") as f:
        json.dump(data, f)

_sessions: Dict[str, "InterviewSession"] = _load_sessions()

TOTAL_QUESTIONS = 5


class InterviewSession:
    """Represents a single proctored interview session."""

    def __init__(self, session_id: str, candidate_name: str, job_role: str):
        self.session_id = session_id
        self.candidate_name = candidate_name
        self.job_role = job_role
        self.history: List[Dict] = []
        self.current_question: str = ""
        self.question_number: int = 0
        self.total_questions: int = TOTAL_QUESTIONS
        self.start_time: float = time.time()
        self.completed: bool = False
        self.proctor_events: List[Dict] = []
        self.report: Optional[Dict] = None

    def to_dict(self):
        return {
            "session_id": self.session_id,
            "candidate_name": self.candidate_name,
            "job_role": self.job_role,
            "current_question": self.current_question,
            "question_number": self.question_number,
            "total_questions": self.total_questions,
            "history": self.history,
            "completed": self.completed,
            "proctor_events": self.proctor_events,
            "report": self.report,
            "start_time": self.start_time,
        }


class InterviewAgent:
    """Agentic question generator — uses Gemini AI."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate_question(self, session: InterviewSession) -> str:
        """Generate the next interview question using DeepSeek, avoiding repeats."""
        from services.llm_client import LLMClient

        # Build history context
        asked_questions = [h["question"] for h in session.history]
        history_str = ""
        if session.history:
            history_str = "\n\n".join([
                f"Q{i+1}: {h['question']}\nAnswer: {h['answer']}\nScore: {h.get('evaluation', {}).get('score', 'N/A')}/10"
                for i, h in enumerate(session.history)
            ])

        avoid_str = ""
        if asked_questions:
            avoid_str = "\n\nALREADY ASKED (do NOT repeat these):\n" + "\n".join(f"- {q}" for q in asked_questions)

        q_num = session.question_number

        prompt = f"""You are conducting a professional job interview.
Candidate: {session.candidate_name}
Role: {session.job_role}
This is question {q_num} of {session.total_questions}.

Interview history so far:
{history_str if history_str else "This is the first question."}
{avoid_str}

Adaptive rules:
- If the last score >= 8, ask a harder technical question.
- If the last score < 5, ask a simpler/clarifying question.
- Each question must be DIFFERENT from all previously asked questions.
- Be specific and relevant to the {session.job_role} role.

Return ONLY the question text, nothing else."""

        try:
            client = LLMClient(model_type="deepseek")
            # Override with the backend's provided key if needed, though LLMClient uses its own hardcoded key
            response_text = client.generate_content(
                prompt=prompt,
                system_prompt="You are a strict technical interviewer. Return ONLY the question."
            )
            question = response_text.strip()
            # Strip any quotes or prefixes the model might add
            question = question.strip('"\'').lstrip("Q: ").lstrip("Question: ")
            return question if len(question) > 10 else self._fallback_question(q_num, session.job_role)
        except Exception as e:
            print(f"[InterviewAgent] DeepSeek question gen error: {e}")
            return self._fallback_question(q_num, session.job_role)

    def _fallback_question(self, q_num: int, job_role: str) -> str:
        """Diverse fallback questions to avoid repetition even when LLM fails."""
        fallbacks = [
            f"Can you walk me through a challenging problem you solved in your {job_role} experience?",
            f"What technologies or frameworks have you used most extensively in your recent {job_role} work?",
            "Describe a time when you had to learn a new technology quickly under pressure. How did you approach it?",
            "How do you ensure code quality and maintainability in your projects?",
            "Can you tell me about a project you are particularly proud of and what made it successful?",
        ]
        return fallbacks[(q_num - 1) % len(fallbacks)]


# ─── SESSION MANAGEMENT FUNCTIONS ───

def create_session(candidate_name: str, job_role: str) -> InterviewSession:
    session_id = str(uuid.uuid4())
    session = InterviewSession(session_id, candidate_name, job_role)
    _sessions[session_id] = session
    _save_sessions()
    return session


def get_session(session_id: str) -> Optional[InterviewSession]:
    return _sessions.get(session_id)


def save_session_state():
    _save_sessions()


def remove_session(session_id: str):
    _sessions.pop(session_id, None)
    _save_sessions()
