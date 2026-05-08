#  HireFlow — AI Recruitment Platform

**HireFlow** is an end-to-end AI-powered recruitment engine designed to automate resume screening, candidate evaluation, and proctored technical interviews. Built for the modern recruiter, it leverages Google Gemini and deep-learning heuristics to identify top-tier talent with extreme precision.

![HIREFLOW_BANNER](file:///c:/Users/ramar/OneDrive/Desktop/HireFlow/smart_hiring_platform/frontend/banner.png)

---

##  Core Features

### 🔍 Intelligent Screening
*   **AI Resume Parsing:** Extracts skills, experience, and key achievements using LLM-based entity recognition.
*   **Vector-like Matching:** Semantic matching between candidate profiles and job descriptions.
*   **Social Signal Analysis:** Deep-dives into **GitHub** and **LinkedIn** profiles to verify technical expertise and career trajectory.

###  Advanced Evaluation
*   **Autonomous Interview Agents:** AI-driven interviewers that generate questions dynamically based on the candidate's background.
*   **Multi-Modal Evaluation:** Scores candidates across technical accuracy, communication clarity, and behavioral consistency.
*   **Composite Scoring:** Weighted engine that combines resume strength, social signals, and interview performance into a single "Hireability Score."

###  Standalone Proctoring System (v2.0)
*   **Live AI Proctoring:** Real-time face detection, eye-tracking, and tab-switching monitoring.
*   **Multi-Channel Recording:** Analyzes audio, video (JPEG frames), and transcript data for integrity.
*   **Comprehensive Reports:** Detailed AI assessment reports generated instantly after interview completion.

---

##  Tech Stack

| Component | Technology |
| :--- | :--- |
| **Backend** | Python, FastAPI, SQLAlchemy, Uvicorn |
| **Frontend** | Vanilla JS, HTML5, CSS3 (Modern Glassmorphism) |
| **Database** | SQLite (Production-ready v2.0 schema) |
| **AI/LLM** | Google Gemini (GenAI), Custom Evaluation Engine |
| **Utilities** | pdfplumber, httpx, numpy, scipy |

---

##  Quickstart

### 1. Prerequisites
Ensure you have **Python 3.10+** installed.

### 2. Environment Setup
```bash
# Clone the repository
cd smart_hiring_platform

# Install dependencies (using existing venv or globally)
pip install -r backend/requirements.txt
```

### 3. Launch the Platform
```bash
cd backend
python main.py
```
*   **Recruiter Portal:** [http://localhost:8000/recruiter-login.html](http://localhost:8000/recruiter-login.html)
*   **Candidate Portal:** [http://localhost:8000/index.html](http://localhost:8000/index.html)
*   **API Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📂 Project Structure

```text
smart_hiring_platform/
├── backend/
│   ├── main.py              # FastAPI core & API gateway
│   ├── database.py          # SQLAlchemy models & DB config
│   ├── services/            # AI & Business logic
│   │   ├── jd_engine.py      # Job Description processor
│   │   ├── resume_engine.py  # AI Resume extractor
│   │   ├── interview_agents.py # AI Interviewer logic
│   │   └── scoring_engine.py  # Multi-weighted score computer
│   └── proctoring_v2/       # Advanced AI proctoring service
└── frontend/                # Glassmorphic UI & logic
    ├── index.html           # Candidate landing page
    └── recruiter-login.html  # Recruiter dashboard entry
```

---

##  Configuration
To enable the full AI evaluation pipeline, navigate to the **Settings** menu in the Recruiter Dashboard and configure your **Gemini AI API Key**.

---

> [!NOTE]
> This platform was optimized during the final refactoring phase to remove redundant legacy code and transition to the `v2.0` schema. Only essential, high-performance components have been retained.
