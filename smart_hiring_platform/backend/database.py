from sqlalchemy import create_engine, Column, Integer, String, Float, Text, JSON, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

import os
basedir = os.path.abspath(os.path.dirname(__file__))
DATABASE_URL = f"sqlite:///{os.path.join(basedir, 'hiring_platform_v2.db')}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class CandidateDB(Base):
    __tablename__ = "candidates"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    skills = Column(JSON)
    years_experience = Column(Integer, default=0)
    summary = Column(Text, default="")
    baseline_score = Column(Float, default=0.0)
    status = Column(String, default="Applied")  # Applied, Interviewing, Deep Evaluated, Shortlisted, Rejected
    company = Column(String, index=True, default="")
    
    github_url = Column(String, default="")
    linkedin_url = Column(String, default="")
    github_data = Column(JSON, default={})
    
    # AI Scores
    github_score = Column(Float, default=0.0)
    interview_qa_score = Column(Float, default=0.0)
    behavior_score = Column(Float, default=0.0)
    career_growth_score = Column(Float, default=0.0)
    consistency_score = Column(Float, default=0.0)
    composite_score = Column(Float, default=0.0)
    
    behavioral_insights = Column(JSON, default={})
    ai_summary = Column(Text, default="")
    skill_gaps = Column(JSON, default=[])
    strengths = Column(JSON, default=[])
    
    applied_at = Column(DateTime, default=datetime.utcnow)
    evaluated_at = Column(DateTime, nullable=True)

class JobDescriptionDB(Base):
    __tablename__ = "job_descriptions"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(Text)
    required_skills = Column(JSON, default=[])
    seniority = Column(String, default="Mid")
    responsibilities = Column(JSON, default=[])
    created_at = Column(DateTime, default=datetime.utcnow)

# Create the database tables
Base.metadata.create_all(bind=engine)
