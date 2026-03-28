def calculate_final_score(
    resume_score: float,
    github_score: float,
    interview_qa_score: float,
    behavior_score: float,
    career_growth_score: float,
    consistency_score: float
) -> float:
    """
    Composite Scoring Engine
    Weighted formula:
    0.25 * Resume
    0.20 * GitHub (real data)
    0.20 * Interview QA (AI evaluated)
    0.15 * Behavior Analysis (AI)
    0.10 * Career Growth
    0.10 * Consistency
    
    Handles cases where some scores may be 0 (not available)
    by redistributing weight proportionally.
    """
    weights = {
        "resume": 0.25,
        "github": 0.20,
        "interview_qa": 0.20,
        "behavior": 0.15,
        "career_growth": 0.10,
        "consistency": 0.10
    }
    
    scores = {
        "resume": resume_score,
        "github": github_score,
        "interview_qa": interview_qa_score,
        "behavior": behavior_score,
        "career_growth": career_growth_score,
        "consistency": consistency_score
    }
    
    # Find active scores (non-zero)
    active = {k: v for k, v in scores.items() if v > 0}
    
    if not active:
        return 0.0
    
    # Redistribute weights for zero-scored dimensions
    active_weight_sum = sum(weights[k] for k in active)
    
    final_score = 0.0
    for key, score in active.items():
        normalized_weight = weights[key] / active_weight_sum
        final_score += normalized_weight * score
    
    return round(final_score, 1)


def calculate_behavior_score(
    communication_clarity: float,
    confidence: float,
    technical_explanation: float,
    consistency: float,
    professionalism: float
) -> float:
    """
    Interview Behavior Analysis KPI Score
    """
    score = (
        (0.30 * communication_clarity) +
        (0.25 * confidence) +
        (0.20 * technical_explanation) +
        (0.15 * consistency) +
        (0.10 * professionalism)
    )
    return round(score, 1)
