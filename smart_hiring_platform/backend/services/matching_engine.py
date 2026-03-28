def match_candidates_to_job(candidates: list, required_skills: list) -> list:
    """
    Matches candidates against required job skills using actual skill overlap scoring.
    Returns candidates sorted by match percentage.
    """
    if not required_skills or not candidates:
        return candidates
    
    required_set = set(s.lower().strip() for s in required_skills)
    
    matched = []
    for candidate in candidates:
        candidate_skills = candidate.get("skills", [])
        if not candidate_skills:
            candidate_skills = []
        
        candidate_set = set(s.lower().strip() for s in candidate_skills)
        
        # Calculate overlap
        overlap = candidate_set & required_set
        match_pct = round((len(overlap) / len(required_set)) * 100, 1) if required_set else 0
        
        matched_entry = {
            **candidate,
            "match_percentage": match_pct,
            "matched_skills": list(overlap),
            "missing_skills": list(required_set - candidate_set)
        }
        matched.append(matched_entry)
    
    # Sort by match percentage descending, then by composite score
    matched.sort(key=lambda x: (x["match_percentage"], x.get("composite_score", 0)), reverse=True)
    
    return matched
