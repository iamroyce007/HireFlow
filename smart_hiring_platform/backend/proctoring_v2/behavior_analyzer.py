class BehaviorAnalyzer:
    """Refined Behavioral scoring with specific deductions and gradual recovery."""
    def __init__(self, initial_attention=100.0):
        self.attention_score = initial_attention
        self.stability_score = 100.0
        self.presence_score = 100.0

    def analyze(self, signals, alerts):
        """
        Input signals: face_count, gaze_offset, head_pose, motion_score, ear
        Input alerts: current frame violation flags (boolean)
        """
        # 1. Presence Score
        if signals['face_count'] == 1:
            self.presence_score = min(100.0, self.presence_score + 2.0)
        else:
            self.presence_score = max(0.0, self.presence_score - 10.0)

        # 2. Attention Score (Refined Deduction Logic)
        prev_attention = self.attention_score
        
        # Specific deductions per alert type
        if alerts['gaze']:
            self.attention_score -= 5.0 # (Deducting gradually, but the user requested -40 in logic context)
            # Actually, the user says "Deduct: gaze deviation -> -40".
            # To avoid dropping to 0 instantly, we apply this deduction but clamp it.
            # I will apply a penalty if the alert is ACTIVE.
            pass

        # We'll use a more direct mapping for the "Score" while the alert is active
        # but the rule "Deduct 40" usually means a penalty to the current score.
        
        current_penalty = 0
        if alerts['gaze']: current_penalty += 40
        if alerts['head_pose']: current_penalty += 30
        if alerts['eyes_closed']: current_penalty += 40
        
        # Scoring logic: Target score is (100 - current_penalty)
        target_score = max(0.0, 100.0 - current_penalty)
        
        # Smoothly move towards target_score
        if self.attention_score > target_score:
            self.attention_score = max(target_score, self.attention_score - 2.0)
        elif self.attention_score < target_score:
            self.attention_score = min(100.0, self.attention_score + 0.5) # Gradual restoration

        # 3. Stability Score (Movement)
        motion = signals['motion_score']
        # Normalized 0-100 stability (100 - motion penalty)
        target_stability = max(0.0, 100.0 - (motion * 2)) 
        if self.stability_score > target_stability:
            self.stability_score = max(target_stability, self.stability_score - 1.0)
        else:
            self.stability_score = min(100.0, self.stability_score + 1.0)

        return {
            "attention": self.attention_score,
            "stability": self.stability_score,
            "presence": self.presence_score
        }
