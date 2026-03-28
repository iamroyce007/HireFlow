class ViolationEngine:
    def __init__(self):
        self.violations = {k: 0 for k in ["no_face", "multi_face", "gaze", "movement", "head_pose", "eyes_closed", "attention"]}
        self.reasons = {
            "no_face": "NO FACE DETECTED IN CAMERA",
            "multi_face": "MULTIPLE FACES DETECTED",
            "gaze": "USER LOOKING AWAY FROM CAMERA",
            "movement": "MOVEMENT DETECTED IN CAMERA",
            "head_pose": "HEAD MOVEMENT DETECTED",
            "eyes_closed": "EYE CONTACT LOST",
            "attention": "ATTENTION SCORE BREACHED"
        }
        self.counters = {k: 0 for k in self.violations.keys()}
        self.thresholds = {
            "no_face": 0,
            "multi_face": 0,
            "gaze": 0,
            "movement": 0,
            "head_pose": 0,
            "eyes_closed": 0,
            "attention": 0
        }

    def update(self, alerts):
        v_triggered = False
        v_type = None
        v_reason = ""
        for key, active in alerts.items():
            if active:
                self.counters[key] += 1
                if self.counters[key] >= self.thresholds.get(key, 0):
                    self.violations[key] += 1
                    self.counters[key] = 0
                    v_triggered = True
                    v_type = key
                    v_reason = self.reasons.get(key, key.upper())
            else:
                self.counters[key] = 0
        return v_triggered, v_type, v_reason

    def get_summary(self):
        return self.violations
