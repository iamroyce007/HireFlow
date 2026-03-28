import numpy as np

class EyeMonitor:
    """Detect eye closure (drowsiness/cheating) via EAR for Tasks API."""
    def __init__(self, threshold=0.18):
        self.threshold = threshold
        # Indices for EAR calculation
        self.left_eye_idx = [33, 160, 158, 133, 153, 144]
        self.right_eye_idx = [362, 385, 387, 263, 373, 380]

    def get_ear(self, landmarks):
        """Compute average Eye Aspect Ratio (EAR)."""
        if not landmarks:
            return 1.0
            
        def calculate_ear(eye_landmarks, landmarks):
            # Distance between vertical landmarks
            p2 = landmarks[eye_landmarks[1]]
            p6 = landmarks[eye_landmarks[5]]
            p3 = landmarks[eye_landmarks[2]]
            p5 = landmarks[eye_landmarks[4]]
            dist_v1 = np.sqrt((p2.x - p6.x)**2 + (p2.y - p6.y)**2)
            dist_v2 = np.sqrt((p3.x - p5.x)**2 + (p3.y - p5.y)**2)
            
            # Distance between horizontal landmarks
            p1 = landmarks[eye_landmarks[0]]
            p4 = landmarks[eye_landmarks[3]]
            dist_h = np.sqrt((p1.x - p4.x)**2 + (p1.y - p4.y)**2)
            
            return (dist_v1 + dist_v2) / (2.0 * dist_h)

        left_ear = calculate_ear(self.left_eye_idx, landmarks)
        right_ear = calculate_ear(self.right_eye_idx, landmarks)
        
        return (left_ear + right_ear) / 2.0

    def is_eyes_closed(self, ear):
        # Strict threshold for closure
        return ear < self.threshold
