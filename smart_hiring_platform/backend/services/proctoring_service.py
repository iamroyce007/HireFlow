import cv2
import mediapipe as mp
import numpy as np
import time
from typing import Dict

# Global state for session-based motion tracking
SESSION_CV_STATE = {}

class ProctoringService:
    """
    Ultra-Strict Production Proctoring Service.
    Uses MediaPipe Face Mesh for Gaze Tracking, Head Pose, and NumPy for Noise/Motion.
    """

    def __init__(self, api_key: str):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self._min_interval = 0.5  # High-frequency processing (2 frames per second)

    def analyze_frame(self, session_id: str, image_bytes: bytes) -> Dict:
        """
        Production-grade frame analysis.
        Detects: Face Count, Gaze Direction, Head Orientation, and Motion Intensity.
        """
        now = time.time()
        
        # Persistent state for this session
        if session_id not in SESSION_CV_STATE:
            SESSION_CV_STATE[session_id] = {
                "last_check": 0,
                "prev_frame": None,
                "violation_count": 0
            }
        
        state = SESSION_CV_STATE[session_id]
        if now - state["last_check"] < self._min_interval:
            return {"status": "skipped", "reason": "Rate limited"}
        
        state["last_check"] = now

        try:
            # 1. Decode Frame
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                return {"status": "error", "reason": "Invalid image"}

            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            violation = False
            violation_type = None
            reason = "OK"

            # 2. FACE COUNT VALIDATION (STRICT)
            if not results.multi_face_landmarks:
                violation = True
                violation_type = "NO_FACE"
                reason = "FATAL ERROR: Candidate not visible in frame."
            elif len(results.multi_face_landmarks) > 1:
                violation = True
                violation_type = "MULTIPLE_FACES"
                reason = "FATAL ERROR: Multiple people detected in proctored zone."

            if not violation:
                # 3. ADVANCED GAZE & HEAD POSE (MediaPipe Face Mesh)
                mesh = results.multi_face_landmarks[0]
                
                # Extract key landmarks for pose (Nose tip, Chin, Left Eye, Right Eye, Left Mouth, Right Mouth)
                # Landmark indices: Nose=1, Chin=152, LeftEyeLeft=33, RightEyeRight=263, LeftMouth=61, RightMouth=291
                idx = [1, 152, 33, 263, 61, 291]
                points_2d = []
                for i in idx:
                    landmark = mesh.landmark[i]
                    points_2d.append([landmark.x * w, landmark.y * h])
                points_2d = np.array(points_2d, dtype="double")

                # Generic 3D model points
                model_points = np.array([
                    (0.0, 0.0, 0.0),             # Nose tip
                    (0.0, -330.0, -65.0),        # Chin
                    (-225.0, 170.0, -135.0),     # Left eye left corner
                    (225.0, 170.0, -135.0),      # Right eye right corner
                    (-150.0, -150.0, -125.0),    # Left Mouth corner
                    (150.0, -150.0, -125.0)      # Right mouth corner
                ])

                # Camera matrix
                focal_length = w
                center = (w/2, h/2)
                camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")
                dist_coeffs = np.zeros((4,1))
                
                (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, points_2d, camera_matrix, dist_coeffs)
                
                # Check Head Pose (Rotation)
                # We care about Pitch (up/down) and Yaw (left/right)
                r_mat, _ = cv2.Rodrigues(rotation_vector)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(r_mat)
                pitch, yaw, roll = angles[0], angles[1], angles[2]

                # STRICT THRESHOLDS: Pitch > 15 (too high/low), Yaw > 15 (looking left/right)
                if abs(yaw) > 18 or abs(pitch) > 18:
                    violation = True
                    violation_type = "HEAD_POSE_VIOLATION"
                    reason = "ERROR: Candidate not facing the screen."

                # 4. GAZE TRACKING (Iris landmarks)
                # MediaPipe Iris indices approx: Left=468, Right=473
                left_iris = mesh.landmark[468]
                right_iris = mesh.landmark[473]
                
                # If iris is too far from center of eye bounds, they are looking away
                # Simplified gaze check using iris horizontal position relative to head yaw
                if not violation:
                    # If yaw is balanced but iris is skewed, user is looking away
                    if (left_iris.x < 0.45 or left_iris.x > 0.55) and abs(yaw) < 10:
                         violation = True
                         violation_type = "GAZE_VIOLATION"
                         reason = "ERROR: Candidate looking away from the screen."

            # 5. MOVEMENT DETECTION
            motion_score = 0
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small_gray = cv2.resize(gray, (160, 120))
            if state["prev_frame"] is not None:
                diff = cv2.absdiff(state["prev_frame"], small_gray)
                motion_score = (np.count_nonzero(diff > 30) / diff.size) * 100
                if motion_score > 35.0 and not violation:
                    violation = True
                    violation_type = "MOVEMENT_ERROR"
                    reason = "ERROR: Excessive movement detected."
            state["prev_frame"] = small_gray

            if violation:
                state["violation_count"] += 1

            return {
                "status": "checked",
                "violation": violation,
                "violation_type": violation_type,
                "reason": reason,
                "violation_count": state["violation_count"],
                "gaze_details": {"yaw": round(yaw, 1) if 'yaw' in locals() else 0, "pitch": round(pitch, 1) if 'pitch' in locals() else 0},
                "motion_score": round(motion_score, 2),
                "timestamp": now
            }

        except Exception as e:
            print(f"[ProctoringService] Error: {e}")
            return {"status": "error", "reason": str(e)}
