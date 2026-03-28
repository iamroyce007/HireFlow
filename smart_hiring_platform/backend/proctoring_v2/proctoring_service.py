"""
Real-Time AI Proctoring Engine v3.0
Uses: mediapipe FaceMesh + opencv-python + numpy
Temporal logic — no instant violations. All thresholds are frame-buffered.
"""

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from typing import Dict, List, Tuple, Optional


# ─── THRESHOLDS ───
GAZE_X_THRESHOLD = 0.18
GAZE_Y_THRESHOLD = 0.18
GAZE_FRAME_BUFFER = 15

HEAD_YAW_THRESHOLD = 18.0  # degrees
HEAD_FRAME_BUFFER = 12

MOVEMENT_THRESHOLD = 25.0
MOVEMENT_HISTORY_SIZE = 10
MOVEMENT_FRAME_BUFFER = 10

NO_FACE_FRAME_BUFFER = 10
MULTI_FACE_FRAME_BUFFER = 10

EYE_AR_THRESHOLD = 0.20
EYE_FRAME_BUFFER = 15

# ─── LANDMARK INDICES ───
# MediaPipe FaceMesh 468 landmarks
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]

LEFT_IRIS_CENTER = 468    # iris landmarks (refined, index 468-472)
RIGHT_IRIS_CENTER = 473   # iris landmarks (refined, index 473-477)

# Nose tip and face boundary points for head pose
NOSE_TIP = 1
FOREHEAD = 10
CHIN = 152
LEFT_CHEEK = 234
RIGHT_CHEEK = 454
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263


class ProctoringEngine:
    """Complete proctoring engine with temporal violation logic."""

    def __init__(self):
        # ─── MediaPipe Tasks API Setup ───
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        import os
        
        # Path to the model file
        model_path = os.path.join(os.path.dirname(__file__), "models", "face_landmarker.task")
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=2,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.IMAGE
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

        # ─── Temporal Counters ───
        self.gaze_frames = 0
        self.head_frames = 0
        self.movement_frames = 0
        self.no_face_frames = 0
        self.multi_face_frames = 0
        self.eye_closed_frames = 0

        # ─── Movement Detection ───
        self.prev_gray = None
        self.movement_history = deque(maxlen=MOVEMENT_HISTORY_SIZE)

        # ─── Violation Counts ───
        self.violations = {
            "no_face": 0,
            "multi_face": 0,
            "gaze": 0,
            "head_pose": 0,
            "movement": 0,
            "eyes_closed": 0,
        }

        # ─── Scores ───
        self.attention_score = 100.0
        self.stability_score = 100.0

    # ─────────────────────────────────
    # EYE ASPECT RATIO
    # ─────────────────────────────────
    def _eye_aspect_ratio(self, landmarks, eye_indices, w, h) -> float:
        """Compute Eye Aspect Ratio (EAR) from 6 eye landmarks."""
        pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
        # Vertical distances
        v1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
        v2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
        # Horizontal distance
        h_dist = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
        if h_dist == 0:
            return 0.0
        return (v1 + v2) / (2.0 * h_dist)

    # ─────────────────────────────────
    # GAZE DIRECTION
    # ─────────────────────────────────
    def _compute_gaze(self, landmarks, w, h) -> Tuple[float, float]:
        """Compute gaze direction using iris center relative to eye corners."""
        try:
            # Left eye
            l_inner = np.array([landmarks[362].x * w, landmarks[362].y * h])
            l_outer = np.array([landmarks[263].x * w, landmarks[263].y * h])
            l_iris = np.array([landmarks[LEFT_IRIS_CENTER].x * w, landmarks[LEFT_IRIS_CENTER].y * h])

            # Right eye
            r_inner = np.array([landmarks[133].x * w, landmarks[133].y * h])
            r_outer = np.array([landmarks[33].x * w, landmarks[33].y * h])
            r_iris = np.array([landmarks[RIGHT_IRIS_CENTER].x * w, landmarks[RIGHT_IRIS_CENTER].y * h])

            # Relative position of iris within eye (0=left_corner, 1=right_corner)
            l_eye_width = np.linalg.norm(l_outer - l_inner)
            r_eye_width = np.linalg.norm(r_outer - r_inner)

            if l_eye_width == 0 or r_eye_width == 0:
                return 0.0, 0.0

            # X: how far iris is from center of eye (normalized -0.5 to 0.5)
            l_ratio_x = (l_iris[0] - l_inner[0]) / l_eye_width - 0.5
            r_ratio_x = (r_iris[0] - r_inner[0]) / r_eye_width - 0.5
            gaze_x = (l_ratio_x + r_ratio_x) / 2.0

            # Y: vertical offset from eye center
            l_center_y = (l_inner[1] + l_outer[1]) / 2.0
            r_center_y = (r_inner[1] + r_outer[1]) / 2.0
            l_ratio_y = (l_iris[1] - l_center_y) / l_eye_width
            r_ratio_y = (r_iris[1] - r_center_y) / r_eye_width
            gaze_y = (l_ratio_y + r_ratio_y) / 2.0

            return float(gaze_x), float(gaze_y)
        except (IndexError, Exception):
            return 0.0, 0.0

    # ─────────────────────────────────
    # HEAD POSE ESTIMATION
    # ─────────────────────────────────
    def _estimate_head_pose(self, landmarks, w, h) -> Tuple[float, float, float]:
        """Approximate head pose (pitch, yaw, roll) from face landmarks."""
        try:
            # 2D image points
            image_points = np.array([
                [landmarks[NOSE_TIP].x * w, landmarks[NOSE_TIP].y * h],
                [landmarks[CHIN].x * w, landmarks[CHIN].y * h],
                [landmarks[LEFT_EYE_OUTER].x * w, landmarks[LEFT_EYE_OUTER].y * h],
                [landmarks[RIGHT_EYE_OUTER].x * w, landmarks[RIGHT_EYE_OUTER].y * h],
                [landmarks[LEFT_CHEEK].x * w, landmarks[LEFT_CHEEK].y * h],
                [landmarks[RIGHT_CHEEK].x * w, landmarks[RIGHT_CHEEK].y * h],
            ], dtype=np.float64)

            # 3D model points (generic face model)
            model_points = np.array([
                [0.0, 0.0, 0.0],          # Nose tip
                [0.0, -330.0, -65.0],      # Chin
                [-225.0, 170.0, -135.0],   # Left eye outer
                [225.0, 170.0, -135.0],    # Right eye outer
                [-150.0, -150.0, -125.0],  # Left cheek
                [150.0, -150.0, -125.0],   # Right cheek
            ], dtype=np.float64)

            # Camera matrix (approximate)
            focal_length = w
            center = (w / 2, h / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1],
            ], dtype=np.float64)

            dist_coeffs = np.zeros((4, 1))

            success, rotation_vec, translation_vec = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                return 0.0, 0.0, 0.0

            rotation_mat, _ = cv2.Rodrigues(rotation_vec)
            pose_mat = cv2.hconcat([rotation_mat, translation_vec])
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(
                cv2.hconcat([rotation_mat, translation_vec.reshape(3, 1)])
            )

            pitch = float(euler_angles[0][0])
            yaw = float(euler_angles[1][0])
            roll = float(euler_angles[2][0])

            return pitch, yaw, roll
        except Exception:
            return 0.0, 0.0, 0.0

    # ─────────────────────────────────
    # MOVEMENT DETECTION
    # ─────────────────────────────────
    def _detect_movement(self, gray_frame) -> float:
        """Frame differencing to detect excessive motion."""
        if self.prev_gray is None:
            self.prev_gray = gray_frame.copy()
            return 0.0

        diff = cv2.absdiff(self.prev_gray, gray_frame)
        movement = float(np.mean(diff))
        self.prev_gray = gray_frame.copy()

        self.movement_history.append(movement)
        avg_movement = float(np.mean(self.movement_history))
        return avg_movement

    # ─────────────────────────────────
    # DRAW VISUAL OVERLAY
    # ─────────────────────────────────
    def _draw_overlay(self, frame, face_count, gaze, head_pose, movement,
                      ear, violation, violation_type, landmarks_list) -> np.ndarray:
        """Draw HUD overlay on the frame."""
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Semi-transparent top bar
        cv2.rectangle(overlay, (0, 0), (w, 45), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # Status
        status_color = (0, 0, 255) if violation else (0, 255, 0)
        status_text = f"VIOLATION: {violation_type}" if violation else "STATUS: SECURE"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        # Face count
        fc_color = (0, 255, 0) if face_count == 1 else (0, 0, 255)
        cv2.putText(frame, f"FACES: {face_count}", (w - 160, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fc_color, 2)

        # Bottom info bar
        y_base = h - 10
        info_texts = [
            f"GAZE: ({gaze[0]:.2f}, {gaze[1]:.2f})",
            f"YAW: {head_pose[1]:.1f}°",
            f"MOVE: {movement:.1f}",
            f"EAR: {ear:.2f}",
        ]
        x_pos = 10
        for txt in info_texts:
            cv2.putText(frame, txt, (x_pos, y_base), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            x_pos += 180

        # Gaze arrow
        if face_count >= 1:
            cx, cy = w // 2, h // 2
            arrow_len = 80
            end_x = int(cx + gaze[0] * arrow_len * 10)
            end_y = int(cy + gaze[1] * arrow_len * 10)
            arrow_color = (0, 255, 255) if not violation else (0, 0, 255)
            cv2.arrowedLine(frame, (cx, cy), (end_x, end_y), arrow_color, 2, tipLength=0.3)

        # Draw face mesh points
        if landmarks_list:
            for pt in landmarks_list:
                px = int(pt["x"] * w)
                py = int(pt["y"] * h)
                dot_color = (0, 200, 0) if not violation else (0, 0, 255)
                cv2.circle(frame, (px, py), 1, dot_color, -1)

        return frame

    # ─────────────────────────────────
    # MAIN ANALYSIS PIPELINE
    # ─────────────────────────────────
    def analyze_frame(self, frame) -> Dict:
        """Process a single OpenCV BGR frame. Returns full analysis dict."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        import mediapipe as mp
        # Convert to MP Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        # Run MediaPipe Tasks API
        results = self.detector.detect(mp_image)

        face_count = 0
        gaze = (0.0, 0.0)
        head_pose = (0.0, 0.0, 0.0)
        ear = 1.0
        landmarks_list = []

        if getattr(results, 'face_landmarks', None):
            face_count = len(results.face_landmarks)

            # Use first face for analysis
            face_landmarks = results.face_landmarks[0]

            # Extract landmarks for frontend overlay
            for i, lm in enumerate(face_landmarks):
                if i % 3 == 0:  # every 3rd point to reduce payload
                    landmarks_list.append({"x": lm.x, "y": lm.y, "z": lm.z})

            # Gaze
            gaze = self._compute_gaze(face_landmarks, w, h)

            # Head pose
            head_pose = self._estimate_head_pose(face_landmarks, w, h)

            # Eye Aspect Ratio
            left_ear = self._eye_aspect_ratio(face_landmarks, LEFT_EYE_IDX, w, h)
            right_ear = self._eye_aspect_ratio(face_landmarks, RIGHT_EYE_IDX, w, h)
            ear = (left_ear + right_ear) / 2.0

        # Movement
        avg_movement = self._detect_movement(gray)

        # ─── TEMPORAL VIOLATION LOGIC ───
        violation = False
        violation_type = ""
        reason = ""

        # 1. No Face
        if face_count == 0:
            self.no_face_frames += 1
            if self.no_face_frames > NO_FACE_FRAME_BUFFER:
                violation = True
                violation_type = "NO_FACE"
                reason = "No face detected in camera for extended period"
                self.violations["no_face"] += 1
                self.no_face_frames = 0
        else:
            self.no_face_frames = max(0, self.no_face_frames - 2)  # decay

        # 2. Multiple Faces
        if face_count > 1:
            self.multi_face_frames += 1
            if self.multi_face_frames > MULTI_FACE_FRAME_BUFFER:
                violation = True
                violation_type = "MULTI_FACE"
                reason = "Multiple faces detected in camera"
                self.violations["multi_face"] += 1
                self.multi_face_frames = 0
        else:
            self.multi_face_frames = max(0, self.multi_face_frames - 2)

        # 3. Gaze
        if abs(gaze[0]) > GAZE_X_THRESHOLD or abs(gaze[1]) > GAZE_Y_THRESHOLD:
            self.gaze_frames += 1
            if self.gaze_frames > GAZE_FRAME_BUFFER:
                violation = True
                violation_type = "GAZE_AWAY"
                reason = "User looking away from screen for extended period"
                self.violations["gaze"] += 1
                self.gaze_frames = 0
        else:
            self.gaze_frames = max(0, self.gaze_frames - 1)

        # 4. Head Pose (yaw)
        _, yaw, _ = head_pose
        if abs(yaw) > HEAD_YAW_THRESHOLD:
            self.head_frames += 1
            if self.head_frames > HEAD_FRAME_BUFFER:
                violation = True
                violation_type = "HEAD_TURNED"
                reason = "Head turned away from camera for extended period"
                self.violations["head_pose"] += 1
                self.head_frames = 0
        else:
            self.head_frames = max(0, self.head_frames - 1)

        # 5. Movement
        if avg_movement > MOVEMENT_THRESHOLD:
            self.movement_frames += 1
            if self.movement_frames > MOVEMENT_FRAME_BUFFER:
                violation = True
                violation_type = "EXCESSIVE_MOVEMENT"
                reason = "Excessive movement detected in camera"
                self.violations["movement"] += 1
                self.movement_frames = 0
        else:
            self.movement_frames = max(0, self.movement_frames - 1)

        # 6. Eyes Closed
        if ear < EYE_AR_THRESHOLD and face_count >= 1:
            self.eye_closed_frames += 1
            if self.eye_closed_frames > EYE_FRAME_BUFFER:
                violation = True
                violation_type = "EYES_CLOSED"
                reason = "Eyes closed for extended period"
                self.violations["eyes_closed"] += 1
                self.eye_closed_frames = 0
        else:
            self.eye_closed_frames = max(0, self.eye_closed_frames - 1)

        # ─── Compute Scores ───
        # Attention: decays on violations, recovers on normal
        if violation:
            self.attention_score = max(0, self.attention_score - 8)
        else:
            self.attention_score = min(100, self.attention_score + 0.5)

        # Stability: based on movement
        if avg_movement > MOVEMENT_THRESHOLD * 0.6:
            self.stability_score = max(0, self.stability_score - 3)
        else:
            self.stability_score = min(100, self.stability_score + 0.8)

        return {
            "face_count": face_count,
            "gaze": {"x": round(gaze[0], 3), "y": round(gaze[1], 3)},
            "head_pose": {
                "pitch": round(head_pose[0], 1),
                "yaw": round(head_pose[1], 1),
                "roll": round(head_pose[2], 1),
            },
            "movement": round(avg_movement, 2),
            "ear": round(ear, 3),
            "scores": {
                "attention": round(self.attention_score, 1),
                "stability": round(self.stability_score, 1),
            },
            "violation": violation,
            "violation_type": violation_type,
            "reason": reason,
            "landmarks": landmarks_list,
            "violation_counters": self.violations,
            "temporal_state": {
                "gaze_frames": self.gaze_frames,
                "head_frames": self.head_frames,
                "movement_frames": self.movement_frames,
                "no_face_frames": self.no_face_frames,
                "multi_face_frames": self.multi_face_frames,
                "eye_closed_frames": self.eye_closed_frames,
            },
        }


class ProctoringServiceV2:
    """Production wrapper for FastAPI integration."""

    def __init__(self):
        self.engine = ProctoringEngine()
        print("[ProctoringServiceV2] Initialized with MediaPipe FaceMesh + OpenCV")

    def analyze_frame(self, frame) -> Dict:
        """Analyze a pre-decoded OpenCV frame."""
        return self.engine.analyze_frame(frame)

    def analyze_bytes(self, image_bytes: bytes) -> Dict:
        """Decode JPEG bytes from FastAPI UploadFile and analyze."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {
                "face_count": 0,
                "violation": True,
                "violation_type": "INVALID_FRAME",
                "reason": "Could not decode image frame",
                "scores": {"attention": 0, "stability": 0},
                "landmarks": [],
            }
        return self.engine.analyze_frame(frame)

    def get_annotated_frame(self, image_bytes: bytes) -> Optional[bytes]:
        """Decode, analyze, draw overlay, return annotated JPEG bytes."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return None

        result = self.engine.analyze_frame(frame)

        annotated = self.engine._draw_overlay(
            frame,
            face_count=result["face_count"],
            gaze=(result["gaze"]["x"], result["gaze"]["y"]),
            head_pose=(result["head_pose"]["pitch"], result["head_pose"]["yaw"], result["head_pose"]["roll"]),
            movement=result["movement"],
            ear=result["ear"],
            violation=result["violation"],
            violation_type=result["violation_type"],
            landmarks_list=result["landmarks"],
        )

        _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buffer.tobytes()
