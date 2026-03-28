import cv2
import numpy as np

class HeadPoseEstimator:
    def __init__(self):
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])

    def get_pose(self, landmarks, w, h):
        if not landmarks:
            return 0.0, 0.0, 0.0
        idx = [1, 152, 33, 263, 61, 291]
        points_2d = []
        for i in idx:
            lm = landmarks[i]
            points_2d.append([lm.x * w, lm.y * h])
        points_2d = np.array(points_2d, dtype="double")
        focal_length = w
        center = (w/2, h/2)
        camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")
        dist_coeffs = np.zeros((4,1))
        (success, rotation_vector, translation_vector) = cv2.solvePnP(self.model_points, points_2d, camera_matrix, dist_coeffs)
        r_mat, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(r_mat)
        return angles[0], angles[1], angles[2]

    def is_turned_away(self, pitch, yaw):
        return abs(yaw) > 2.0 or abs(pitch) > 2.0
