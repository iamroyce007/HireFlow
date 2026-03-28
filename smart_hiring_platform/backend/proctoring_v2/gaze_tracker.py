import numpy as np

class GazeTracker:
    def __init__(self):
        self.gaze_x_threshold = 0.05
        self.gaze_y_threshold = 0.05

    def get_gaze_direction(self, landmarks):
        if not landmarks:
            return 0.0, 0.0
        left_eye_center = landmarks[468]
        gaze_x = left_eye_center.x - 0.5 
        gaze_y = left_eye_center.y - 0.5
        return gaze_x, gaze_y

    def is_looking_away(self, gaze_x, gaze_y):
        return abs(gaze_x) > self.gaze_x_threshold or abs(gaze_y) > self.gaze_y_threshold
