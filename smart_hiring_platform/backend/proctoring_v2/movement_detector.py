import numpy as np
from collections import deque

class MovementDetector:
    def __init__(self, history_size=1, threshold=1.0):
        self.prev_frame = None
        self.history = deque(maxlen=history_size)
        self.threshold = threshold

    def get_motion_score(self, frame):
        if self.prev_frame is None:
            self.prev_frame = frame
            return 0.0
        diff = np.abs(frame.astype(float) - self.prev_frame.astype(float)).mean()
        self.history.append(diff)
        avg_diff = sum(self.history) / len(self.history)
        self.prev_frame = frame
        return avg_diff

    def is_excessive_movement(self, motion_score):
        return motion_score > self.threshold

