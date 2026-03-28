import cv2

class CameraManager:
    """Handles CV2 VideoCapture and frame preprocessing."""
    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise Exception("Could not open webcam.")
        
    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        # Mirror image for natural feel
        frame = cv2.flip(frame, 1)
        return frame

    def release(self):
        self.cap.release()
