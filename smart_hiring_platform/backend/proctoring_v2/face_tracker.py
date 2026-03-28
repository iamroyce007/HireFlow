import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import os

class FaceTrackerV2:
    """Modern MediaPipe Tasks Face Landmarker."""
    def __init__(self, model_path="proctoring_v2/models/face_landmarker.task"):
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=2,
            min_face_detection_confidence=0.6,
            min_face_presence_confidence=0.6,
            min_tracking_confidence=0.6,
            running_mode=vision.RunningMode.IMAGE
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def process_frame(self, frame):
        """Processes frame and returns FaceLandmarkerResult."""
        # Convert to MP Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = self.detector.detect(mp_image)
        return result

    def get_face_count(self, result):
        if not result.face_landmarks:
            return 0
        return len(result.face_landmarks)
