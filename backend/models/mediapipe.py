from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class mp_model:
    def __init__(self):
        self.base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
        self.options = vision.PoseLandmarkerOptions(
            base_options=self.base_options,
            output_segmentation_masks=False,
            running_mode=vision.RunningMode.IMAGE
        )
        self.detector = vision.PoseLandmarker.create_from_options(self.options)
    
    def get_detector(self):
        return self.detector
