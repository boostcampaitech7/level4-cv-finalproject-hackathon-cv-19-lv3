import os
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from util import draw_landmarks_on_image, download_model
import warnings
from tqdm import tqdm


class PoseDetector:
    def __init__(self, model_size=2, mode = 'IMAGE'):
        """
        model size : 0 ~ 2(int) 모델사이즈. 클수록 큰모델
        running_mode : video inference를 수행해야할 시 running_mode를 video로 설정 (video or image)
        mode : default mode setting (IMAGE or VIDEO)
        """
        self.model_path = download_model(model_size)
        self.base_options = python.BaseOptions(self.model_path)

        if mode == 'IMAGE':
            self.options = vision.PoseLandmarkerOptions(
                base_options=self.base_options,
                output_segmentation_masks=True, running_mode=vision.RunningMode.IMAGE)
        else:
            self.options = vision.PoseLandmarkerOptions(
                base_options=self.base_options,
                output_segmentation_masks=True, running_mode=vision.RunningMode.VIDEO)
        self.detector = vision.PoseLandmarker.create_from_options(self.options)
        self._last_shape = (None, None)
    
    @property
    def last_shape(self):
        return self._last_shape
    @last_shape.setter
    def last_shape(self, shape):
        if len(shape) != 2:
            raise ValueError("shape should be (H, W) Tuple")
        self._last_shape = shape

    def reset_detector(self):
        self.detector = vision.PoseLandmarker.create_from_options(self.options)
    

    def get_detection(self, img_path, landmarks_c=(234,63,247), connection_c=(117,249,77), 
                    thickness=3, circle_r=3, display=False):
        if self.detector._running_mode != vision.RunningMode.IMAGE:
            self.detector._running_mode = vision.RunningMode.IMAGE

        # Read the input image
        if isinstance(img_path, str) :
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else :
            image = img_path
        if image is None:
            raise ValueError(f"image path {img_path} is wrong")
        # image = cv2.resize(image, (1024, 1024))

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        detection_result = self.detector.detect(mp_image)
        
        xmin, xmax, ymin, ymax, zmin, zmax = 0, 0, 0, 0, 0, 0
        if detection_result.pose_landmarks:
            for landmark in detection_result.pose_landmarks[0]:
                if xmin == 0:
                    xmin, ymin, zmin = landmark.x, landmark.y, landmark.z
                else:
                    xmin, xmax, ymin, ymax, zmin, zmax = min(xmin, landmark.x), max(xmax, landmark.x), min(ymin, landmark.y), max(ymax, landmark.y), min(zmin, landmark.z), max(zmax, landmark.z)
            annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result, 
                                                      landmarks_c=landmarks_c, connection_c=connection_c, 
                                                      thickness=thickness, circle_r=circle_r)
            
            boxsize = (xmin, xmax, ymin, ymax, zmin, zmax)
            boxsize = np.array([boxsize[2 * i + 1] - boxsize[2 * i] for i in range(3)])
        
            if display:
                plt.imshow(annotated_image)
                plt.show()
        else:
            warnings.warn("there is no pose_landmarks in the image!!")
            return None, None, None, None
        return detection_result.pose_landmarks[0], detection_result.segmentation_masks, annotated_image, boxsize
    

    def estimPose_video(self, input_file, do_resize=True, landmarks_c=(234,63,247), connection_c=(117,249,77), 
                    thickness=1, circle_r=1):
        if self.detector._running_mode != vision.RunningMode.VIDEO:
            self.detector._running_mode = vision.RunningMode.VIDEO

        # Initialize the VideoCapture object to read from a video stored in the disk.
        video = cv2.VideoCapture(input_file)

        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(video.get(cv2.CAP_PROP_FPS))
        print("video information!!")
        print("FPS: ", fps)
        print("total frame length: ", total_frames)

        frame_duration = int(1000 / fps)
        frames = []
        original_video_frames = []
        only_skeleton_frames = []
        
        all_landmarks = []
        for i in tqdm(range(total_frames)):
            # Read a frame.
            ok, frame = video.read()

            # Check if frame is not read properly.
            if not ok:
                # Break the loop.
                break
            
            frame_timestamp_ms = i * frame_duration
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Get the width and height of the frame
            frame_height, frame_width, _ =  frame.shape
            self.last_shape = (frame_height, frame_width)

            # Resize the frame while keeping the aspect ratio.
            if do_resize:
                frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            pose_landmarker_result = self.detector.detect_for_video(mp_image, frame_timestamp_ms)

            if pose_landmarker_result.pose_landmarks:
                landmarks = pose_landmarker_result.pose_landmarks[0]

                annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), pose_landmarker_result,
                                                        landmarks_c=landmarks_c, connection_c=connection_c,
                                                        thickness=thickness, circle_r=circle_r)
                skeleton = draw_landmarks_on_image(np.zeros_like(mp_image.numpy_view()), pose_landmarker_result,
                                                landmarks_c=landmarks_c, connection_c=connection_c,
                                                thickness=thickness, circle_r=circle_r)
            else:
                landmarks = None
                annotated_image = frame.copy()
                skeleton = np.zeros_like(mp_image.numpy_view())
            

            original_video_frames.append(frame.copy())
            frames.append(annotated_image)
            all_landmarks.append(landmarks)
            only_skeleton_frames.append(skeleton)
        return original_video_frames, only_skeleton_frames, frames, all_landmarks