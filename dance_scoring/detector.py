import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import warnings
from tqdm import tqdm
from util import draw_landmarks_on_image
from keypoint_map import KEYPOINT_MAPPING, NORMALIZED_LANDMARK_KEYS


class PoseDetector:
    def __init__(self):
        # MediaPipe Pose 초기화
        mp_pose = mp.solutions.pose
        self.pose = mp_pose.Pose()
    
    def get_image_landmarks(self, img_path, landmarks_c=(234,63,247), connection_c=(117,249,77), 
                    thickness=3, circle_r=3, display=False):
        '''
        image에 대한 pose landmarks 추출을 수행하는 메서드입니다.
        inputs:
            - img_path : 이미지 경로 혹은 cv2로 읽어온 numpy array
            - landmarks_c, connection_c, ... : image에 keypoint를 draw할 때 색, 두께를 지정
        
        returns:
            - 만약 이미지 상에 사람이 없을 경우 None, None, None, None을 반환
            - 랜드마크가 존재할 경우 다음을 반환
                - pose_landmarks : list(landmark(x, y, z, visibility))
                - segmentation_mask
                - annotated_image(np.array) : numpy array(H, W, C)
                - boxsize(list) : 각 좌표축 별 landmark의 min, max값의 차이를 통해 [x, y, z] 박스 사이즈를 계산. [dx, dy, dz]
        '''
        # Read the input image
        if isinstance(img_path, str) :
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else :
            image = img_path
        if image is None:
            raise ValueError(f"image path {img_path} is wrong")
        
        detection_result = self.pose.process(image)
        
        xmin, xmax, ymin, ymax, zmin, zmax = 0, 0, 0, 0, 0, 0
        if detection_result.pose_landmarks:
            for landmark in detection_result.pose_landmarks.landmark:
                if xmin == 0:
                    xmin, ymin, zmin = landmark.x, landmark.y, landmark.z
                else:
                    xmin, xmax, ymin, ymax, zmin, zmax = min(xmin, landmark.x), max(xmax, landmark.x), min(ymin, landmark.y), max(ymax, landmark.y), min(zmin, landmark.z), max(zmax, landmark.z)
            annotated_image = draw_landmarks_on_image(image, detection_result.pose_landmarks.landmark, 
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
        return detection_result.pose_landmarks.landmark, detection_result.segmentation_mask, annotated_image, boxsize
    

    def get_video_landmarks(self, video_path, do_resize=True, resize_shape=None):
        '''
        video에 대한 landmark 추출을 진행하는 메서드

        inputs:
            - video_path(str) : video 경로(str)
            - do_resize(bool) : 리사이즈 할지 여부. resize_shpae가 지정되어있지 않으면 비율에 맞춰서 height를 640으로 조정
            - resize_shape : Tuple(height, width)
        
        outputs:
            - original_video_frames : list(numpy.array). video.read()로 읽어온 frame을 담은 list
            - pose_landmarker_results : PoseLandmarks. landmarks
        '''

        # Initialize the VideoCapture object to read from a video stored in the disk.
        video = cv2.VideoCapture(video_path)

        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(video.get(cv2.CAP_PROP_FPS))
        print("video information!!")
        print("FPS: ", fps)
        print("total frame length: ", total_frames)

        original_video_frames = []
        pose_landmarker_results = []

        for i in tqdm(range(total_frames)):
            # Read a frame.
            ok, frame = video.read()

            # Check if frame is not read properly.
            if not ok:
                # Break the loop.
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Get the width and height of the frame
            frame_height, frame_width, _ =  frame.shape
            img_shape = (frame_height, frame_width)

            # Resize the frame while keeping the aspect ratio.
            if do_resize:
                if resize_shape:
                    frame = cv2.resize(frame, resize_shape[1], resize_shape[0])
                else:
                    frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))

            pose_landmarker_result = self.pose.process(frame)
            original_video_frames.append(frame.copy())
            pose_landmarker_results.append(pose_landmarker_result)

        return original_video_frames, pose_landmarker_results, img_shape, fps


def get_overlap_from_landmarks(
        pose_landmarker_result, 
        original_video_frame,
        return_shape=None,
        landmarks_c=(234,63,247), connection_c=(117,249,77), 
        thickness=1, circle_r=1
    ):
    ann_image = draw_landmarks_on_image(
        original_video_frame, pose_landmarker_result,
        landmarks_c=landmarks_c, connection_c=connection_c,
        thickness=thickness, circle_r=circle_r
    )
    if return_shape:
        ann_image = cv2.resize(ann_image, return_shape)
    return ann_image


def get_skeleton_from_landmarks(
        pose_landmarker_result,
        original_video_frame,
        return_shape=None,
        landmarks_c=(234,63,247), connection_c=(117,249,77), 
        thickness=1, circle_r=1
    ):
    skeleton_image = draw_landmarks_on_image(
        np.zeros_like(original_video_frame), pose_landmarker_result,
        landmarks_c=landmarks_c, connection_c=connection_c,
        thickness=thickness, circle_r=circle_r
    )
    if return_shape:
        ann_image = cv2.resize(ann_image, return_shape)
    return skeleton_image


def post_process_pose_landmarks(pose_landmarks_results, fill_value=0.99999):
    # pose landmark result로부터 list(landmarks)의 형태를 만듦. 만약 landmarks가 없다면 None으로 채움
    if isinstance(pose_landmarks_results[0].pose_landmarks, list):
        pose_landmarks_results_list =  [res.pose_landmarks[0] if res.pose_landmarks else None for res in pose_landmarks_results]
    else:
        pose_landmarks_results_list = [res.pose_landmarks.landmark if res.pose_landmarks else None for res in pose_landmarks_results]
    
    # fill None frame pose_landmarks and None keypoints
    NormalizedLandmark = namedtuple('NormalizedLandmark', NORMALIZED_LANDMARK_KEYS)
    none_fill_value = [NormalizedLandmark(**{k:fill_value for k in NORMALIZED_LANDMARK_KEYS}) for _ in range(len(KEYPOINT_MAPPING))]
    for i in range(len(pose_landmarks_results_list)):
        if pose_landmarks_results_list[i] is None:
            pose_landmarks_results_list[i] = none_fill_value
    return pose_landmarks_results_list