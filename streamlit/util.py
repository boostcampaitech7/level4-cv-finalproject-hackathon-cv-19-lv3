import numpy as np
from keypoint_map import KEYPOINT_MAPPING
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


# 간격 추가하여 두 프레임 이어붙이기
def concat_frames_with_spacing(frames, spacing=20, color=(0, 0, 0)):
    # 프레임 높이와 동일한 간격 이미지를 생성
    spacer = np.full((frames[0].shape[0], spacing, 3), color, dtype=np.uint8)
    
    # 프레임 + 간격 + 프레임 이어붙이기
    final_frames = []
    for frame in frames[:-1]:
        final_frames.append(frame)
        final_frames.append(spacer)
    final_frames.append(frames[-1])
    combined_frame = np.hstack(final_frames)

    return combined_frame


def landmarks_to_dict(all_landmarks):
    landmark_dict = {}
    
    for i, landmarks in enumerate(all_landmarks):
        d = {j: {
                "name": KEYPOINT_MAPPING[j],
                "x": landmarks[j][0],
                "y": landmarks[j][1],
                "z": landmarks[j][2],
                "visibility": landmarks[j][3]
            } for j in KEYPOINT_MAPPING.keys()}
        landmark_dict[i] = d
    return landmark_dict


def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image