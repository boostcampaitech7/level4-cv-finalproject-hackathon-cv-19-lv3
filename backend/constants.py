from enum import Enum

class FilePaths(Enum):
    ORIGIN_H5 = "video.h5"
    USER_H5 = "user.h5"
    ORIGIN_MP4 = "video.mp4"
    USER_MP4 = "user.mp4"

class ResponseMessages(Enum):
    H5FILE_LOAD_FAIL = "Failed to read pose data from {}: {}"
    POSE_EXTRACT_POSE_SUCCESS = "Success video pose extract"
    FEEDBACK_POSE_FAIL = "포즈가 감지되지 않아 피드백을 드릴 수 없습니다 (┬┬﹏┬┬)"
    FEEDBACK_CLEAR_SUCCESS = "Cache cleared for folder_id: {}"
