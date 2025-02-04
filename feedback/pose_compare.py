import os
import sys
sys.path.append("./")
import cv2
import json
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from dance_scoring.util import draw_landmarks_on_image


def extract_pose_landmarks(result, image_width, image_height):
    landmarks = {}
    
    # 각 포즈의 첫 번째 검출 결과만 사용
    if result:
        pose_landmarks = result
        
        # 각 부위별 랜드마크 좌표 추출
        landmarks = {
            "head": {
                "0": {
                    "x": int(pose_landmarks[0].x * image_width),
                    "y": int(pose_landmarks[0].y * image_height),
                },
                "7": {
                    "x": int(pose_landmarks[7].x * image_width),
                    "y": int(pose_landmarks[7].y * image_height),
                },
                "8": {
                    "x": int(pose_landmarks[8].x * image_width),
                    "y": int(pose_landmarks[8].y * image_height),
                }
            },
            "left_arm": {
                "11": {
                    "x": int(pose_landmarks[11].x * image_width),
                    "y": int(pose_landmarks[11].y * image_height),
                },
                "13": {
                    "x": int(pose_landmarks[13].x * image_width),
                    "y": int(pose_landmarks[13].y * image_height),
                },
                "15": {
                    "x": int(pose_landmarks[15].x * image_width),
                    "y": int(pose_landmarks[15].y * image_height),
                }
            },
            "right_arm": {
                "12": {
                    "x": int(pose_landmarks[12].x * image_width),
                    "y": int(pose_landmarks[12].y * image_height),
                },
                "14": {
                    "x": int(pose_landmarks[14].x * image_width),
                    "y": int(pose_landmarks[14].y * image_height),
                },
                "16": {
                    "x": int(pose_landmarks[16].x * image_width),
                    "y": int(pose_landmarks[16].y * image_height),
                }
            },
            "left_leg": {
                "23": {
                    "x": int(pose_landmarks[23].x * image_width),
                    "y": int(pose_landmarks[23].y * image_height),
                },
                "25": {
                    "x": int(pose_landmarks[25].x * image_width),
                    "y": int(pose_landmarks[25].y * image_height),
                },
                "27": {
                    "x": int(pose_landmarks[27].x * image_width),
                    "y": int(pose_landmarks[27].y * image_height),
                }
            },
            "right_leg": {
                "24": {
                    "x": int(pose_landmarks[24].x * image_width),
                    "y": int(pose_landmarks[24].y * image_height),
                },
                "26": {
                    "x": int(pose_landmarks[26].x * image_width),
                    "y": int(pose_landmarks[26].y * image_height),
                },
                "28": {
                    "x": int(pose_landmarks[28].x * image_width),
                    "y": int(pose_landmarks[28].y * image_height),
                }
            },
            "left_foot": {
                "27": {
                    "x": int(pose_landmarks[27].x * image_width),
                    "y": int(pose_landmarks[27].y * image_height),
                },
                "29": {
                    "x": int(pose_landmarks[29].x * image_width),
                    "y": int(pose_landmarks[29].y * image_height),
                },
                "31": {
                    "x": int(pose_landmarks[31].x * image_width),
                    "y": int(pose_landmarks[31].y * image_height),
                }
            },
            "right_foot": {
                "28": {
                    "x": int(pose_landmarks[28].x * image_width),
                    "y": int(pose_landmarks[28].y * image_height),
                },
                "30": {
                    "x": int(pose_landmarks[30].x * image_width),
                    "y": int(pose_landmarks[30].y * image_height),
                },
                "32": {
                    "x": int(pose_landmarks[32].x * image_width),
                    "y": int(pose_landmarks[32].y * image_height),
                }
            }
        }
    
    return landmarks


def make_pose_jsons(img_path_list, detector, result_folder="./results"):
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)


    for p in img_path_list:
        image = cv2.imread(p)
        image_height, image_width = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = detector.detect(mp_image)
        pose_landmarks = extract_pose_landmarks(result, image_width, image_height)


        # make save folders
        file_name = os.path.splitext(os.path.basename(p))[0]
        result_path = os.path.join(result_folder, file_name)
        if not os.path.exists(result_path):
            os.mkdir(result_path)

        # json dumping
        json_path = os.path.join(result_path, "result.json")
        with open(json_path, 'w') as f:
            json.dump(pose_landmarks, f, indent=4)
        
        # save annotated image
        img_path = os.path.join(result_path, "result.png")
        output_image = draw_landmarks_on_image(
            image.copy(), 
            result, 
            landmark_color=(255, 0, 0),
            connection_color=(255, 150, 150) 
        )
        cv2.imwrite(img_path, output_image)