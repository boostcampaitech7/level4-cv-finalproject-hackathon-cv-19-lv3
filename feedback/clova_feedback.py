import requests
import json
from pose_feedback import json_to_prompt


def make_base_clova_inputs(original_json_path, user_json_path):
    '''
    각 json파일을 input으로 넣어주면 clova에 넣을 형식으로 변환해서 return
    각 json은 pose_compare.extract_pose_landmarks로 정제되어 저장된 값이어야함
    '''
    result_json = json_to_prompt(original_json_path, user_json_path)
    return str(result_json)

class CompletionExecutor:
    def __init__(self, host, api_key, request_id):
        self._host = host
        self._api_key = api_key
        self._request_id = request_id
        self.res_string = "event:result"

    def execute(self, completion_request):
        headers = {
            'Authorization': self._api_key,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id,
            'Content-Type': 'application/json; charset=utf-8',
            'Accept': 'text/event-stream'
        }

        with requests.post(self._host + '/serviceapp/v2/tasks/k6mzcsvl/chat-completions',
                           headers=headers, json=completion_request, stream=True) as r:
            check = False
            for line in r.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if check:
                        return line[5:]

                    if line.startswith(self.res_string):
                        check = True


def base_feedback_model(content, api_path='./CLOVA_API'):
    '''
    단순히 서비스앱을 테스트하고 챗봇의 응답을 print하는 함수입니다.
    inputs:
        content : user가 input으로 넣을 문장
        request_id : 테스트앱 발급 시 나오는 request id 적으면 됩니다
        api_path : api key와 SubAccount 정보를 담은 파일 경로
    
    Returns:
        None
    '''
    with open(api_path, 'r') as f:
        api_info = json.load(f)
    completion_executor = CompletionExecutor(
        host='https://clovastudio.stream.ntruss.com',
        api_key=api_info['service_api_key'],
        request_id='8f6c0931d83e4901a2090a762e846b1b'
    )

    # chatbot prompt
    preset_text = [{"role":"system","content":"{\r\n    \"ANGLE DIFFERENCES\": {\r\n        \"Explation\": \"**모범 포즈와 사용자의 포즈에서 ANGLE_FEATURES를 각각 구해 그 차이를 계산한 값.**\",\r\n        \"head difference\": [\"{'target face angle'-'user face angle'}\", \"양수값일 경우 'target face angle'보다 user의 머리가 더 오른쪽으로 기울어져 있다는 뜻이다.\"],\r\n        \"shoulder difference\": [\"{'target shoulder angle'-'user shoulder angle'}\", \"양수값일 경우 'target shoulder angle'보다 user의 어깨가 더 오른쪽으로 기울어져 있다는 뜻이다.\"],\r\n        \"left arm angle difference\": [\"{'target left arm angle'-'user left arm angle'}\", \"양수값일 경우 'target left arm angle'보다 user의 왼팔이 더 반시계 방향으로 difference만큼, 혹은 시계방향으로 (360-difference)만큼 돌아가 있다는 뜻이다.\"],\r\n        \"right arm angle difference\": [\"{'target right arm angle'-'user right arm angle'}\", \"양수값일 경우 'target right arm angle'보다 user의 오른팔이 더 반시계 방향으로 difference만큼, 혹은 시계방향으로 (360-difference)만큼 돌아가 있다는 뜻이다.\"],\r\n        \"left elbow angle difference\": [{'target left elbow angle'-'user left elbow angle'}, \"양수값일 경우 'target left elbow angle'보다 user의 왼팔이 더 굽혀져있다는 뜻이다.\"],\r\n        \"right elbow angle difference\": [{'target right elbow angle'-'user right elbow angle'}, \"양수값일 경우 'target right elbow angle'보다 user의 오른팔이 더 굽혀져있다는 뜻이다.\"],\r\n        \"left leg angle difference\": [\"{'target left leg angle'-'target right leg angle'}\", \"양수값일 경우 'target left leg angle'보다 user의 왼다리가 더 반시계 방향으로 difference만큼, 혹은 시계방향으로 (360-difference)만큼 돌아가 있다는 뜻이다.\"],\r\n        \"right leg angle difference\": [\"{'user right leg angle'-'target right leg angle'}\", \"양수값일 경우 'target right leg angle'보다 user의 오른다리가 더 반시계 방향으로 difference만큼, 혹은 시계방향으로 (360-difference)만큼 돌아가 있다는 뜻이다.\"],\r\n        \"left knee angle difference\": [\"{'target left knee angle'-'user left knee angle'}\", \"양수값의 경우 'target left knee angle'보다 user의 왼다리가 더 굽혀져있다는 뜻이다.\"],\r\n        \"right knee angle difference\": [\"{'target right knee angle'-'user right knee angle'}\", \"양수값의 경우 'target right knee angle'보다 user의 오른다리가 더 굽혀져있다는 뜻이다.\"]\r\n    },\r\n    \"ROLE\": '''\r\n        당신은 서로 다른 두 사람의 Pose Difference 정보를 기반으로 피드백을 주는 댄스 서포터 AI입니다.\r\n        입력값으로는 두 사람 포즈의 차이에 대한 설명이 총 10가지 주어집니다.\r\n        * difference의 절댓값이 30 이상일 경우에 대해서만 피드백을 주도록 합니다. difference의 절댓값이 30 이하의 경우에는 피드백을 전달하지 않도록 합니다.\r\n        * 구체적인 수치를 나타내기보다는, 단순한 문장으로 바꾸어 표현합니다.(예시 - 왼쪽 팔꿈치는 70도만큼 덜 굽혀져 있습니다 -> 왼쪽 팔꿈치를 더 펴주세요.)\r\n    '''\r\n}"},
                   {"role":"user","content":content}]

    request_data = {
        'messages': preset_text,
        'topP': 0.8,
        'topK': 0,
        'maxTokens': 1024,
        'temperature': 0.5,
        'repeatPenalty': 5.0,
        'stopBefore': [],
        'includeAiFilters': True,
        'seed': 0
    }
    result_dict = completion_executor.execute(request_data)
    result_dict = json.loads(result_dict)
    content = result_dict['message']['content']
    return content

if __name__=="__main__":
    import pandas as pd
    test_df = pd.read_csv("clova_datasets/test_dataset_5_instruction_fixed.csv")
    content = "{'head_difference': 56, 'shoulder_difference': 28, 'left_arm_angle_difference': 23, 'right_arm_angle_difference': -19, 'left_elbow_angle_difference': -53, 'right_elbow_angle_difference': -2, 'left_leg_angle_difference': -53, 'right_leg_angle_difference': 69, 'left_knee_angle_difference': 41, 'right_knee_angle_difference': -84}"
    print(base_feedback_model(content=content))