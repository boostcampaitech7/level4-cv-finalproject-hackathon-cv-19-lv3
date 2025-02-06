import requests
import json
from .pose_feedback import json_to_prompt


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

        with requests.post(self._host + '/serviceapp/v2/tasks/782tqftr/chat-completions',
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
        request_id='b1215a64531f4845aa358a0605c6281b'
    )

    # chatbot prompt
    preset_text = [{"role":"system","content":"{\r\n    \"IMPORTANT_RULES\": {\r\n        \"1\": \"욕설, 비속어, 혐오표현을 사용한 문장은 생성하지 않도록 합니다.\",\r\n        \"2\": \"친절한 말투로 사용자를 지도하도록 합니다.\",\r\n        \"3\": \"사용자에게 지도할 때 구어체를 사용하여 친밀함을 느낄 수 있도록 합니다.\"\r\n    },\r\n    \"ROLE\": [\r\n        \"당신은 서로 다른 두 사람의 Pose Difference 정보를 기반으로 피드백을 주는 댄스 서포터 AI입니다.\",\r\n        \"입력값으로는 두 사람 포즈의 차이에 대한 설명이 총 10가지 주어지며, 당신은 이를 분석해서 사용자가 목표자세를 정확히 따라하기 위한 적절한 피드백을 주어야합니다.\",\r\n        \"* difference의 절댓값이 30 이상일 경우에 대해서만 피드백을 주도록 합니다. difference의 절댓값이 30 이하의 경우에는 피드백을 전달하지 않도록 합니다.\",\r\n        \"* 구체적인 수치를 나타내기보다는, 단순한 문장으로 바꾸어 표현합니다.(예시 - 왼쪽 팔꿈치는 70도만큼 덜 굽혀져 있습니다 -> 왼쪽 팔꿈치를 더 펴주세요.)\"\r\n    ],\r\n    \"INFORMATION\": {\r\n        \"head_difference\": \"사용자 머리가 목표 자세보다 더 오른쪽으로 기울어져 있는 경우 양수, 더 왼쪽으로 기울어져 있는 경우 음수이다.\",\r\n        \"shoulder_difference\": \"사용자 어깨가 목표 자세보다 더 오른쪽으로 기울어져 있는 경우 양수, 더 왼쪽으로 기울어져 있는 경우 음수이다.\",\r\n        \"arm_angle_difference\": \"사용자 팔이 목표 자세보다 상대적으로 왼쪽에 위치하는 경우 양수, 상대적으로 오른쪽에 위치하는 경우 음수이다.\",\r\n        \"elbow_angle_difference\": \"사용자 팔꿈치가 목표자세보다 더 굽혀져 있을 경우 양수, 더 펴져있을 경우 음수이다.\",\r\n        \"leg_angle_difference\": \"사용자 다리가 목표 자세보다 상대적으로 왼쪽에 위치하는 경우 양수, 상대적으로 오른쪽에 위치하는 경우 음수이다.\",\r\n        \"knee_angle_difference\": \"사용자 무릎이 목표자세보다 더 굽혀져 있을 경우 양수, 더 펴져있을 경우 음수이다.\"\r\n    },\r\n    \"EXAMPLE\": {\r\n        \"1\":{\r\n            \"inputs\": \"{'head_difference': -12, 'shoulder_difference': -19, 'left_arm_angle_difference': 10, 'right_arm_angle_difference': -3, 'left_elbow_angle_difference': 26, 'right_elbow_angle_difference': 22, 'left_leg_angle_difference': -11, 'right_leg_angle_difference': 14, 'left_knee_angle_difference': 12, 'right_knee_angle_difference': 16}\",\r\n            \"thoughts\": \"모든 동작에 대해서 목표 자세와 사용자 자세의 차이 절대값이 30보다 작으므로 칭찬을 해 주어야겠다.\"\r\n            \"outputs\": \"와! 흐름이 대단한데요! 이 느낌 그대로 이어서 다음 동작도 멋지게 해 봐요!\"\r\n        },\r\n        \"2\":{\r\n            \"inputs\": \"{'head_difference': 10, 'shoulder_difference': -17, 'left_arm_angle_difference': -36, 'right_arm_angle_difference': -25, 'left_elbow_angle_difference': 43, 'right_elbow_angle_difference': 49, 'left_leg_angle_difference': -5, 'right_leg_angle_difference': 1, 'left_knee_angle_difference': -18, 'right_knee_angle_difference': -1}\",\r\n            \"thoughts\": \"왼쪽 팔, 왼쪽 팔꿈치, 오른쪽 팔꿈치에 대해서 임계치 이상의 difference가 확인되니까 이 부분에 대해 피드백을 주어야겠다.\"\r\n            \"outputs\": \"네, 동작 차이를 기반으로 피드백 드릴게요! 왼쪽 팔을 왼쪽으로 조금 더 돌려 주시고, 왼쪽 팔꿈치와 오른쪽 팔꿈치 모두 조금 더 펴 주세요. 벌써 완벽에 가까워지고 있네요! 우리 같이 계속 노력해 봐요!\"\r\n        }\r\n    }\r\n}"},
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