import json
import requests
import pandas as pd
from tqdm import tqdm


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

        with requests.post(self._host + '/serviceapp/v1/chat-completions/HCX-003',
                           headers=headers, json=completion_request, stream=True) as r:
            check = False
            for line in r.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if check:
                        return line[5:]

                    if line.startswith(self.res_string):
                        check = True


if __name__ == '__main__':
    api_path = "./CLOVA_API"
    dataset_path = "./clova_datasets/output.csv"
    df = pd.read_csv(dataset_path)

    with open(api_path, 'r') as f:
        api_info = json.load(f)

    completion_executor = CompletionExecutor(
        host='https://clovastudio.stream.ntruss.com',
        api_key=api_info['service_api_key'],
        request_id='d6e3ba157a7b487485b83545dfbed72f'
    )

    preset_text = [
        {"role":"system","content":"* 욕설, 비속어를 사용한 문장은 생성하지 않습니다.\n* 친절한 말투로 사용자를 지도하도록 합니다.\n\n[역할]\n- 단답형 문장이 여러개 이어진 형태의 문장이 주어졌을 때 이를 댄스코치가 말하는 것과 같은 자연스러운 구어체 문장으로 변환해야 합니다.\n\n\n[예시 1] - 피드백을 주어야하는 경우\n[User] : \"정확한 동작 분석을 통해 개선할 점을 알려드리겠습니다.고개를 왼쪽으로 더 기울이세요. 오른쪽 팔꿈치를 조금 더 펴세요. 왼쪽 다리를 오른쪽으로 좀 더 돌려주세요. 오른쪽 다리를 왼쪽으로 좀 더 돌려주세요. 멋진 동작이에요! 앞으로도 함께 최선을 다해봐요!\"\n\n[Assistant]: \"정확한 동작 분석을 통해 개선할 점을 알려드릴게요! 고개를 조금 더 왼쪽으로 기울여 주세요. 오른쪽 팔꿈치는 살짝 더 펴 주시고, 왼다리는 오른쪽으로 조금 더 돌려 주세요. 오른다리는 반대로 왼쪽으로 조금 더 돌려 주세요. 멋진 동작이에요! 앞으로도 함께 최선을 다해봐요!\"\n\n\n[예시 2] - 유저가 정확한 동작을 한 경우\n[User]: \"춤이 너무 매력적이에요!\"\n[Assistant: \"와, 춤이 정말 매력적이에요! 너무 잘하고 있어요!\"\n\n\n[예시 3] - 피드백을 주어야하는 경우\n[User]: \"자세 차이를 바탕으로 더 나은 동작을 위한 피드백을 드리겠습니다.몸이 너무 오른쪽으로 기울어져 있으니 왼쪽 어깨를 조금 내리세요. 오른쪽 팔을 오른쪽으로 좀 더 돌려주세요. 왼쪽 팔꿈치를 조금 더 펴세요. 오른쪽 다리를 왼쪽으로 좀 더 돌려주세요. 왼쪽 무릎을 조금 더 구부리세요. 오른쪽 무릎을 조금 더 구부리세요. 좋은 흐름이에요! 이대로 쭉 가봅시다!\"\n\n[Assistant]: \"자세 차이를 바탕으로 더 나은 동작을 위한 피드백 드릴게요! 몸이 너무 오른쪽으로 기울어져 있으니 왼쪽 어깨를 살짝 내려주세요. 오른팔은 오른쪽으로 좀 더 돌려주시고, 왼쪽 팔꿈치는 조금 더 펴 주세요. 오른다리는 왼쪽으로 살짝 더 돌려주시고, 왼쪽 무릎과 오른쪽 무릎도 조금 더 구부려 주세요. 흐름 좋습니다! 이 느낌 그대로 쭉 가볼게요!\"\n\n\n"},
    ]
    for idx, row in tqdm(df.iterrows()):
        s = df['Completion'][idx]
        user_input = [
            {"role":"user","content":f"다음 문장을 댄스코치가 말하는 것과 같은 구어체 문장으로 바꿔 줘.\n{s}"}
        ]
        request_data = {
            'messages': preset_text + user_input,
            'topP': 0.8,
            'topK': 0,
            'maxTokens': 512,
            'temperature': 0.5,
            'repeatPenalty': 5.0,
            'stopBefore': [],
            'includeAiFilters': True,
            'seed': 0
        }
        result_dict = completion_executor.execute(request_data)
        
        if result_dict:
            result_dict = json.loads(result_dict)
            modified_s = result_dict['message']['content']
            df['Completion'][idx] = modified_s
            print(modified_s)
    
    df.to_csv(dataset_path, index=False, encoding="utf-8-sig")