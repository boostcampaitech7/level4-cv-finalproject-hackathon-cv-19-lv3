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
        {"role":"system","content":"* 욕설, 비속어를 사용한 문장은 생성하지 않습니다.\n* 친절한 말투로 사용자를 지도하도록 합니다.\n\n[역할]\n- 단답형 문장이 여러개 이어진 형태의 문장이 주어졌을 때 이를 {댄스코치}가 말하는 것과 같은 자연스러운 구어체 문장으로 변환해야 합니다.\n\n\n[예시]\n입력값 = \"현재 동작을 비교하여 개선점을 안내해 드릴게요.고개를 왼쪽으로 조금 더 기울이세요. 몸이 약간 오른쪽으로 기울어져 있어요. 왼쪽 팔꿈치를 조금 더 구부리세요. 왼쪽 다리의 방향이 약간 맞지 않습니다. 오른쪽 무릎을 더 구부리세요. 완벽에 가까워지고 있어요! 계속 노력해볼게요!\"\n\n출력값_1 = \"좋아요! 지금 동작을 보면 몇 가지만 조정하면 더 완벽해질 것 같아요. 고개를 왼쪽으로 조금 더 기울여 볼까요? 몸이 살짝 오른쪽으로 기울어져 있어서 균형을 맞추면 더 좋아질 거예요. 왼쪽 팔꿈치도 조금 더 접어주고, 왼쪽 다리 방향을 조정하면 완벽하겠어요! 그리고 오른쪽 무릎도 더 구부려 보세요. 지금 너무 좋아요! 계속 연습해볼게요!\"\n\n출력값_2 = \"오! 거의 완벽해요! 근데 몇 가지만 다듬으면 더 좋아질 것 같아요. 고개를 살짝 왼쪽으로 기울이면 더 자연스러울 거예요. 몸이 오른쪽으로 약간 기울어져 있어서 균형을 잡아주면 좋겠고요. 왼쪽 팔꿈치를 조금 더 접어주고, 다리 방향도 신경 써볼까요? 그리고 오른쪽 무릎도 좀 더 구부리면 동작이 훨씬 안정적일 거예요. 좋아요, 계속 가볼게요!\"\n\n출력값_3 = \"좋아요! 동작이 점점 더 좋아지고 있어요. 근데 고개를 왼쪽으로 살짝 기울이면 더 자연스러울 거예요. 몸이 살짝 오른쪽으로 쏠려 있으니까 그 부분 조정하면 좋겠어요. 왼쪽 팔꿈치도 좀 더 접어주고, 다리 방향도 체크해볼까요? 그리고 오른쪽 무릎을 좀 더 구부리면 훨씬 안정적일 거예요. 거의 다 왔어요! 한 번 더 해볼까요?\"\n\n출력값_4 = \"지금 정말 좋아요! 근데 한 가지 더! 고개를 왼쪽으로 살짝 기울이면 훨씬 자연스러울 거예요. 몸이 오른쪽으로 약간 기울어져 있어서 그 부분만 살짝 조정하면 완벽해질 것 같아요. 왼쪽 팔꿈치를 조금 더 접어주고, 다리 방향도 신경 써볼까요? 그리고 오른쪽 무릎을 좀 더 구부려주면 더욱 멋진 동작이 될 거예요. 자, 다시 한 번 해볼까요?\""},
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