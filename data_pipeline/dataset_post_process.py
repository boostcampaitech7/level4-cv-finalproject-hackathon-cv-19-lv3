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
        {"role":"system","content":"* 욕설, 비속어를 사용한 문장은 생성하지 않습니다.\n* 피드백에 있어서 균형감보다는 자세의 정확함을 강조하도록 합니다.\n\n[역할]\n- \"~요.\"로 끝나는 문장이 여러개 주어졌을 때, 각 문장을 자연스럽게 연결하여 구어체로 변환하세요.\n- 주어지는 각각의 문장은 댄스 동작에 대한 피드백입니다.\n\n\n"},
    ]
    for idx, row in tqdm(df.iterrows()):
        s = df['Completion'][idx]
        print("원본 문장 : ", s)

        user_input = [
            {"role":"user","content":f"{s}"}
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
            df.loc['Completion', idx] = modified_s
            print("수정된 문장: ", modified_s)
    
    df.to_csv(dataset_path, index=False, encoding="utf-8-sig")