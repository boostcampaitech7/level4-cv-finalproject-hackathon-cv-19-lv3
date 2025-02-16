import sys
sys.path.append("./")
import requests
import json
from config import API_PATH, CLOVA_HOST, SYSTEM_PROMPT

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

        with requests.post(self._host, headers=headers, json=completion_request, stream=True) as r:
            check = False
            for line in r.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if check:
                        return line[5:]

                    if line.startswith(self.res_string):
                        check = True

with open(API_PATH, 'r') as f:
    api_info = json.load(f)
completion_executor = CompletionExecutor(
    host=CLOVA_HOST,
    api_key=api_info['service_api_key'],
    request_id=''
)


def base_feedback_model(content):
    '''
    단순히 서비스앱을 테스트하고 챗봇의 응답을 print하는 함수입니다.
    inputs:
        content : user가 input으로 넣을 문장
        request_id : 테스트앱 발급 시 나오는 request id 적으면 됩니다
        api_path : api key와 SubAccount 정보를 담은 파일 경로
    
    Returns:
        None
    '''
    # chatbot prompt
    preset_text = [{"role":"system","content":SYSTEM_PROMPT},
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