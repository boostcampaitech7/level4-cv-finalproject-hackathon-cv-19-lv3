import requests
import json


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

        with requests.post(self._host + '/testapp/v1/chat-completions/HCX-003',
                           headers=headers, json=completion_request, stream=True) as r:
            check = False
            for line in r.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    print(line)
                    if check:
                        return line[5:]

                    if line.startswith(self.res_string):
                        check = True


def test_simple_chatbot(content, request_id, api_path='./CLOVA_API', is_test=True):
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

    # chatbot prompt
    preset_text = [
        {"role":"system","content":"[역할]\r\n피드백이 연속된 단문으로 주어졌을 때, 이걸 자연스러운 문장으로 연결해 줘.\r\n\r\n\r\n[예시 입력 1]\r\n왼쪽 팔을 반시계 방향으로 더 돌리세요. 왼쪽 팔꿈치를 구부리세요. 오른쪽 팔꿈치를 구부리세요. 오른쪽 다리를 반시계 방향으로 더 돌리세요. 왼쪽 무릎을 구부리세요. 좋은 흐름이에요! 이대로 쭉 가봅시다!\r\n\r\n[예시 출력 1]\r\n왼쪽 팔을 반시계 방향으로 천천히 더 돌려주시고, 이제 왼쪽 팔꿈치를 부드럽게 구부려 보세요. 오른쪽 팔꿈치도 함께 구부려주시고, 오른쪽 다리는 반시계 방향으로 조금 더 돌려봅시다. 왼쪽 무릎도 자연스럽게 구부려주세요. 아주 좋아요! 이 흐름을 유지하면서 계속 이어가봅시다. 잘하고 있어요!\n"},
        {"role":"user", "content":content}
    ]
    executor_setting = {
        'host': 'https://clovastudio.stream.ntruss.com',
        'api_key': api_info['test_api_key'] if is_test else api_info['service_api_key'],
        'request_id': request_id
    }

    completion_executor = CompletionExecutor(
        **executor_setting
    )

    request_data = {
        'messages': preset_text,
        'topP': 0.8,
        'topK': 0,
        'maxTokens': 256,
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


class CreateTaskExecutor:
    def __init__(self, host, uri, api_key, request_id):
        self._host = host
        self._uri = uri
        self._api_key = api_key
        self._request_id = request_id

    def _send_request(self, create_request):

        headers = {
            'Authorization': self._api_key,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id
        }
        result = requests.post(self._host + self._uri, json=create_request, headers=headers).json()
        return result

    def execute(self, create_request):
        res = self._send_request(create_request)
        if 'status' in res and res['status']['code'] == '20000':
            return res['result']
        else:
            return res

def train_request(
    name,
    trainingDatasetBucket = 'cv19-storage',
    trainingDatasetFilePath='test_dataset_1.csv',
    lr='1e-5f',
    epoch='8',
    api_path='./CLOVA_API',
    is_test = True
):
    '''
    Storage에 업로드 되어 있는 데이터셋을 기반으로 Tuning을 수행합니다.
    inputs:
        name : tuning을 어떤 이름으로 저장할지
        trainingDatasetBucket : 버킷 이름
        trainingDatasetFilePath : 버킷을 root로 했을 때 dataset의 경로
        lr : learning rate
        epoch : peft로 finetuning할 총 에폭 수
        api_path : api key와 SubAccount 정보를 담은 파일 경로

    returns:
        response_text : 모델 튜닝 요청에 대한 response text
    '''
    with open(api_path, 'r') as f:
        api_info = json.load(f)

    completion_executor = CreateTaskExecutor(
        host='https://clovastudio.stream.ntruss.com',
        uri='/tuning/v2/tasks',
        api_key=api_info['test_api_key'] if is_test else api_info['service_api_key'],
        request_id='<request_id>'
    )

    request_data = {'name': name,
                    'model': 'HCX-003',
                    'tuningType': 'PEFT',
                    'taskType': 'GENERATION',
                    'trainEpochs': epoch,
                    'learningRate': lr,
                    'trainingDatasetBucket': trainingDatasetBucket,
                    'trainingDatasetFilePath': trainingDatasetFilePath,
                    'trainingDatasetAccessKey': api_info['trainingDatasetAccessKey'],
                    'trainingDatasetSecretKey': api_info['trainingDatasetSecretKey']
                    }
    response_text = completion_executor.execute(request_data)
    print(request_data)
    print(response_text)

    return response_text

class FindTaskExecutor:
    def __init__(self, host, uri, api_key, request_id):
        self._host = host
        self._uri = uri
        self._api_key = api_key
        self._request_id = request_id

    def _send_request(self, task_id):
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'Authorization': self._api_key,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id
        }

        result = requests.get(self._host + self._uri + task_id, headers=headers).json()
        return result

    def execute(self, taskId):
        res = self._send_request(taskId)
        if 'status' in res and res['status']['code'] == '20000':
            return res['result']
        else:
            return res

def train_check(taskId, api_path='./CLOVA_API', is_test=True):
    '''
    TaskID를 기반으로 훈련 진행 상황을 체크하는 함수입니다.
    inputs:
        taskId : 확인할 tuning task의 taskId
        api_path : api key와 SubAccount 정보를 담은 파일 경로 
    
    returns:
        None
    '''
    with open(api_path, 'r') as f:
        api_info = json.load(f)

    completion_executor = FindTaskExecutor(
        host='https://clovastudio.stream.ntruss.com',
        uri='/tuning/v2/tasks/',
        api_key=api_info['test_api_key'] if is_test else api_info['service_api_key'],
        request_id='<request_id>',
        api_path='./CLOVA_API'
    )

    response_text = completion_executor.execute(taskId)
    print(taskId)
    print(response_text)

if __name__=="__main__":
    # train_check('5s8kvr2k')
    train_request("dance_model_real_coach_2", trainingDatasetFilePath='test_dataset_7_instruction.csv')