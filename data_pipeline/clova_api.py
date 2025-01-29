import requests
import json


class CompletionExecutor:
    def __init__(self, host, api_key, request_id):
        self._host = host
        self._api_key = api_key
        self._request_id = request_id

    def execute(self, completion_request):
        headers = {
            'Authorization': self._api_key,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id,
            'Content-Type': 'application/json; charset=utf-8',
            'Accept': 'text/event-stream'
        }

        with requests.post(self._host + '/testapp/v1/chat-completions/HCX-003',
                           headers=headers, json=completion_request, stream=True) as r:
            for line in r.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    print(line)


def test_simple_chatbot(content, request_id, api_path='./CLOVA_API'):
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
        {"role":"system","content":"피드백이 key, value 쌍으로 주어졌을 때, 이거를 자연스러운 문장으로 연결해줘.\n\n\n[예시 입력 1]\nright_arm: Raise your right arm.\r\nleft_elbow: Bend your left elbow.\r\nright_elbow: Bend your right elbow.\r\n\n[예시 출력 1]\n자세 차이를 기반으로 피드백을 드리도록 하겠습니다.\n오른쪽 팔을 조금 더 들어주시고, 왼쪽 팔목은 좀 더 굽혀주세요. 오른쪽 팔목도 좀 더 굽혀주세요.\r\n나머지 자세는 모두 완벽합니다! 앞으로도 함께 노력해봐요!\n"},
        {"role":"user", "content":content}
    ]
    executor_setting = {
        'host': 'https://clovastudio.stream.ntruss.com',
        'api_key': api_info['api_key'],
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
    completion_executor.execute(request_data)


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
    api_path='./CLOVA_API'
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
        api_key=api_info['api_key'],
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

def train_check(taskId, api_path='./CLOVA_API'):
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
        api_key=api_info['api_key'],
        request_id='<request_id>',
        api_path='./CLOVA_API'
    )

    response_text = completion_executor.execute(taskId)
    print(taskId)
    print(response_text)

if __name__=="__main__":
    # train_check('5s8kvr2k')
    train_request("dance_model_test_1", trainingDatasetFilePath='test_dataset_noinstruction_3.csv')