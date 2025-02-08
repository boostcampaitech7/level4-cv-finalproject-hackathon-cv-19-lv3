import requests
import json



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
    train_request("dance_model_real_coach_2", trainingDatasetFilePath='test_dataset_7_instruction.csv')