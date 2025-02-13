import requests

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
