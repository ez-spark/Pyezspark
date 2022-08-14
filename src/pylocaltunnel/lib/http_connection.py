import requests
from utils import http_utils
from utils.logger import logger


class HTTPConnection:
    def __init__(self, url, port):
        self.port = port
        self.url = url

    def send_request(self, method, path, headers, body):
        try:
            response = requests.request(
                method, self.url + path, headers=headers, data=body, timeout=5)
            logger.info(
                f"Request sent to {self.url + path}: {response.status_code} {response.text}")
            return http_utils.create_http_response(response.status_code, response.headers, response.text)
        except requests.exceptions.RequestException as e:
            logger.critical(
                f"Error while sending request to {self.url + path}: {e}")
            res = http_utils.create_http_response(
                "500", {"Content-Type": "text/plain"}, "Internal Server Error")
            return res
