from unittest import TestCase

from utils import http_utils

class TestParser(TestCase):

    def test_parse_http_request(self):
        data = "GET / HTTP/1.1\r\nHost: localhost:5000\r\n\r\n"
        request = http_utils.parse_http_request(data)
        self.assertEqual(request["method"], "GET")
        self.assertEqual(request["path"], "/")
        self.assertEqual(request["headers"]["Host"], "localhost:5000")

    def test_parse_http_response(self):
        data = "HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n"
        response = http_utils.parse_http_response(data)
        self.assertEqual(response["status"], "200")
        self.assertEqual(response["headers"]["Content-Length"], "0")

    def test_create_http_response(self):
        headers = {"Content-Type": "text/html"}
        body = "<h1>Hello, World!</h1>"
        response = http_utils.create_http_response("200", headers, body)
        self.assertEqual(
            response, b"HTTP/1.1 200\r\nContent-Type: text/html\r\n\r\n<h1>Hello, World!</h1>")
