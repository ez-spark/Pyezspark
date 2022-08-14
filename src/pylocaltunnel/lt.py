import socket
import requests

from .utils.http_utils import parse_http_request, create_http_response


def create_socket(remote_url, remote_port):
    try:
        remote_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        remote_socket.connect_ex((remote_url, remote_port))
        return remote_socket
    except socket.error as e:
        print("Error: " + str(e))
        return None


def listen_for_data(remote_socket):
    try:
        data = remote_socket.recv(10000)
        return data
    except socket.error as e:
        print("Error: " + str(e))
        return None


# send data to the remote server
def send_data(remote_socket, data):
    try:
        remote_socket.send(data)
    except socket.error as e:
        print("Error: " + str(e))
        return None


# parse the data received from the remote server
def parse_data(data):
    if data:
        return parse_http_request(data.decode("utf-8"))
    else:
        return None


# send parsed data to localhost:8000 and receive response
def send_data_to_localhost(parsed_data, ip, port):
    if parsed_data:
        res = requests.request(parsed_data["method"], f"http://{ip}:{port}" +
                               parsed_data["path"], headers=parsed_data["headers"], data=parsed_data["body"], timeout=5)
        return res
    else:
        return None


# parse the response received from localhost:8000 and send it to the remote server
def parse_response(res):
    if res is not None:
        response = create_http_response(res.status_code, res.headers, res.text)
        return response
    else:
        return None

def get_localtunnel_url():
    res = requests.get("https://localtunnel.me?new", timeout=5)
    url = res.json()["url"]
    port = res.json()["port"]
    return url+':'+str(port)

def main(ip, port, url):
    print(url)
    
    remote_url_https = url.replace("https://", "")
    remote_url = remote_url_https[:remote_url_https.find(':')]
    remote_port = int(remote_url_https[remote_url_https.find(':')+1:])
    print(remote_url)
    print(remote_port)
    print(ip)
    print(port)
    print(f"https://{remote_url}")

    remote_socket = create_socket(remote_url, remote_port)
    remote_socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    remote_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 10)
    remote_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 1)

    while True:
        print("Waiting for data...")
        data = listen_for_data(remote_socket)

        # if data is b'' close the socket and create a new one
        if data == b'':
            remote_socket.close()
            remote_socket = create_socket(remote_url, remote_port)
            continue

        parsed_data = parse_data(data)
        res = send_data_to_localhost(parsed_data, ip, port)
        parsed_response = parse_response(res)
        send_data(remote_socket, parsed_response)
    
