def parse_http_header(data):
    headers = {}
    lines = data.split("\r\n")
    for line in lines:
        if line == "":
            break
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key] = value.strip()
    return headers


def parse_http_response(data):
    response = {}
    lines = data.split("\r\n")
    response["status"] = lines[0].split(" ")[1]
    response["headers"] = parse_http_header(data)
    return response


def parse_http_request(data):
    request = {}
    head_string = data[:data.find('{')]
    body_string = data[data.find('{'):]
    print('HEAD:\n'+head_string)
    head_string2 = head_string[:]
    head_string = head_string.split("\r\n")
    
    
    print('BODY:\n'+body_string)
    request["method"] = head_string[0].split(" ")[0]
    request["path"] = head_string[0].split(" ")[1]
    request["headers"] = parse_http_header(head_string2)
    request["body"] = body_string
    return request


def create_http_response(status, headers, body):
    response = f"HTTP/1.1 {status}\r\n"
    for key, value in headers.items():
        response += f"{key}: {value}\r\n"
    response += "\r\n"
    response += body
    return response.encode("utf-8")
