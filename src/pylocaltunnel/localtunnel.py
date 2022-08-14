import logging
import argparse
import selectors
import socket
import types

import requests

import utils.http_utils as http_utils

from utils.logger import logger


def accept_wrapper(sock):
    conn, addr = sock.accept()  # Should be ready to read
    logger.info(f"Accepted connection from {addr}")
    conn.setblocking(False)
    data = types.SimpleNamespace(addr=addr, inb=b"", outb=b"")
    events = selectors.EVENT_READ | selectors.EVENT_WRITE
    sel.register(conn, events, data=data)


def service_connection(key, mask, port):
    sock = key.fileobj
    data = key.data

    if mask & selectors.EVENT_READ:
        recv_data = sock.recv(4096)  # Should be ready to read

        if recv_data:
            logger.debug(f"{data.addr} received {recv_data}")
            req = http_utils.parse_http_request(recv_data.decode("utf-8"))
            res = requests.request(req["method"], f"http://localhost:{port}" +
                                   req["path"], headers=req["headers"], data=req["body"], timeout=5)

            response_to_send = http_utils.create_http_response(
                res.status_code, res.headers, res.text)
            data.outb += response_to_send
        else:
            print(f"Closing connection to {data.addr}")
            sel.unregister(sock)
            sock.close()

    if mask & selectors.EVENT_WRITE:
        if data.outb:
            print(f"Echoing {data.outb!r} to {data.addr}")
            sent = sock.send(data.outb)  # Should be ready to write
            data.outb = data.outb[sent:]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start a localtunnel client")
    parser.add_argument("-p", "--port", type=int,
                        default=8000, help="Port to bind to")

    args = parser.parse_args()

    logger.info(f"Starting localtunnel client on port {args.port}")

    sel = selectors.DefaultSelector()

    port = args.port
    lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        lsock.bind(("127.0.0.1", port))
    except OSError:
        logger.error(f"Failed to bind to port {port}: {OSError.strerror}")
        print(f"Failed to bind to port {port}: {OSError.strerror}")
        exit(1)

    lsock.listen()
    logger.info(f"Listening on localhost:{port}")
    lsock.setblocking(False)
    sel.register(lsock, selectors.EVENT_READ, data=None)


    try:
        while True:
            events = sel.select(timeout=None)
            for key, mask in events:
                if key.data is None:
                    accept_wrapper(key.fileobj)
                else:
                    service_connection(key, mask)
    except KeyboardInterrupt:
        print("\nCaught keyboard interrupt, exiting")
        logger.info("Caught keyboard interrupt, exiting")
    finally:
        sel.close()
