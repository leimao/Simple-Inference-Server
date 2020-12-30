# Modified from
# https://docs.python.org/3.8/library/socketserver.html#asynchronous-mixins

import socket
import json
import time
import argparse


def client_session(host, port) -> None:

    # Create a socket (SOCK_STREAM means a TCP socket)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # Connect to server and send data
        sock.connect((host, port))

        print("=" * 50)
        print("QA Client")
        print("=" * 50)

        while True:
            text = input("Please Input Text: \n")
            print("-" * 50)
            question = input("Please Input Question: \n")
            print("-" * 50)
            data_dict = {"text": text, "question": question}
            data = json.dumps(data_dict)
            sock.sendall(bytes(data, "utf-8"))

            # Receive data from the server and shut down
            start_time = time.time()
            received = str(sock.recv(1024), "utf-8")
            end_time = time.time()
            latency = (end_time - start_time) * 1000
            print("Answer: ")
            print(received)
            print("(Latency: {} ms)".format(latency))
            print("-" * 50)


def main() -> None:

    host_default = "localhost"
    port_default = 9999

    parser = argparse.ArgumentParser(
        description="Question and answer server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--host",
                        type=str,
                        help="Default host IP.",
                        default=host_default)
    parser.add_argument("--port",
                        type=int,
                        help="Default port ID.",
                        default=port_default)

    argv = parser.parse_args()

    host = argv.host
    port = argv.port

    client_session(host=host, port=port)


if __name__ == "__main__":

    main()
