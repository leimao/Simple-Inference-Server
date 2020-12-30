# Modified from
# https://docs.python.org/3.8/library/socketserver.html#asynchronous-mixins
import socket
import sys
import json
import time


HOST, PORT = "localhost", 9999
# data = " ".join(sys.argv[1:])

# Create a socket (SOCK_STREAM means a TCP socket)
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    # Connect to server and send data
    sock.connect((HOST, PORT))

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

        # print("Sent:     {}".format(data))
        # print("Received: {}".format(received))
        print("-" * 50)