# Modified from
# https://docs.python.org/3.8/library/socketserver.html#asynchronous-mixins

import socket
import threading
import socketserver

class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):

    def handle(self):
        while True:
            try:
                data = str(self.request.recv(1024), 'utf-8')
                print("{} wrote:".format(self.client_address[0]))
                print(data)
                cur_thread = threading.current_thread()
                print("Number of active threads: {}".format(threading.active_count()))
                response = bytes("{}: {}".format(cur_thread.name, data), 'utf-8')
                self.request.sendall(response)
            except:
                print("Connection lost.")
                break

class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass

if __name__ == "__main__":
    # Port 0 means to select an arbitrary unused port
    # HOST, PORT = "localhost", 9999
    HOST, PORT = "0.0.0.0", 9999

    # Create the server, binding to localhost on port 9999
    with ThreadedTCPServer((HOST, PORT), ThreadedTCPRequestHandler) as server:
        # Activate the server; this will keep running until you
        # interrupt the program with Ctrl-C
        server.serve_forever()