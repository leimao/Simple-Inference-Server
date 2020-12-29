# Modified from
# https://docs.python.org/3.8/library/socketserver.html#asynchronous-mixins

# https://stackoverflow.com/questions/46138771/python-multipleclient-server-with-queues

import socket
import threading
import socketserver
import queue

request_content_queue = queue.Queue()


class RequestQueueExecutionThread(threading.Thread):

    def __init__(self, request_content_queue):
        super().__init__()
        self.request_content_queue = request_content_queue

    def run(self):
        while True:
            if not self.request_content_queue.empty():
                request_content = self.request_content_queue.get()
                # logging.debug("Queueing data: " + command.data)
                # time.sleep(3)
                # logging.debug("Finshed queue: " + command.data)
                request_content.request.sendall(request_content.response)
                self.request_content_queue.task_done()


class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):

    def handle(self):
        while True:
            try:
                data = str(self.request.recv(1024), 'ascii')
                if not data:
                    print("Connection lost.")
                    break
                print("{} wrote:".format(self.client_address[0]))
                print(data)
                cur_thread = threading.current_thread()
                print("Number of active threads: {}".format(threading.active_count()))
                self.response = bytes("{}: {}".format(cur_thread.name, data), 'ascii')
                request_content_queue.put(self)
            except:
                print("Connection lost.")
                break

class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass

if __name__ == "__main__":
    # Port 0 means to select an arbitrary unused port
    HOST, PORT = "localhost", 9999

    c = RequestQueueExecutionThread(request_content_queue=request_content_queue)
    c.start()

    # Create the server, binding to localhost on port 9999
    with ThreadedTCPServer((HOST, PORT), ThreadedTCPRequestHandler) as server:
        # Activate the server; this will keep running until you
        # interrupt the program with Ctrl-C
        server.serve_forever()
    
    c.join()