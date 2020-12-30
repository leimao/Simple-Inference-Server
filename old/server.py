# Modified from
# https://docs.python.org/3.8/library/socketserver.html#asynchronous-mixins
# https://stackoverflow.com/questions/46138771/python-multipleclient-server-with-queues

import socket
import threading
import socketserver
import queue
import json
import argparse
from typing import Tuple
from qa import QaTorchInferenceSession, QaOnnxInferenceSession


class InferenceExecutionThread(threading.Thread):
    """
    Inference execution thread.
    """

    def __init__(self, model_filepath: str, tokenizer_filepath: str, inference_engine_type: str = "onnx") -> None:

        super(InferenceExecutionThread, self).__init__()
        # Python queue library is thread-safe.
        # https://docs.python.org/3.8/library/queue.html#module-Queue
        # We can put tasks into queue from multiple threads safely.
        self.model_filepath = model_filepath
        self.tokenizer_filepath = tokenizer_filepath
        self.request_content_queue = queue.Queue()
        self.inference_engine_type = inference_engine_type
        if self.inference_engine_type == "onnx":
            self.inference_session = QaOnnxInferenceSession(model_filepath=self.model_filepath, tokenizer_filepath=self.tokenizer_filepath)
        elif self.inference_engine_type == "pytorch":
            self.inference_session = QaTorchInferenceSession(model_filepath=self.model_filepath, tokenizer_filepath=self.tokenizer_filepath)
        else:
            raise RuntimeError("Unsupported inference engine type.")
        #self.cur_thread_name = threading.current_thread().name
    
    def put_task(self, task: Tuple[socketserver.BaseRequestHandler, str]) -> None:
        """
        Put task into the queue.

        Args:
            task (Tuple[socketserver.BaseRequestHandler, str]): task content.
        """

        self.request_content_queue.put(task)

    def get_qsize(self) -> int:
        """
        Return the approximate size of the queue.

        Returns:
            int: The approximate size of the queue.
        """

        return self.request_content_queue.qsize()

    def run(self) -> None:
        """
        Run inference for the tasks in the queue.
        """

        while True:
            #print(threading.current_thread().name)
            if not self.request_content_queue.empty():
                print("Current Thread: {}, Number of Active Threads: {}".format(threading.current_thread().name, threading.active_count()))
                handler, data_dict = self.request_content_queue.get()
                
                question = data_dict["question"]
                text = data_dict["text"]
                answer = self.inference_session.run(question=question, text=text)
                if answer in ["", "[CLS]"]:
                    answer = "Unknown"
                # answer = "hello underwold"
                # response_dict = {"answer": answer}
                # response = json.dumps(response_dict)
                response = bytes(answer, "utf-8")
                print("Sending answer \"{}\" ...".format(answer))
                handler.request.sendall(response)
                self.request_content_queue.task_done()
                print("Inference Done.")


class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):
    """
    TCP request handler.
    """

    def find_shortest_queue(self) -> int:
        """
        Find approximately the shortest queue.

        Returns:
            int: The index of the execution thread that has approximately the shortest queue.
        """
        qsizes = [execution_thread.get_qsize() for execution_thread in execution_threads]
        min_qsize = float("inf")
        min_idx = 0
        for i in range(len(qsizes)):
            if qsizes[i] < min_qsize:
                min_qsize = qsizes[i]
                min_idx = i
        return min_idx

    def handle(self) -> None:
        """
        Handle method to override.
        """

        while True:
            print(self.client_address[0])
            data = str(self.request.recv(1024), "utf-8")
            if not data:
                print("User disconnected.")
                break
            data_dict = json.loads(data)
            print("{} wrote:".format(self.client_address[0]))
            print(data)
            # Find approximately the shortest queue
            queue_idx = self.find_shortest_queue()
            # queue_idx = 0
            # Put the task into the shortest queue
            execution_threads[queue_idx].put_task(task=(self, data_dict))
            print("Task sent to queue.")


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """
    Mutlithread TCP server.
    """

    pass


def main():

    host_default = "0.0.0.0"
    port_default = 9999
    num_inference_sessions_default = 2
    inference_engine_type_default = "onnx"

    parser = argparse.ArgumentParser(description="Question and answer server.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--host", type=str, help="Default host IP.", default=host_default)
    parser.add_argument("--port", type=int, help="Default port ID.", default=port_default)
    parser.add_argument("--num_inference_sessions", type=int, help="Number of inference sessions.", default=num_inference_sessions_default)
    parser.add_argument("--inference_engine_type", type=str, choices=["onnx", "pytorch"], help="Inference engine type.", default=inference_engine_type_default)
    
    argv = parser.parse_args()

    host = argv.host
    port = argv.port
    num_inference_sessions = argv.num_inference_sessions
    inference_engine_type = argv.inference_engine_type

    onnx_model_filepath = "./saved_models/bert-base-cased-squad2_model.onnx"
    torch_model_filepath = "./saved_models/bert-base-cased-squad2_model.pt"
    tokenizer_filepath = "./saved_models/bert-base-cased-squad2_tokenizer.pt"

    # Number of inference sessions.
    # Each inference session gets executed in an independent execution thread.
    global execution_threads
    if inference_engine_type == "onnx":
        model_filepath = onnx_model_filepath
    elif inference_engine_type == "pytorch":
        model_filepath = torch_model_filepath
    else:
        raise RuntimeError("Unsupported inference engine type.")
    
    print("Starting {} engine x {} ...".format(inference_engine_type, num_inference_sessions))
    execution_threads = [InferenceExecutionThread(model_filepath=model_filepath, tokenizer_filepath=tokenizer_filepath, inference_engine_type=inference_engine_type) for _ in range(num_inference_sessions)]

    for execution_thread in execution_threads:
        execution_thread.start()

    print("Starting QA Server ...")
    # Create the server, binding to localhost on port
    with ThreadedTCPServer((host, port), ThreadedTCPRequestHandler) as server:
        # Activate the server; this will keep running until you
        # interrupt the program with Ctrl-C
        print("=" * 50)
        print("QA Server")
        print("=" * 50)
        server.serve_forever()
    
    for execution_thread in execution_threads:
        execution_thread.join()

if __name__ == "__main__":

    main()