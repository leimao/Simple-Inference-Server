"""
Stress test the server in small scale using one single PC.
Make sure there are n clients connected to the server simultaneously, each client does m QA requests and disconnect.
"""

import socket
import json
import time
import argparse
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

def run_imap_unordered_multiprocessing(func, argument_list, num_processes):
    """
    https://leimao.github.io/blog/Python-tqdm-Multiprocessing/
    """

    pool = Pool(processes=num_processes)

    result_list_tqdm = []
    for result in tqdm(pool.imap_unordered(func=func, iterable=argument_list), total=len(argument_list)):
        result_list_tqdm.append(result)

    return result_list_tqdm

def run_apply_async_multiprocessing(func, argument_list, num_processes):

    pool = Pool(processes=num_processes)

    jobs = [pool.apply_async(func=func, args=(*argument,)) if isinstance(argument, tuple) else pool.apply_async(func=func, args=(argument,)) for argument in argument_list]
    pool.close()
    result_list_tqdm = []
    for job in tqdm(jobs):
        result_list_tqdm.append(job.get())

    return result_list_tqdm

def client_session(host, port, num_requests) -> float:

    latencies = []

    # Create a socket (SOCK_STREAM means a TCP socket)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # Connect to server and send data
        sock.connect((host, port))

        question = "What publication printed that the wealthiest 1% have more money than those in the bottom 90%?"
        text = "According to PolitiFact the top 400 richest Americans \"have more wealth than half of all Americans combined.\" According to the New York Times on July 22, 2014, the \"richest 1 percent in the United States now own more wealth than the bottom 90 percent\". Inherited wealth may help explain why many Americans who have become rich may have had a \"substantial head start\". In September 2012, according to the Institute for Policy Studies, \"over 60 percent\" of the Forbes richest 400 Americans \"grew up in substantial privilege\"."

        for _ in range(num_requests):

            data_dict = {"text": text, "question": question}
            data = json.dumps(data_dict)
            sock.sendall(bytes(data, "utf-8"))

            # Receive data from the server and shut down
            start_time = time.time()
            _ = str(sock.recv(1024), "utf-8")
            end_time = time.time()
            latency = (end_time - start_time) * 1000
            latencies.append(latency)

    latency_mean = np.mean(latencies)
    
    return latency_mean


def main() -> None:

    host_default = "localhost"
    port_default = 9999
    num_simultaneous_clients_default = 5
    num_total_clients_default = 100
    num_request_per_client_default = 10

    parser = argparse.ArgumentParser(description="Question and answer server.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--host", type=str, help="Default host IP.", default=host_default)
    parser.add_argument("--port", type=int, help="Default port ID.", default=port_default)
    parser.add_argument("--num_simultaneous_clients", type=int, help="Number of simultaneous clients.", default=num_simultaneous_clients_default)
    parser.add_argument("--num_total_clients", type=int, help="Number of total clients.", default=num_total_clients_default)
    parser.add_argument("--num_request_per_client", type=int, help="Number of request per client.", default=num_request_per_client_default)

    argv = parser.parse_args()

    host = argv.host
    port = argv.port
    num_simultaneous_clients = argv.num_simultaneous_clients
    num_total_clients = argv.num_total_clients
    num_request_per_client = argv.num_request_per_client

    num_processes = num_simultaneous_clients
    num_requests = num_request_per_client

    argument_list = [(host, port, num_requests) for _ in range(num_total_clients)]

    latencies = run_apply_async_multiprocessing(func=client_session, argument_list=argument_list, num_processes=num_processes)

    latency_mean = np.mean(latencies)
    latency_std = np.std(latencies)

    print("Client Latencies Measured: ")
    print("Mean: {} ms".format(latency_mean))
    print("Std: {} ms".format(latency_std))

if __name__ == "__main__":
    
    main()