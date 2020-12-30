# Simple Inference Server

## Introduction

Implementation of a simple multi-thread TCP/IP server for machine learning model inference. Specifically, Question and Answering (QA) service was implemented as an example. The server is designed to have a thread-safe queue where all the inference requests were hold and multiple inference engine worker threads will get the inference requests and process concurrently.


## Usages

### Build Docker Image

```
$ docker build -f docker/server_amd64.Dockerfile --no-cache --tag=qa-server:0.0.1 .
```

### Run Docker Container

To run Docker container for a server, we have to use GPU for inference.

```
$ docker run -it --rm --gpus device=0 --network=host -v $(pwd):/mnt qa-server:0.0.1
```

To run Docker container for a client, we don't need GPU at all.

```
$ docker run -it --rm --network=host -v $(pwd):/mnt qa-server:0.0.1
```

### Prepare Models

Before running a inference server, we have to prepare the machine learning models accordingly.

```
$ prepare_model.py
```

A PyTorch BERT-QA model and its ONNX conversion will be saved to a `saved_models` directory.

### Run Server and Client

#### Local Host

To run a server on one local PC.


```
$ python server.py --host localhost
```

To run a client on the same local PC.


```
$ python client.py --host localhost
```

#### Host Service on Server

To run a server on one PC.


```
$ python server.py --host 0.0.0.0
```

To run a client connecting to the server from another one local PC.


```
$ python client.py --host <server-IP>
```

#### QA Service Client Demo

To run QA service from the client, we need to input a question and a piece of text where the answer to the question lies. The answer analyzed by the QA server will be sent back to the client once the inference request is processed.

```
$ python client.py --host localhost
==================================================
QA Client
==================================================
Please Input Text: 
According to PolitiFact the top 400 richest Americans "have more wealth than half of all Americans combined." According to the New York Times on July 22, 2014, the "richest 1 percent in the United States now own more wealth than the bottom 90 percent". Inherited wealth may help explain why many Americans who have become rich may have had a "substantial head start". In September 2012, according to the Institute for Policy Studies, "over 60 percent" of the Forbes richest 400 Americans "grew up in substantial privilege".
--------------------------------------------------
Please Input Question: 
What publication printed that the wealthiest 1% have more money than those in the bottom 90%?
--------------------------------------------------
Answer: 
New York Times
(Latency: 28.34296226501465 ms)
--------------------------------------------------
```

#### Other Server Choices

Both ONNX and PyTorch inference engines were implemented as the backends for the QA service. The number of inference sessions is also configurable.

```
$ python server.py --help
usage: server.py [-h] [--host HOST] [--port PORT]
                 [--num_inference_sessions NUM_INFERENCE_SESSIONS]
                 [--inference_engine_type {onnx,pytorch}]

Question and answer server.

optional arguments:
  -h, --help            show this help message and exit
  --host HOST           Default host IP. (default: 0.0.0.0)
  --port PORT           Default port ID. (default: 9999)
  --num_inference_sessions NUM_INFERENCE_SESSIONS
                        Number of inference sessions. (default: 2)
  --inference_engine_type {onnx,pytorch}
                        Inference engine type. (default: onnx)
```

### Stress Test

#### Multi-process Client Simulation

The server could be stress tested in small scale on a PC with a multi-core CPU.

```
$ python stress_test.py --host localhost
100%|█████████████████████████████████████████| 100/100 [00:05<00:00, 19.86it/s]
Client Latencies Measured: 
Mean: 25.058329820632935 ms
Std: 1.3205614012742213 ms
```

The test parameters, such as the number of simultaneous clients, are also configurable.

```
$ python stress_test.py --help
usage: stress_test.py [-h] [--host HOST] [--port PORT]
                      [--num_simultaneous_clients NUM_SIMULTANEOUS_CLIENTS]
                      [--num_total_clients NUM_TOTAL_CLIENTS]
                      [--num_request_per_client NUM_REQUEST_PER_CLIENT]

Question and answer server.

optional arguments:
  -h, --help            show this help message and exit
  --host HOST           Default host IP. (default: localhost)
  --port PORT           Default port ID. (default: 9999)
  --num_simultaneous_clients NUM_SIMULTANEOUS_CLIENTS
                        Number of simultaneous clients. (default: 5)
  --num_total_clients NUM_TOTAL_CLIENTS
                        Number of total clients. (default: 100)
  --num_request_per_client NUM_REQUEST_PER_CLIENT
                        Number of request per client. (default: 10)
```

#### AMD64 Platform

ONNX Runtime CUDA inference session with one `intra_op_num_threads`. The ONNX inference session does not run entirely on GPU as some ONNX operators used for the QA model were not supported on GPU and fall back to CPU. ONNX Runtime CPU inference session was not used as it was ~10x slower than ONNX Runtime CUDA inference session. PyTorch CUDA inference was not used as it was ~3x slower than ONNX Runtime CUDA inference session. The amd64 platform is Intel i9-9900K + NVIDIA RTX 2080 TI. Latencies were measured from the clients and the unit of the latency is millisecond.

| Number of Inference Sessions |  1 Client  |  5 Clients |  20 Clients  |  50 Clients  |
|:----------------------------:|:----------:|:----------:|:------------:|:------------:|
|               1              | 26.25±1.41 | 39.21±1.43 | 135.29±35.00 | 180.59±76.91 |
|               2              | 19.32±2.56 | 26.75±1.11 |  94.60±16.62 | 201.82±89.11 |
|               4              | 20.54±2.35 | 30.82±3.44 |  96.10±15.83 | 176.96±66.71 |
|               8              | 20.40±3.03 | 56.61±8.79 |  97.89±19.97 | 204.38±62.40 |

#### ARM64 Platform

ONNX Runtime CPU inference session with one `intra_op_num_threads` was used for the stress test in this case, since there is no ONNX Runtime GPU version directly available via `pip`. The inference latency was ~100x (5W mode) slower than the inference latency from the CUDA inference session on the amd64 platform above. Increasing `intra_op_num_threads` might increase the performance of inference. The arm64 platform is NVIDIA Jetson-Nano. Latencies were measured from the clients.


## TODO

- [ ] Investigate whether Python GIL is a problem for high-load concurrency.
- [ ] Try process-safe queue and multiple inference engine worker processes to see whether the high-load concurrency could be further improved.
- [ ] Implement TensorRT inference engine backend.

## References

* [BERT-QA Inference](https://leimao.github.io/blog/PyTorch-Dynamic-Quantization/)
* [BERT ONNX Conversion](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/transformers/notebooks/PyTorch_Bert-Squad_OnnxRuntime_GPU.ipynb)
* [Python Socket Server](https://docs.python.org/3.8/library/socketserver.html)
* [Python Server with Queue](https://stackoverflow.com/questions/46138771/python-multipleclient-server-with-queues)
