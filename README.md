# Simple Inference Server

## Introduction

Implementation of a simple multi-thread TCP/IP server for machine learning model inference. Specifically, Question and Answering (QA) service was implemented as an example. The server is designed to have a thread-safe queue where all the inference requests were hold and multiple inference engine threads will process the inference requests concurrently.


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

#### One PC

To run a server.


```
$ python server.py --host localhost
```

To run a client.


```
$ python client.py --host localhost
```

#### Multiple PC

To run a server on one PC.


```
$ python server.py --host 0.0.0.0
```

To run a client on another PC.


```
$ python client.py --host <server-IP>
```

### Run QA-Service

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

### Stress Test

#### AMD64 Platform

ONNX Runtime CUDA inference session with one `intra_op_num_threads`. The ONNX inference session does not run entirely on GPU as some ONNX operators were not supported on GPU and fall back to CPU. ONNX Runtime CPU inference session was not used as it was ~10x slower than CUDA inference session. The amd64 platform is Intel i9-9900K + NVIDIA RTX 2080 TI. Latency measured from the clients.

| Number of Inference Sessions |  1 Client  |  5 Clients |  20 Clients  |  50 Clients  |
|:----------------------------:|:----------:|:----------:|:------------:|:------------:|
|               1              | 26.25±1.41 | 39.21±1.43 | 135.29±35.00 | 180.59±76.91 |
|               2              | 19.32±2.56 | 26.75±1.11 |  94.60±16.62 | 201.82±89.11 |
|               4              | 20.54±2.35 | 30.82±3.44 |  96.10±15.83 | 176.96±66.71 |
|               8              | 20.40±3.03 | 56.61±8.79 |  97.89±19.97 | 204.38±62.40 |

#### ARM64 Platform

ONNX Runtime CPU inference session was used for the stress test in this case, since there is no ONNX Runtime GPU version directly available via `pip`. The inference latency was ~30x (5W mode) slower than the inference latency from the CUDA inference session on the amd64 platform above. The arm64 platform is Jetson-Nano. Latency measured from the clients.
