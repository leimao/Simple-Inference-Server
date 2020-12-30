# Inference Server Prototype

## Usages

### Build Docker Image

```
docker build -f docker/server_amd64.Dockerfile --no-cache --tag=qa-server:0.0.1 .
```

### Run Docker Container

#### One PC

To run a server, please run the following command.

```
docker run -it --rm --gpus device=0 --network=host -v $(pwd):/mnt qa-server:0.0.1
```

```
python server.py --host localhost
```

To run a client, please run the following command.


```
docker run -it --rm --network=host -v $(pwd):/mnt qa-server:0.0.1
```

```
python client.py --host localhost
```

#### Multiple PC

To run a server on one PC, please run the following command.

```
docker run -it --rm --gpus device=0 --network=host -v $(pwd):/mnt qa-server:0.0.1
```

```
python server.py --host 0.0.0.0
```

To run a client on another PC, please run the following command.


```
docker run -it --rm --network=host -v $(pwd):/mnt qa-server:0.0.1
```

```
python client.py --host <server-IP>
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

ONNX Runtime CPU inference session was used for the stress test in this case, since there is no ONNX Runtime GPU version directly available via `pip`. The inference latency was ~30x slower than the inference latency from the CUDA inference session on the amd64 platform above. The arm64 platform is Jetson-Nano. Latency measured from the clients.
