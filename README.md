# Inference Server Prototype

## Usages

### Build Docker Image

```
docker build -f docker/server.Dockerfile --no-cache --tag=qa-server:0.0.1 .
```

### Run Docker Container

Server:

```
docker run -it --rm --gpus device=0 --host=network -v $(pwd):/mnt qa-server:0.0.1
```

Client:


```
docker run -it --rm --host=network -v $(pwd):/mnt qa-server:0.0.1
```



Do not use multiple queues. It will slow down Python application significantly.