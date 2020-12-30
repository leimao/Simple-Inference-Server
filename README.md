# Inference Server Prototype
https://github.com/pricheal/python-client-server

```
docker build -f docker/server.Dockerfile --no-cache --tag=qa-server:0.0.1 .
```

Do not use multiple queues. It will slow down Python application significantly.