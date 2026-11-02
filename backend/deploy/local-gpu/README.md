# Local GPU and generic Docker hosts

Status: written for portability; not exercised during development. Docker with the NVIDIA
Container Toolkit and a Linux amd64 GPU host is required. The 6 GB local GPU may run fakes but
does not establish support for every real model.

## Local Compose profile

From the repository root, use `uv run main.py up-local-gpu`. Set `GPU_TARGET` to `local_gpu`
and `LOCAL_GPU_EXTRAS=gpu` for real Algorithms. The compute service keeps its Target Workspaces in `/data/target` on the
Scene data volume; like any Target, it is reached over HTTP and the worker downloads only the results. Model weights bind-mount the host cache.
Check `/ping` from inside the Compose network and inspect `/info`; no public port is required.

## Publish the portable image

Set `COMPUTE_IMAGE_REPO` to your repository (tags become `fake-<git short sha>` or `gpu-<git short sha>`) or pass `--image`, and authenticate Docker to that registry. From the repository root,
run `uv run main.py compute-image` then `uv run main.py push-compute-image`.
The build uses repository-root context and `--platform linux/amd64`. `--extras gpu`
builds the image with every real GPU Algorithm; the default image runs contract fakes.
Registry credentials stay with Docker, outside the image.

## Generic host

Create a private environment file on the remote host containing `COMPUTE_TOKEN` and, if needed,
`HF_TOKEN`. Run this on the Docker host, substituting the image tag:

```sh
docker run -d --name scene-compute --gpus all --restart unless-stopped \
  --env-file compute.env -e PORT=9000 -e DATA_DIR=/data -e MODELS_DIR=/models \
  -e HF_HOME=/models/hf -e TORCH_HOME=/models/torch \
  -v scene-compute-data:/data -v scene-models:/models \
  -p 127.0.0.1:9000:9000 ghcr.io/OWNER/scene-compute:TAG
```

Expose it through an authenticated TLS reverse proxy or a private tunnel. For public reachability,
`COMPUTE_TOKEN` is mandatory; configure the proxy to preserve `X-Compute-Token` and allow bodies
of at least 25 MB. Set `GENERIC_COMPUTE_URL` and the matching token locally. The `generic`
entry in `config/compute.yaml` uses HTTP with non-shared storage; set the pipeline's Target variables to `generic`.
The API and worker remain local. Check `/ping`, authenticated `/info`, then a fake Compute Run.

The model download command (`uv run main.py download-models`) populates the host model cache without
GPU inference. Full reset removes Scene/database state while retaining that cache. A remote cache
requires downloading on that host or copying the cache there; local download alone does not warm
the remote volume.
