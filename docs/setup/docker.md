# Docker Setup Guide

The repository's `Dockerfile` defines a pre-configured environment
(`linux/amd64`) for running the notebooks and scripts without installing the
dependencies on the host machine.

You can either run the image published to the GitHub Container Registry or build
the image locally from source.

## Run the Published Image

A container image is published to the GitHub Container Registry each time a
release is published. Pull the latest image:

```bash
docker pull --platform linux/amd64 ghcr.io/google/sbsim:latest
```

Run the container:

```bash
docker run -d \
  --name sbsim-container \
  --platform linux/amd64 \
  -p 8888:8888 \
  ghcr.io/google/sbsim:latest
```

The image starts a Jupyter notebook server on port 8888 with token
authentication disabled (as configured by the `Dockerfile`). Open your browser
at:

```
http://localhost:8888
```

Images are tagged with `latest` and the release tag. For semantic version tags
(for example `v1.2.3`), the `1.2.3` and `1.2` tags are also published. Pin a
specific tag instead of `latest` when you need a reproducible environment:

```bash
docker pull --platform linux/amd64 ghcr.io/google/sbsim:1.2.3
```

> **Note:** The GitHub Container Registry package must be public for anonymous
> pulls. If it is private, authenticate first with `docker login ghcr.io` using
> a GitHub personal access token that has the `read:packages` scope.

The `docker exec` and cleanup commands in the local build section below work the
same way for the published image.

## Build the Image Locally

To get the repository set up on **non-Linux** environments (e.g. macOS on Apple
Silicon), use the pre-configured Docker environment (`linux/amd64`) defined in
the `Dockerfile`.

### Prerequisites

1. **Docker Desktop**: Download and install from
    [docker.com](https://www.docker.com/products/docker-desktop).

2. **Rosetta on Apple Silicon**: In Docker Desktop, enable **Use Rosetta for
    x86/amd64 images** under **Settings ▶ Experimental Features**.

3. **Verify** installation:

    ```bash
    docker --version
    docker run --platform linux/amd64 hello-world
    ```

If `hello-world` succeeds, you're ready to proceed.

> **Apple Silicon Note:** Some TensorFlow-related dependencies may not work
> properly on `linux/arm64`. If Docker build fails, add `--platform=linux/amd64`
> to force x86_64 emulation. Note that emulation may be slow and TensorFlow
> workloads may crash or hang due to resource constraints. For heavy TF
> workloads, consider using a native Linux environment or cloud-based compute.

### Build the Docker Image

From the project root (where `Dockerfile` lives), run:

```bash
# Build the image (uses python:3.11-slim as base)
docker build -t sbsim:latest .
```

> On Apple Silicon, the Dockerfile already specifies `--platform=linux/amd64`.
> If you encounter issues, try explicitly passing the platform flag:
>
> ```bash
> docker build --platform linux/amd64 -t sbsim:latest .
> ```

Confirm the image exists:

```bash
docker images sbsim:latest
```

### Run the Container in Detached Mode

We recommend running the container _detached_ so you can open shells, run tests,
and launch Jupyter without tying up your terminal:

```bash
docker run -d \
  --name sbsim-container \
  -p 8888:8888 \
  -v "$(pwd)":/workspace \
  sbsim:latest
```

> **Note:** This mounts your local code into `/workspace` in the container,
> enabling live edits.

### Access Jupyter

Open your browser at:

```
http://localhost:8888
```

Because we disable the token in our `CMD`, no password is needed. If you see a
deprecation warning for `NotebookApp.token`, you can instead use:

```bash
jupyter notebook --no-browser --ServerApp.token=''
```

### Exec into the Running Container

To run commands inside the live container:

```bash
# Open a shell in the container
docker exec -it sbsim-container bash
```

Inside the container, you're already in `/workspace`. Use Poetry to run
commands:

```bash
# Run tests
poetry run pytest -q

# Run a Python script
poetry run python path/to/script.py

# Check Python version (should be 3.11.x)
poetry run python --version

# Launch Jupyter (if not already running via CMD)
poetry run jupyter notebook --ip=0.0.0.0 --no-browser --ServerApp.token=''
```

Alternatively, run commands directly without exec:

```bash
# Run tests from outside the container
docker exec sbsim-container poetry run pytest -q

# Check Python version
docker exec sbsim-container python --version
```

### Stop and Clean Up

```bash
# Stop the container
docker stop sbsim-container

# Remove the container
docker rm sbsim-container

# Remove the image
docker rmi sbsim:latest
```

### Troubleshooting

- **Daemon not running**: If you see `Cannot connect to the Docker daemon`, open
    Docker Desktop or run:

    ```bash
    /Applications/Docker.app/Contents/Resources/bin/docker --version
    ```

- **Platform mismatch**: If you still get a warning about `linux/amd64` vs
    `arm64`, ensure Rosetta support is enabled in Docker Desktop.

- **Permission errors**: By default, files created inside the container are
    owned by `root`. To write files to your host, either adjust volume
    permissions or run with `--user=$(id -u):$(id -g)`.

______________________________________________________________________

_For ongoing improvements and discussion, see Issue
[#80](https://github.com/google/sbsim/issues/80)._
