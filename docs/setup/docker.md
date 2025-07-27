# Docker Setup Guide

To get the repository set up on **non-Linux** environments (e.g. macOS on Apple Silicon), use the pre-configured Docker environment (`linux/amd64`) defined in the `Dockerfile`.

## 1. Prerequisites

1. **Docker Desktop**: Download and install from [docker.com](https://www.docker.com/products/docker-desktop).
2. **Rosetta on Apple Silicon**: In Docker Desktop, enable **Use Rosetta for x86/amd64 images** under **Settings ▶ Experimental Features**.
3. **Verify** installation:

   ```bash
   docker --version
   docker run --platform linux/amd64 hello-world
   ```

If `hello-world` succeeds, you’re ready to proceed.

---

## 2. Build the Docker Image

From the project root (where `Dockerfile` lives), run:

```bash
# Build for x86_64 and tag it
docker build --platform linux/amd64 \
  -t sbsim:latest \
  .
```

Confirm the image exists:

```bash
docker images sbsim:latest
```

---

## 3. Run the Container in Detached Mode

We recommend running the container _detached_ so you can open shells, run tests, and launch Jupyter without tying up your terminal:

```bash
docker run -d \
  --name sbsim-container \
  -p 8888:8888 \
  -v "$(pwd)":/workspace \
  sbsim:latest
```

> **Note:** This mounts your local code into `/workspace` in the container, enabling live edits.

### 3.1 Access Jupyter

Open your browser at:

```
http://localhost:8888
```

Because we disable the token in our `CMD`, no password is needed. If you see a deprecation warning for `NotebookApp.token`, you can instead use:

```bash
jupyter notebook --no-browser --ServerApp.token=''
```

---

## 4. Exec into the Running Container

To run commands inside the live container:

```bash
# Open a shell in the container
docker exec -it sbsim-container bash

# Activate Poetry’s virtualenv
source /opt/venv/bin/activate

# Change into the workspace
cd /workspace
```

Then you can:

- **Run tests**: `poetry run pytest`
- **Execute scripts**: `python path/to/script.py`
- **Launch a notebook**: `jupyter notebook --no-browser --ServerApp.token=''`

---

## 5. Stop & Clean Up

```bash
# Stop the container
docker stop sbsim-container

# Remove the container
docker rm sbsim-container

# Remove the image
docker rmi sbsim:latest
```

---

## 6. Troubleshooting

- **Daemon not running**: If you see `Cannot connect to the Docker daemon`, open Docker Desktop or run:

  ```bash
  /Applications/Docker.app/Contents/Resources/bin/docker --version
  ```

- **Platform mismatch**: If you still get a warning about `linux/amd64` vs `arm64`, ensure Rosetta support is enabled in Docker Desktop.
- **Permission errors**: By default, files created inside the container are owned by `root`. To write files to your host, either adjust volume permissions or run with `--user=$(id -u):$(id -g)`.

---

_For ongoing improvements and discussion, see Issue [#80](https://github.com/google/sbsim/issues/80)._
