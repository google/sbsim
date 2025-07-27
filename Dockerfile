# Use an x86_64 specific base image and pin the version
FROM --platform=linux/amd64 ubuntu:20.04

# Set up environment
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    POETRY_HOME="/opt/poetry" \
    VENV_HOME="/opt/venv"
ENV PATH="$POETRY_HOME/bin:$VENV_HOME/bin:$PATH"

# Set shell to fail on errors in pipelines
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install system dependencies
# Combining into a single RUN layer and cleaning up reduces image size
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    gnupg && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-distutils \
    protobuf-compiler \
    ffmpeg \
    git \
    curl \
    unzip \
    build-essential \
    libffi-dev \
    libssl-dev \
    pkg-config \
    libhdf5-dev \
    openjdk-11-jdk && \
    rm -rf /var/lib/apt/lists/*

# Install Poetry and create virtual environment
RUN python3.10 -m venv $VENV_HOME && \
    curl -sSL https://install.python-poetry.org | python3.10 - && \
    poetry config virtualenvs.in-project true

# Install Bazel (required for some dependencies)
# Use curl instead of wget for consistency
WORKDIR /tmp
RUN curl -L -o bazel-installer.sh https://github.com/bazelbuild/bazel/releases/download/5.1.1/bazel-5.1.1-installer-linux-x86_64.sh && \
    chmod +x bazel-installer.sh && \
    ./bazel-installer.sh && \
    rm bazel-installer.sh

# Set main working directory
WORKDIR /workspace

# Copy only the necessary files for dependency installation
COPY pyproject.toml poetry.lock ./

# Install python dependencies using Poetry
# This replaces the old pip install and poetry install steps
RUN poetry lock && poetry install --no-root --with dev,notebooks

# Copy the rest of the application code
COPY . .

# Build .proto files
RUN poetry run python -m grpc_tools.protoc --proto_path=/workspace --python_out=/workspace smart_control/proto/*.proto

# Expose port and define default command
EXPOSE 8888
CMD ["poetry", "run", "jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser", "--NotebookApp.token=''"]
