# Use Python 3.11 slim image for reliable cross-platform builds
FROM --platform=linux/amd64 python:3.11-slim

# Set up environment
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1
ENV PATH="$POETRY_HOME/bin:$PATH"

# Set shell to fail on errors in pipelines
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install system dependencies
# Combining into a single RUN layer and cleaning up reduces image size
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    protobuf-compiler \
    libprotobuf-dev \
    ffmpeg \
    git \
    curl \
    unzip \
    build-essential \
    libffi-dev \
    libssl-dev \
    pkg-config \
    libhdf5-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Set main working directory
WORKDIR /workspace

# Copy only the necessary files for dependency installation first (layer caching)
COPY pyproject.toml poetry.lock ./

# Install Python dependencies using Poetry (without root package)
RUN poetry install --no-root --with dev,notebooks

# Copy the rest of the application code
COPY . .

# Install the project and regenerate protobuf files
# Uses the apt-installed protoc (3.21.12) to match the version used on Linux
# and Mac setups, so the generated files match the ones checked into the repo.
RUN poetry install && \
    protoc \
        --proto_path=smart_control/proto \
        --python_out=smart_control/proto \
        smart_control/proto/smart_control_building.proto \
        smart_control/proto/smart_control_normalization.proto \
        smart_control/proto/smart_control_reward.proto

# Expose port for Jupyter
EXPOSE 8888

# Default command: launch Jupyter notebook
CMD ["poetry", "run", "jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser", "--ServerApp.token=''"]
