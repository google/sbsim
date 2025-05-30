# Use an x86_64 specific base image
FROM --platform=linux/amd64 ubuntu:20.04

# Set up environment
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3.10-distutils \
    protobuf-compiler ffmpeg git curl build-essential libffi-dev libssl-dev \
    pkg-config libhdf5-dev openjdk-11-jdk wget gnupg unzip && \
    rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.10 -m venv /opt/venv

# Install pip properly
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | /opt/venv/bin/python

# Upgrade pip and install essential packages
RUN /opt/venv/bin/pip install --upgrade pip setuptools wheel

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry /opt/venv/bin/python - && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry

# Add virtual environment and Poetry to PATH
ENV PATH="/opt/venv/bin:/opt/poetry/bin:$PATH"

# Install Bazel (required for some dependencies)
RUN wget https://github.com/bazelbuild/bazel/releases/download/5.1.1/bazel-5.1.1-installer-linux-x86_64.sh && \
    chmod +x bazel-5.1.1-installer-linux-x86_64.sh && \
    ./bazel-5.1.1-installer-linux-x86_64.sh && \
    rm bazel-5.1.1-installer-linux-x86_64.sh

# Clone the repository to a temporary location
RUN git clone https://github.com/google/sbsim.git /tmp/sbsim

# Configure poetry
RUN cd /tmp/sbsim && \
    poetry config virtualenvs.create false && \
    poetry config installer.max-workers 10

# Install base dependencies first
RUN /opt/venv/bin/pip install \
    jupyter \
    notebook \
    dm-reverb==0.14.0 \
    urllib3 \
    html5lib \
    requests

# Install dependencies from the temporary location
RUN cd /tmp/sbsim && \
    poetry lock --no-update && \
    poetry install --no-root

# Build .proto files
RUN cd /tmp/sbsim/smart_control/proto && \
    protoc --python_out=. smart_control_building.proto \
           smart_control_normalization.proto \
           smart_control_reward.proto

# Create a startup script that copies files on container start
RUN echo '#!/bin/bash\n\
source /opt/venv/bin/activate\n\
if [ ! -d "/workspace/sbsim/.git" ]; then\n\
    cp -r /tmp/sbsim/. /workspace/sbsim/\n\
fi\n\
cd /workspace/sbsim\n\
jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --NotebookApp.token="" --no-browser --notebook-dir=/workspace/sbsim' > /start.sh && \
    chmod +x /start.sh

# Set up environment variables for Jupyter Notebook
EXPOSE 8888

CMD ["/start.sh"]
