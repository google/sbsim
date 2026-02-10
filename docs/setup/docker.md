# Docker Setup Guide

To get the repository set up on non-Linux environments, you can use the
pre-configured Docker environment ("Linux/amd64") specified by the "Dockerfile".

## Installing Docker

First, install
[Docker Desktop](https://www.docker.com/products/docker-desktop/), and accept
the terms.

Open Docker Desktop, and wait until it is running before proceeding.

Verify the installation:

```sh
docker --version

docker run hello-world
```

### Troubleshooting Installation Issues on Mac

On Mac, if verification fails, try:

```sh
/Applications/Docker.app/Contents/Resources/bin/docker --version
```

If that works, as a one time setup step, update the ".zshrc" file to add the
installed location to the path:

```sh
# this is the "~/.zshrc" file...
export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"
```

Remember to restart your shell afterwards:

```sh
source ~/.zshrc
```

Now you should be able to verify the installation:

```sh
docker --version

docker run hello-world
```

## Image Operations

Ensure you have navigated to the root directory of the repository, where the
"Dockerfile" is located, before proceeding.

Build the image:

```bash
docker build -t sbsim-docker-env .
```

Listing images:

```sh
docker images
```

Removing the image, as necessary:

```sh
docker rmi sbsim-docker-env
```

## Container Operations

After the image is built, run the container (in interactive mode, with open
ports):

```bash
docker run -it -p 8888:8888 -v $(pwd):/workspace sbsim-docker-env
```

> NOTE: the container will copy the repository into "/workspace/sbsim" on the
> first run. Use -v to persist changes.

To access Jupyter notebooks, visit
[http://localhost:8888](http://localhost:8888) in the browser.

To run scripts or tests inside the actively running docker container:

```sh
# activate the virtual environment:
source /opt/venv/bin/activate

# navigate to the repository:
cd /workspace/sbsim

# running scripts:
python path/to/script.py

# running tests:
pytest
```

To stop the container:

```sh
docker stop sbsim-docker-env
```

Listing containers (to get their identifiers):

```sh
docker ps -a
```

Removing a container:

```sh
docker rm <container-id>
```

> NOTE: in the future we would like to further update these instructions and
> improve the Dockerfile. See
> [issue #80](https://github.com/google/sbsim/issues/80) (contributions
> welcome)!
