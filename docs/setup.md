# Local Development Setup Guide

This document provides instructions for getting the project set up for local
development.

## Prerequisites

This project requires the following system dependencies:

- Git
- [Protocol Buffer Compiler](https://grpc.io/docs/protoc-installation/)
  (`v 3.21.12`)
- [FFmpeg](https://ffmpeg.org/) (`v 7.1.1`)
- Python (`>=3.10.12 and <3.12`)

## Repository Setup

To download the codebase, you can clone the repository (for example using an SSH
approach, however an HTTPS approach should be fine as well):

```sh
git clone git@github.com:google/sbsim.git
```

After downloading the repository, navigate to the root directory from the
command line before continuing:

```
cd sbsim/
```

## System-specific Setup

By default, we use Linux OS for development. However it is also possible to
develop on Mac OS. We are also providing a "Dockerfile" to facilitate
development on non-Linux systems (Mac or Windows). Windows users can
alternatively use
[Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/install).

Consult one of the following system-specific setup guides, based on your
operating system and preferred tools:

- [Linux OS Setup](./setup/linux.md)
- [Mac OS Setup](./setup/mac.md)
- [Docker Setup](./setup/docker.md)

After completing the setup, you should be able to run notebooks or scripts as
desired.
