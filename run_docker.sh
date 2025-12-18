#!/bin/bash

# Name of the Docker container
CONTAINER_NAME="vla"

# Docker image to use
DOCKER_IMAGE="docker-ubuntu20_ros:latest"

# Path to the workspace directory
# You should be in the grapheqa_ws directory
WORKSPACE_DIR="$(pwd)"
DATA_DIR="/media/SSD/"

# Environment variables
SSH_AUTH_SOCK_VAR=$SSH_AUTH_SOCK

WS="/ws/external"

# Run the Docker container with the appropriate arguments
docker run -it \
  --name $CONTAINER_NAME \
  --privileged \
  --net=host \
  --workdir ${WS} \
  --env="DISPLAY=$DISPLAY" \
  --env="XAUTHORITY:$XAUTHORITY" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e QT_X11_NO_MITSHM=1 \
  -v $SSH_AUTH_SOCK_VAR:/run/ssh-agent \
  -e SSH_AUTH_SOCK=/run/ssh-agent \
  -v $WORKSPACE_DIR:${WS}:cached \
  -v $DATA_DIR:${WS}/data \
  --runtime=nvidia \
  -p 11311:11311 \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e LD_LIBRARY_PATH=/usr/lib/nvidia-535:$LD_LIBRARY_PATH \
   --rm \
  $DOCKER_IMAGE \
  bash