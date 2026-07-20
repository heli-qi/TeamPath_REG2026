#!/usr/bin/env bash

# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_IMAGE_TAG="reg2026_algorithm"

DOCKER_NOOP_VOLUME="${DOCKER_IMAGE_TAG}-volume"

INPUT_DIR="${SCRIPT_DIR}/test/input"
OUTPUT_DIR="${SCRIPT_DIR}/test/output"

echo "=+= (Re)build the container"
source "${SCRIPT_DIR}/do_build.sh"

# GPU passthrough: match example_algorithm flow, but only when a host NVIDIA GPU
# is present AND Docker accepts --gpus (avoids Mac CDI / "warming up" probe hangs).
GPU_ARGS=()
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    if docker run --rm --gpus all \
        --platform=linux/amd64 \
        --entrypoint true \
        "$DOCKER_IMAGE_TAG" >/dev/null 2>&1; then
        GPU_ARGS+=(--gpus all)
        echo "=+= Docker GPU passthrough available (--gpus all)"
    else
        echo "=+= NVIDIA GPU detected but Docker GPU passthrough unavailable; using CPU"
    fi
else
    echo "=+= No NVIDIA GPU on host; container will run on CPU"
fi

cleanup() {
    echo "=+= Cleaning permissions ..."
    # Ensure permissions are set correctly on the output
    # This allows the host user (e.g. you) to access and handle these files
    docker run --rm \
      --platform=linux/amd64 \
      --quiet \
      --volume "$OUTPUT_DIR":/output \
      --entrypoint /bin/sh \
      $DOCKER_IMAGE_TAG \
      -c "chmod -R -f o+rwX /output/* || true"

    # Ensure volume is removed
    docker volume rm "$DOCKER_NOOP_VOLUME" > /dev/null
}

# This allows for the Docker user to read
chmod -R -f o+rX "$INPUT_DIR" "${SCRIPT_DIR}/model"


if [ -d "${OUTPUT_DIR}/interf0" ]; then
  # This allows for the Docker user to write
  chmod -f o+rwX "${OUTPUT_DIR}/interf0"

  echo "=+= Cleaning up any earlier output"
  # Use the container itself to circumvent ownership problems
  docker run --rm \
      --platform=linux/amd64 \
      --quiet \
      --volume "${OUTPUT_DIR}/interf0":/output \
      --entrypoint /bin/sh \
      $DOCKER_IMAGE_TAG \
      -c "rm -rf /output/* || true"
else
  mkdir -p -m o+rwX "${OUTPUT_DIR}/interf0"
fi

if [ -d "${OUTPUT_DIR}/interf1" ]; then
  # This allows for the Docker user to write
  chmod -f o+rwX "${OUTPUT_DIR}/interf1"

  echo "=+= Cleaning up any earlier output"
  # Use the container itself to circumvent ownership problems
  docker run --rm \
      --platform=linux/amd64 \
      --quiet \
      --volume "${OUTPUT_DIR}/interf1":/output \
      --entrypoint /bin/sh \
      $DOCKER_IMAGE_TAG \
      -c "rm -rf /output/* || true"
else
  mkdir -p -m o+rwX "${OUTPUT_DIR}/interf1"
fi


docker volume create "$DOCKER_NOOP_VOLUME" > /dev/null

trap cleanup EXIT

run_docker_forward_pass() {
    local interface_dir="$1"

    echo "=+= Doing a forward pass on ${interface_dir}"

    ## Note the extra arguments that are passed here:
    # '--network none'
    #    entails there is no internet connection
    # "${GPU_ARGS[@]}"
    #    enables access to any GPUs present when Docker GPU passthrough works
    # '--volume <NAME>:/tmp'
    #   is added because on Grand Challenge this directory cannot be used to store permanent files
    # '--volume ../model:/opt/ml/model/":ro'
    #   is added to provide access to the (optional) tarball-upload locally
    docker run --rm "${GPU_ARGS[@]}" \
        --platform=linux/amd64 \
        --network none \
        --volume "${INPUT_DIR}/${interface_dir}":/input:ro \
        --volume "${OUTPUT_DIR}/${interface_dir}":/output \
        --volume "$DOCKER_NOOP_VOLUME":/tmp \
        --volume "${SCRIPT_DIR}/model":/opt/ml/model:ro \
        "$DOCKER_IMAGE_TAG"

  echo "=+= Wrote results to ${OUTPUT_DIR}/${interface_dir}"
}


run_docker_forward_pass "interf0"

run_docker_forward_pass "interf1"



echo "=+= Save this image for uploading via ./do_save.sh"
