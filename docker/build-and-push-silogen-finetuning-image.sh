#!/bin/bash
# This utility script builds and pushes the silogen-finetuning-worker image
# First it ensures a local version of the finetuning-base image, which it depends on
# This needs to be called from repository root
set -eu

DEFAULT_REGISTRY=ghcr.io/silogen
REGISTRY=${REGISTRY:-$DEFAULT_REGISTRY}

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <platform> <tag>"
  echo " e.g.: $0 rocm main-$(git rev-parse --short HEAD)"
  echo " This makes the images $REGISTRY/rocm-silogen-finetuning-base:main-xxgithashxx"
  echo " And $REGISTRY/rocm-silogen-finetuning-worker:main-xxgithashxx"
  echo " And pushes them"
  echo "Additionally, use the REGISTRY environment variable to change where the image gets pushed."
  echo " e.g.: REGISTRY=my-domain/my-registry $0 rocm v0.1"
  echo " pushes my-domain/my-registry/rocm-silogen-finetuning-worker:main-xxgithashxx"
  echo " $DEFAULT_REGISTRY is the default value for REGISTRY"
  exit 1
fi
platform="$1"
tag="$2"

tmp_worker_dockerfile=$(mktemp docker/tmp.$platform-silogen-finetuning-worker.Dockerfile.XXXXXXXXXX)
trap 'rm -f "$tmp_worker_dockerfile"' EXIT
sed "s,FROM $DEFAULT_REGISTRY/$platform-silogen-finetuning-base:main,FROM $REGISTRY/$platform-silogen-finetuning-base:$tag,g" \
  docker/$platform-silogen-finetuning-worker.Dockerfile > $tmp_worker_dockerfile
docker build -f docker/$platform-silogen-finetuning-base.Dockerfile -t "$REGISTRY/$platform-silogen-finetuning-base:$tag" .
docker build -f $tmp_worker_dockerfile -t "$REGISTRY/$platform-silogen-finetuning-worker:$tag" --build-arg tag="$tag" .
docker push "$REGISTRY/$platform-silogen-finetuning-base:$tag"
docker push "$REGISTRY/$platform-silogen-finetuning-worker:$tag"
