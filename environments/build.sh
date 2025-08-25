#!/usr/bin/env bash
####################### settings ###########################
default_tag="dev"
cuda_arch="75;89"
############################################################

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the project root directory (parent of environments)
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

tag="$default_tag"
# example: 75;89 -> 7.5;8.9
torch_cuda_arch=$(echo "$cuda_arch" | sed 's/\([0-9]\)\([0-9]\)/\1.\2/g')

# docker build kit
export DOCKER_BUILDKIT=1

# Create log directory if it doesn't exist
mkdir -p "${SCRIPT_DIR}/log"

# 共通のビルドオプション
common_build_opts=(
  --platform linux/amd64
  --progress=plain
  --network=host
)

# training image build
build_image() {
  local image=$1
  local dockerfile=$2
  local context=$3
  local extra_args=("${@:4}")

  echo "Building ${image}:${tag}"
  echo "Using Dockerfile: ${dockerfile}"
  echo "Build context: ${context}"

  docker build "${common_build_opts[@]}" \
    -f "${dockerfile}" \
    -t "${image}:${tag}" \
    "${extra_args[@]}" \
    "${context}" 2>&1 | tee "${SCRIPT_DIR}/log/build_${image}.log"
}

# Execute build from project root
cd "${PROJECT_ROOT}" || exit

build_image \
  "totnet" \
  "${SCRIPT_DIR}/Dockerfile" \
  "." \
  "--build-arg" \
  "CUDA_ARCHITECTURES=${cuda_arch}" \
  "--build-arg" \
  "TORCH_CUDA_ARCH_LIST=${torch_cuda_arch}"
