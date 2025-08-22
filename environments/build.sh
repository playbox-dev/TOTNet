#!/usr/bin/env bash
####################### settings ###########################
default_tag="dev"
cuda_arch="75;89"
############################################################

tag="$default_tag"
# example: 75;89 -> 7.5;8.9
torch_cuda_arch=$(echo "$cuda_arch" | sed 's/\([0-9]\)\([0-9]\)/\1.\2/g')

# docker build kit
DOCKER_BUILDKIT=1

# 共通のビルドオプション
common_build_opts=(
  --platform linux/amd64
  --progress=plain
  --network=host
)

# training image build
build_image() {
  local image=$1
  local dockerfile_dir=$2
  local extra_args=("${@:3}")

  echo "Building ${image}:${tag}"

  # 指定されたディレクトリに移動してビルドを実行
  cd "${dockerfile_dir}" || exit

  docker build "${common_build_opts[@]}" \
    -f "./Dockerfile" \
    -t "${image}:${tag}" \
    "${extra_args[@]}" \
    . 2>&1 | tee "${SCRIPT_DIR}/log/build_${image}.log"
}

build_image \
  "totnet" \
  "./" \
  "--build-arg" \
  "CUDA_ARCHITECTURES=${cuda_arch}" \
  "--build-arg" \
  "TORCH_CUDA_ARCH_LIST=${torch_cuda_arch}"
