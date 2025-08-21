##########################################* totnet builder *##########################################
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 AS totnet_builder
COPY --from=continuumio/miniconda3:23.10.0-1 /opt/conda /opt/conda
ARG SCRIPTS_BRANCH="main"
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/opt/conda/bin:$PATH
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ARG CUDA_ARCHITECTURES=75
ENV CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}
ARG TORCH_CUDA_ARCH_LIST=7.5
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV CUDA_VISIBLE_DEVICES=0
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN --mount=type=cache,target=/var/cache/apt,sharing=private \
  rm -f /etc/apt/apt.conf.d/docker-clean && \
  echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache && \
  apt-get update && \
  apt-get install -y --no-install-recommends --reinstall \
  git \
  cuda-minimal-build-11-8 && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ml/code
RUN conda create -n totnet python=3.10
RUN conda run -n totnet /bin/bash -c "pip install --no-cache-dir --upgrade pip setuptools==69.5.1 pathtools" && \
  conda run -n totnet /bin/bash -c "pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/test/cu118"
# WARNING: ブランチが更新されてもCacheが効いて再ビルドされないので注意
RUN git clone https://github.com/AugustRushG/TOTNet.git && \
  # NOTE: memory不足なら並列数を下げる
  # export MAX_JOBS=2 && \
  TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} FORCE_CUDA=1 conda run -n totnet /bin/bash -c "pip install --no-cache-dir -r TOTNet/requirements.txt"

##########################################* totnet *##########################################
FROM nvidia/cuda:11.8.0-base-ubuntu22.04 AS totnet
ENV PATH=/opt/conda/bin:$PATH
RUN --mount=type=cache,target=/var/cache/apt,sharing=private \
  rm -f /etc/apt/apt.conf.d/docker-clean && \
  echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache && \
  apt-get update && \
  apt-get install -y --no-install-recommends \
  libgl1-mesa-glx \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender-dev \
  libgomp1 \
  libglib2.0-0 \
  libx11-6 && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*
COPY --from=totnet_builder /opt/conda /opt/conda
RUN conda init && \
  echo "conda activate totnet" >> /root/.bashrc
WORKDIR /opt/ml/code
