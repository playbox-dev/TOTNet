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
COPY environments/requirements.txt /opt/ml/code/requirements.txt
# NOTE: memory不足なら並列数を下げる
# export MAX_JOBS=2 && \
RUN TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} FORCE_CUDA=1 conda run -n totnet /bin/bash -c "pip install --no-cache-dir -r /opt/ml/code/requirements.txt"

##########################################* ffmpeg builder *###########################################
FROM nvidia/cuda:11.8.0-base-ubuntu22.04 AS ffmpeg_builder
ENV DEBIAN_FRONTEND noninteractive
ENV TZ Etc/UTC
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64
ENV PKG_CONFIG_PATH /usr/local/lib/pkgconfig/
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES video,compute,utility
# Update base and install build tools
RUN apt update && apt -y upgrade
# Install ffmpeg libraries
RUN apt install -y \
  libass-dev \
  libfdk-aac-dev \
  librtmp-dev \
  libssl-dev \
  libvdpau1 \
  libvorbis-dev \
  libvpx-dev \
  libx264-dev \
  libx265-dev \
  build-essential \
  cmake \
  gcc \
  git \
  libc6 \
  libc6-dev \
  libtool \
  nasm \
  nvidia-cuda-toolkit \
  yasm
# Ffmpeg
WORKDIR /src
RUN git clone https://github.com/FFmpeg/nv-codec-headers.git -b old/sdk/9.0 && cd nv-codec-headers && \
  make && make install
RUN git clone https://github.com/FFmpeg/FFmpeg -b release/5.1 && \
  cd /src/FFmpeg && \
  ./configure \
  --prefix="/usr/local" \
  --disable-debug \
  --enable-cuda \
  --enable-cuda-llvm \
  --enable-cuda-nvcc \
  --enable-cuvid \
  --enable-ffnvcodec \
  --enable-gpl \
  --enable-libass \
  --enable-libfdk-aac \
  --enable-libnpp \
  --enable-librtmp \
  --enable-libvorbis \
  --enable-libvpx \
  --enable-libx264 \
  --enable-libx265 \
  --enable-nonfree \
  --enable-nvenc \
  --enable-opencl \
  --enable-openssl \
  --enable-pic \
  --enable-static \
  --extra-cflags="-I/usr/local/nvidia/include/" \
  --extra-ldflags="-L/usr/local/nvidia/lib64/" && \
  make -j$(nproc) && make install

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
  libass-dev \
  libfdk-aac-dev \
  librtmp-dev \
  libssl-dev \
  libvdpau1 \
  libvorbis-dev \
  libvpx-dev \
  libx264-dev \
  libx265-dev \
  libboost-filesystem1.74.0 \
  libboost-program-options1.74.0 \
  libc6 \
  libopengl0 \
  libceres2 \
  libfreeimage3 \
  libgcc-s1 \
  libgl1 \
  libglew2.2 \
  libgoogle-glog0v5 \
  libqt5core5a \
  libqt5gui5 \
  libqt5widgets5 \
  build-essential \
  ocl-icd-opencl-dev \
  libx11-6 && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*
COPY --from=totnet_builder /opt/conda /opt/conda
RUN conda init && \
  echo "conda activate totnet" >> /root/.bashrc
WORKDIR /opt/ml/code
COPY --from=ffmpeg_builder /usr/local/bin /usr/local/bin
COPY --from=ffmpeg_builder /lib/x86_64-linux-gnu/libnppc.so.11 /lib/x86_64-linux-gnu/libnppc.so.11
COPY --from=ffmpeg_builder /lib/x86_64-linux-gnu/libnppicc.so.11 /lib/x86_64-linux-gnu/libnppicc.so.11
COPY --from=ffmpeg_builder /lib/x86_64-linux-gnu/libnppidei.so.11 /lib/x86_64-linux-gnu/libnppidei.so.11
COPY --from=ffmpeg_builder /lib/x86_64-linux-gnu/libnppif.so.11 /lib/x86_64-linux-gnu/libnppif.so.11
COPY --from=ffmpeg_builder /lib/x86_64-linux-gnu/libnppig.so.11 /lib/x86_64-linux-gnu/libnppig.so.11

# Copy source code into the image
COPY src /opt/ml/code/src
