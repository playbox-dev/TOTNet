whereis cuda # show where

# set correct path
export CUDA_HOME=/ceph-g/opt/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

nvcc --version

pip install causal-conv1d>=1.4.0
pip install mamba-ssm[dev]