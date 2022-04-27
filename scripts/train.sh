#!/usr/bin/env bash

# repo_dir: root directory of the project
repo_dir="$( cd "$( dirname "$0" )" && pwd )"/..
cd "${repo_dir}"
echo "==== Working directory: ====" >&2
echo "${repo_dir}" >&2
echo "============================" >&2

bash scripts/install.sh

# pip uninstall numpy
# pip install numpy

export MKL_THREADING_LAYER=GNU
export PYTHONPATH="."

export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0

python3 toolbox/train.py $@
