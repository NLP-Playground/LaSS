#!/usr/bin/env bash

# repo_dir: root directory of the project
repo_dir="$( cd "$( dirname "$0" )" && pwd )"/..
cd "${repo_dir}"
echo "==== Working directory: ====" >&2
echo "${repo_dir}" >&2
echo "============================" >&2

bash scripts/install.sh

pip uninstall numpy
pip install numpy

python3 -m fairseq_code.toolbox.generate_from_config $@

