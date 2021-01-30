#!/bin/sh
export CC="/usr/local/opt/llvm/bin/clang" 
export DATA_DIR="data/train_data_extra_features"
export RESULTS_DIR="data/results_extra_features"
export PATTERN_MATCHING_TOOL="./build/simple-tool"

python3 train.py

