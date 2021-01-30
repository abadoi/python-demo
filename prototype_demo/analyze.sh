#!/bin/sh
export ML_MODEL="data/results/classifiers/RandomForestClassifier.pkl"
# export ML_MODEL="data/results/classifiers/LinearSVC.pkl"
export PATTERN_MATCHING_TOOL="./build/simple-tool"

python3 analyzer.py '-b' $1

