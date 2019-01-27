#!/bin/bash
source module.sh

./train.py "$@"
# perf record -g ./train.py "$@"
