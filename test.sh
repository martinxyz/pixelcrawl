#!/bin/bash
source module.sh
pytest "$@" --benchmark-columns='min, median, max, rounds, iterations'
