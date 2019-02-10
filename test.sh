#!/bin/bash
source module.sh
pytest lut2d world --ignore outputs --benchmark-columns='min, median, max, rounds, iterations'
