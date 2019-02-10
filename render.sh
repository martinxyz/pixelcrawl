#!/bin/bash
dir=$1
rm -f "$dir"/render-world*.png
set -e -x
./train.sh render with render=$dir $dir/*/config.json
set +x
feh -Z -g'600x600' --force-aliasing $dir/render-world*.png
