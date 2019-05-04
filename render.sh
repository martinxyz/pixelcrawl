#!/bin/bash
dir="$1"
shift

if ! [ -d "$dir" ] ; then
    echo "Existing directory required."
    exit 1
fi

rm -f "$dir"/render-world*.png

# param_file=$(ls "$dir"/mean-*.dat|sort|tail -n 1)
param_file=$(ls "$dir"/xfavorite-*.dat|sort|tail -n 1)
echo "using $param_file"

set -e -x
./train.sh render with render="$param_file" "$dir"/[0-9]*/config.json "$@"
set +x
feh -Z -g'600x600' --force-aliasing "$dir"/render-world*.png

