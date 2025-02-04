#!/bin/sh

pushd

cd $(git rev-parse --show-toplevel)
python run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05

popd
