#!/bin/sh

pushd

cd $(git rev-parse --show-toplevel)
python project/run_sentiment.py

popd
