#!/bin/bash

pushd .

cd $(git rev-parse --show-toplevel)

pytest tests/test_conv.py

popd
