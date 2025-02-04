#!/bin/sh

pushd

cd ..
pytest tests/test_conv.py
popd
