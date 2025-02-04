#!/bin/sh

pushd

cd $(git rev-parse --show-toplevel)

python -m pip install -r requirements.txt
python -m pip install -r requirements.extra.txt
python -m pip install -Ue .

popd
