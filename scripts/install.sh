#!/bin/sh

pushd

cd ..

!python -m pip install -r requirements.txt
!python -m pip install -r requirements.extra.txt
!python -m pip install -Ue .

popd
