#!/bin/bash

if [ ! -f ~/.pypirc ]; then
    echo "Please configure .pypirc for pypi access first"
    exit -2
fi
rm dist/*
export PIP_USER=
python -m build
twine upload dist/*

