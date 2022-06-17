#!/bin/bash

rm dist/*
export PIP_USER=
python -m build
twine upload dist/*

