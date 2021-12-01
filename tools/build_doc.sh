#!/bin/sh

cd docs

echo "[INFO] start scanning modules..."
sphinx-apidoc -o generated ../pyadts --force --module-first
make html
cd ..