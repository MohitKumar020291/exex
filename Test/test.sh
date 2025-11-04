#!/bin/sh

echo "Running test"

if [ $# -eq 0 ]; then
    echo "Running all tests"
    python -m unittest discover -s Test -p "test_*.py" -v
    exit 0
fi

for arg in "$@"; do
    python -m unittest "$arg"
done
