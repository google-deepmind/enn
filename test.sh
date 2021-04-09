#!/bin/bash

# Fail on any error.
set -e
# Display commands being run.
set -x

# Set up a new virtual environment.
python3 -m venv enn_testing --upgrade-deps
source enn_testing/bin/activate

# Install all dependencies.
pip install .

# Install test dependencies.
pip install .[testing]

N_CPU=$(grep -c ^processor /proc/cpuinfo)

# Run static type-checking.
pytype -j "${N_CPU}" enn

# Run all tests.
pytest -n "${N_CPU}" enn

# Clean-up.
deactivate
rm -rf enn_testing/
