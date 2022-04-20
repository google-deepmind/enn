#!/bin/bash
# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# Fail on any error.
set -e
# Display commands being run.
set -x

# Set up a new virtual environment.
python3 -m venv enn_testing
source enn_testing/bin/activate

# This prints the path to the python binary we are using. It should be inside
# the virtualenv you just created (in this case the path should end in
# enn_testing/bin/python).
which python

# This installs and/or updates `pip` and `wheel`. It's good practice to do this
# before you attempt to install anything else inside your virtualenv. In
# particular if `wheel` is not installed then Python may fall back on building
# some packages from source, rather than using a precompiled version.
python -m pip install -U pip wheel

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
