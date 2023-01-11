# pylint: disable=g-bad-file-header
# Copyright 2023 DeepMind Technologies Limited. All Rights Reserved.
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
"""Exposing active learning."""

# Base classes
from enn.active_learning.base import ActiveLearner
from enn.active_learning.base import PerExamplePriority
from enn.active_learning.base import PriorityFn
from enn.active_learning.base import PriorityFnCtor
from enn.active_learning.base import PriorityOutput

# Priorities
from enn.active_learning.priorities import get_implemented_priority_fn_ctors
from enn.active_learning.priorities import get_per_example_priority
from enn.active_learning.priorities import get_priority_fn_ctor
from enn.active_learning.priorities import make_priority_fn_ctor
from enn.active_learning.priorities import make_scaled_mean_per_example
from enn.active_learning.priorities import make_scaled_std_per_example
from enn.active_learning.priorities import make_ucb_per_example

# Prioritized
from enn.active_learning.prioritized import PrioritizedBatcher
