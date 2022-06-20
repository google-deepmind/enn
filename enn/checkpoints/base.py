# pylint: disable=g-bad-file-header
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
"""Base file for defining how to restore pre-trained ENNs for evaluation."""

import dataclasses
from typing import Callable, Optional, Tuple

from enn import datasets
from enn import networks
import haiku as hk

EnnCtor = Callable[[], networks.EnnArray]
ParamsStateLoadFn = Callable[[], Tuple[hk.Params, hk.State]]


@dataclasses.dataclass
class EnnCheckpoint:
  """Maintains necessary info to restore an ENN from checkpoint.

  This should only restore *one* ENN for *one* set of hyperparameters/data.
  """
  name: str  # A string to describe this checkpoint entry.
  load_fn: ParamsStateLoadFn  # Restores params, state for use in enn.
  enn_ctor: EnnCtor  # ENN model constructor

  # Optional attributes used to identify the provenance of these models.
  # This is mostly used *internally*, but can be useful outside too.
  dataset: Optional[datasets.Dataset] = None  # Dataset used in training.
  report_cl: Optional[int] = None  # Integer id for report CL (encouraged).

  # Optionally rescale ENN outputs by 1/ temperature.
  tuned_eval_temperature: Optional[float] = None
