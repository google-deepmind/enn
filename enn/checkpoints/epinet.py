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
from typing import Callable, Optional
from enn import base_legacy as enn_base
from enn import datasets
from enn.checkpoints import base as checkpoint_base
from enn.networks.epinet import base as epinet_base


@dataclasses.dataclass
class EpinetCheckpoint:
  """Specialized checkpoint for epinet architectures.

  You *can* just save the whole (base + epinet) as an ENN checkpoint.
  However, if you break it down like this explicitly, then you can have an
  optimized forward function over multiple ENN samples without recomputing
  the base network each time.
  """
  name: str  # A string to describe this checkpoint entry.
  load_fn: checkpoint_base.ParamsStateLoadFn  # Restores params, state epinet.
  epinet_ctor: Callable[[], epinet_base.EpinetWithState]  # Epinet model
  parse_hidden: epinet_base.BaseHiddenParser  # Parse the hidden representation.
  base_cpt: checkpoint_base.EnnCheckpoint  # Checkpoint for the base model
  base_index: Optional[enn_base.Index] = None  # Optional specify base_index.
  base_scale: float = 1.  # Scaling of base net output.

  # Optionally rescale ENN outputs by 1/ temperature.
  tuned_eval_temperature: Optional[float] = None

  # Optional attributes used to identify the provenance of these models.
  # This is mostly used *internally*, but can be useful outside too.
  dataset: Optional[datasets.Dataset] = None  # Dataset used in training.
  report_cl: Optional[int] = None  # Integer id for report CL (encouraged).
