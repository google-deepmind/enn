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
"""Factory methods for epinet derived from resnet."""

import dataclasses
from typing import Optional, Sequence, Callable

from enn import base as enn_base
from enn.checkpoints import base as checkpoints_base
from enn.checkpoints import epinet as checkpoints_epinet
from enn.networks import priors as enn_priors
from enn.networks.epinet import base as epinet_base
from enn.networks.epinet import last_layer
from enn.networks.epinet import priors


PriorFnCtor = Callable[[], enn_priors.PriorFn]


@dataclasses.dataclass
class ResnetFinalEpinetConfig:
  """Configuration options for ResNet + final layer MLP epinet."""
  base_checkpoint: checkpoints_base.EnnCheckpoint
  index_dim: int
  num_classes: int
  epinet_hiddens: Sequence[int]
  prior_epinet_hiddens: Optional[Sequence[int]] = None
  base_logits_scale: float = 1
  epi_prior_scale: float = 1
  add_prior_scale: float = 1
  prior_fn_ctor: Optional[PriorFnCtor] = None
  freeze_base: bool = True
  parse_hidden: epinet_base.BaseHiddenParser = last_layer.parse_base_hidden
  seed: int = 23
  temperature: Optional[float] = None


class ResnetFinalEpinet(enn_base.EpistemicNetworkWithState):
  """ResNet + final layer MLP epinet."""

  def __init__(self, config: ResnetFinalEpinetConfig):
    epinet_pieces = _make_enn_from_config(config)
    enn = epinet_pieces.enn
    super().__init__(enn.apply, enn.init, enn.indexer)


def make_checkpoint_from_config(
    name: str,
    load_fn: checkpoints_base.ParamsStateLoadFn,
    config: ResnetFinalEpinetConfig,
) -> checkpoints_epinet.EpinetCheckpoint:
  """Returns an EpinetCheckpoint from ResnetFinalEpinetConfig.

  Args:
    name: string identifier for checkpoint.
    load_fn: gives params, state for epinet. Base network init from base_cpt.
    config: ResnetFinalEpinetConfig, which includes base_cpt as component.
  """
  return checkpoints_epinet.EpinetCheckpoint(
      name=name,
      load_fn=load_fn,
      epinet_ctor=lambda: _make_enn_from_config(config).epinet,
      parse_hidden=config.parse_hidden,
      base_cpt=config.base_checkpoint,
      base_scale=config.base_logits_scale,
      temperature=config.temperature,
  )


@dataclasses.dataclass
class _EpinetPieces:
  """Wraps the key components necessary to create either ENN or checkpoint."""
  enn: enn_base.EpistemicNetworkWithState  # Entire network (base+epi) as ENN.
  epinet: epinet_base.EpinetWithState  # Epinet applied on top of base net.


def _make_enn_from_config(config: ResnetFinalEpinetConfig) -> _EpinetPieces:
  """Wires together the epinet."""
  # Make the last layer epinet
  epinet = last_layer.MLPEpinetWithPrior(
      index_dim=config.index_dim,
      num_classes=config.num_classes,
      epinet_hiddens=config.epinet_hiddens,
      prior_epinet_hiddens=config.prior_epinet_hiddens,
      prior_scale=config.epi_prior_scale,
      drop_inputs=True,
  )

  if config.prior_fn_ctor:
    # Form the extra additive prior functions
    prior_fn = config.prior_fn_ctor()

    # Combined epinet is epinet_head with additive prior
    epinet = priors.combine_epinet_and_prior(
        epinet, prior_fn, config.add_prior_scale)

  # Form the ENN by combining them all
  enn = epinet_base.combine_base_epinet_as_enn(
      base_checkpoint=config.base_checkpoint,
      epinet=epinet,
      parse_hidden=config.parse_hidden,
      base_scale=config.base_logits_scale,
      freeze_base=config.freeze_base,
      temperature=config.temperature,
  )

  return _EpinetPieces(enn, epinet)

