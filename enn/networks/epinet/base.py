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
"""Base classes for epinet."""
import dataclasses
from typing import Callable, Optional, Tuple

from enn import base_legacy as enn_base
from enn import utils
from enn.checkpoints import base as checkpoint_base
import haiku as hk
import jax
from typing_extensions import Protocol


class EpinetApplyWithState(Protocol):
  """Applies the epinet at given parameters and state."""

  def __call__(
      self,
      params: hk.Params,  # Epinet parameters
      state: hk.State,  # Epinet state
      inputs: enn_base.Array,  # ENN inputs = x
      index: enn_base.Index,  # ENN index = z
      hidden: enn_base.Array,  # Base net hiddens = phi(x)
  ) -> Tuple[enn_base.OutputWithPrior, hk.State]:
    """Applies the epinet at given parameters and state."""


class EpinetInitWithState(Protocol):
  """Initializes epinet parameters and state."""

  def __call__(
      self,
      key: enn_base.RngKey,  # Random key
      inputs: enn_base.Array,  # ENN inputs = x
      index: enn_base.Index,  # ENN index = z
      hidden: enn_base.Array,  # Base net hiddens = phi(x)
  ) -> Tuple[hk.Params, hk.State]:
    """Initializes epinet parameters and state."""


@dataclasses.dataclass
class EpinetWithState:
  """Convenient pairing of Haiku transformed function and index sampler."""
  apply: EpinetApplyWithState
  init: EpinetInitWithState
  indexer: enn_base.EpistemicIndexer


BaseHiddenParser = Callable[[enn_base.Output], enn_base.Array]


def combine_base_epinet_as_enn(
    base_checkpoint: checkpoint_base.EnnCheckpoint,
    epinet: EpinetWithState,
    parse_hidden: BaseHiddenParser,
    base_index: Optional[enn_base.Index] = None,
    base_scale: float = 1,
    freeze_base: bool = True,
) -> enn_base.EpistemicNetworkWithState:
  """Returns a combined ENN from a base network and an epinet.

  Args:
    base_checkpoint: checkpoint of base model ENN.
    epinet: Epinet to be combined.
    parse_hidden: Function to obtain hidden representation from base_out.
    base_index: Optional index applied to base_enn, otherwise seed=0.
    base_scale: rescale output of the base network by this.
    freeze_base: If True, then the only parameters/state returned will be
      specific just to the epinet. If False, then the parameters/state are
      combined with those of the base network. Useful for finetuning.
  """
  # TODO(author2): Add testing to this function.

  # Parse the base network from checkpoint
  base_params_init, base_state_init = base_checkpoint.load_fn()
  base_enn = base_checkpoint.enn_ctor()
  if base_index is None:
    base_index = base_enn.indexer(jax.random.PRNGKey(0))

  def apply(params: hk.Params,
            state: hk.State,
            inputs: enn_base.Array,
            index: enn_base.Index) -> Tuple[enn_base.OutputWithPrior, hk.State]:
    """Applies the base network and epinet."""
    if freeze_base:
      base_params, base_state = base_params_init, base_state_init
    else:
      base_params, base_state = params, state

    # Forward the base network
    base_out, base_state = base_enn.apply(
        base_params, base_state, inputs, base_index)
    base_out = utils.parse_to_output_with_prior(base_out)

    # Forward the epinet and combine their outputs
    epi_out, epi_state = epinet.apply(
        params, state, inputs, index, parse_hidden(base_out))

    output = enn_base.OutputWithPrior(
        train=base_out.train * base_scale + epi_out.train,
        prior=base_out.prior * base_scale + epi_out.prior,
    )
    state = epi_state if freeze_base else {**base_state, **epi_state}
    return output, state

  def init(key: enn_base.RngKey,
           inputs: enn_base.Array,
           index: enn_base.Index) -> Tuple[hk.Params, hk.State]:
    """Initializes the epinet."""
    base_out, unused_base_state = base_enn.apply(
        base_params_init, base_state_init, inputs, base_index)
    params, state = epinet.init(
        key, inputs, index, parse_hidden(base_out))

    if freeze_base:
      return params, state
    else:
      return {**params, **base_params_init}, {**state, **base_state_init}

  return enn_base.EpistemicNetworkWithState(apply, init, epinet.indexer)
