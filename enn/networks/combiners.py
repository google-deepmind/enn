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
"""Factory methods to combine multiple ENNs."""

import typing as tp

import chex
from enn import base as enn_base
from enn.networks import base as networks_base
from enn.networks import forwarders
from enn.networks import utils
import haiku as hk
import jax


HeadInput = tp.TypeVar('HeadInput')  # Inputs to head enn
BaseOutput = tp.TypeVar('BaseOutput')  # Outputs of base enn


def combine_naive_enn(
    head_enn: enn_base.EpistemicNetwork[HeadInput, networks_base.Output],
    base_enn: enn_base.EpistemicNetwork[enn_base.Input, BaseOutput],
    parse_base_network: tp.Callable[
        [BaseOutput], HeadInput
    ] = utils.parse_net_output,
) -> enn_base.EpistemicNetwork[enn_base.Input, networks_base.Output]:
  """Combines a base enn and a head enn naively without optimization.

  Note: It is assumed that the base enn has identity indexer.

  Args:
    head_enn: An EnnArray which is applied to the output of the base_enn.
    base_enn: An Enn with generic inputs which takes the inputs and returns the
      input for the head_enn.
    parse_base_network: A callable that parses the desired output from base_enn
      to feed into head_enn.

  Returns:
    A combined Enn.
  """
  def apply(
      params: hk.Params,
      state: hk.State,
      inputs: enn_base.Input,
      index: enn_base.Index,
  ) -> tp.Tuple[networks_base.Output, hk.State]:
    """Applies the base enn and head enn."""

    # Forward the base enn
    # Since indexer is PrngIndexer, index is actually a random key.
    key = index
    base_out, base_state = base_enn.apply(params, state, inputs, key)
    base_out = parse_base_network(base_out)

    # Forward the head enn
    head_index = head_enn.indexer(key)
    head_out, head_state = head_enn.apply(
        params, state, base_out, head_index)

    # Combine the state for the base and the head enns.
    state = {**base_state, **head_state}

    return head_out, state

  def init(key: chex.PRNGKey,
           inputs: enn_base.Input,
           index: enn_base.Index) -> tp.Tuple[hk.Params, hk.State]:
    """Initializes the base enn and the head enn."""
    base_key, head_enn_key = jax.random.split(key)

    # initialize the base enn. Note that these params, state are replaced by the
    # params, state of the pre-trained base in the experiment.
    base_params, base_state = base_enn.init(base_key, inputs, index)

    # Forward the base enn to get output and use it as a dummy input to
    # initialize the head enn.
    base_out, unused_base_state = base_enn.apply(
        base_params, base_state, inputs, index)
    base_out = parse_base_network(base_out)

    # initialize the head enn.
    head_index = head_enn.indexer(head_enn_key)
    head_params, head_state = head_enn.init(head_enn_key, base_out, head_index)

    # Combine the params, state for the base and the head enns.
    params = {**head_params, **base_params}
    state = {**head_state, **base_state}
    return (params, state)

  return enn_base.EpistemicNetwork[enn_base.Input, networks_base.Output](
      apply, init, base_enn.indexer
  )


def make_optimized_forward(
    head_enn: enn_base.EpistemicNetwork[HeadInput, networks_base.Output],
    base_enn: enn_base.EpistemicNetwork[enn_base.Input, BaseOutput],
    num_enn_samples: int,
    key: chex.PRNGKey,
    parse_base_network: tp.Callable[
        [BaseOutput], HeadInput
    ] = utils.parse_net_output,
) -> forwarders.EnnBatchFwd[enn_base.Input]:
  """Combines base enn and head enn for multiple ENN samples.

  Note: It is assumed that the base enn has identity indexer.

  Args:
    head_enn: An EnnArray which is applied to the output of the base_enn.
    base_enn: An Enn with generic inputs which takes the inputs and returns the
      input for the head_enn.
    num_enn_samples: Number of enn samples to return for each input.
    key: A random key.
    parse_base_network: A callable that parses the desired output from base_enn
      to feed into head_enn.

  Returns:
    An optimized forward function of combined Enns.
  """
  enn_keys = jax.random.split(key, num_enn_samples)

  def enn_batch_fwd(params: hk.Params,
                    state: hk.State,
                    x: enn_base.Input) -> chex.Array:
    base_out, _ = base_enn.apply(params, state, x, key)
    base_out = parse_base_network(base_out)

    def sample_logits(sub_key: chex.PRNGKey) -> chex.Array:
      index = head_enn.indexer(sub_key)
      out, _ = head_enn.apply(params, state, base_out, index)
      return utils.parse_net_output(out)

    return jax.vmap(sample_logits)(enn_keys)

  return jax.jit(enn_batch_fwd)
