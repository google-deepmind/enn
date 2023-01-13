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
from enn import networks
from enn.networks.bert import base
import haiku as hk
import jax


def combine_naive_enn(
    head_enn: networks.EnnArray,
    base_enn: base.BertEnn,
) -> base.BertEnn:
  """Combines a base enn and a head enn naively without optimization.

  Note: It is assumed that the base enn has identity indexer.

  Args:
    head_enn: An EnnArray which is applied to the output of the base_enn.
    base_enn: A BertEnn which takes the inputs and returns the input for the
      head_enn.
  Returns:
    A combined Enn.
  """
  def apply(
      params: hk.Params,
      state: hk.State,
      inputs: base.BertInput,
      index: enn_base.Index,
  ) -> tp.Tuple[networks.OutputWithPrior, hk.State]:
    """Applies the base enn and head enn."""

    # Forward the base enn
    # Since indexer is PrngIndexer, index is actually a random key.
    key = index
    base_out, base_state = base_enn.apply(params, state, inputs, key)
    base_out = networks.parse_net_output(base_out)

    # Forward the head enn
    head_index = head_enn.indexer(key)
    head_out, head_state = head_enn.apply(
        params, state, base_out, head_index)

    # Combine the state for the base and the head enns.
    state = {**base_state, **head_state}

    return head_out, state

  def init(key: chex.PRNGKey,
           inputs: base.BertInput,
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
    base_out = networks.parse_net_output(base_out)

    # initialize the head enn.
    head_index = head_enn.indexer(head_enn_key)
    head_params, head_state = head_enn.init(head_enn_key, base_out, head_index)

    # Combine the params, state for the base and the head enns.
    params = {**head_params, **base_params}
    state = {**head_state, **base_state}
    return (params, state)

  return base.BertEnn(apply, init, networks.PrngIndexer())


def make_optimized_forward(
    head_enn: networks.EnnArray,
    base_enn: base.BertEnn,
    num_enn_samples: int,
    key: chex.PRNGKey,
) ->  networks.EnnBatchFwd:
  """Combines base enn and head enn for multiple ENN samples.

  Note: It is assumed that the base enn has identity indexer.

  Args:
    head_enn: An EnnArray which is applied to the output of the base_enn.
    base_enn: A BertEnn which takes the inputs and returns the input for the
      head_enn.
    num_enn_samples: Number of enn samples to return for each input.
    key: A random key.

  Returns:
    An optimized forward function of combined Enns.
  """
  enn_keys = jax.random.split(key, num_enn_samples)

  def enn_batch_fwd(params: hk.Params,
                    state: hk.State,
                    x: base.BertInput) -> chex.Array:
    base_out, _ = base_enn.apply(params, state, x, key)
    base_out = networks.parse_net_output(base_out)

    def sample_logits(sub_key: chex.PRNGKey) -> chex.Array:
      index = head_enn.indexer(sub_key)
      out, _ = head_enn.apply(params, state, base_out, index)
      return networks.parse_net_output(out)

    return jax.vmap(sample_logits)(enn_keys)

  return jax.jit(enn_batch_fwd)
