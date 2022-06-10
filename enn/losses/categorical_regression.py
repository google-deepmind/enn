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

"""Implementing categorical regression (MuZero-style) in JAX."""

import dataclasses
import chex
from enn import base_legacy
from enn import networks
from enn.losses import single_index
import haiku as hk
import jax
import jax.numpy as jnp
import rlax


def transform_to_2hot(target: base_legacy.Array,
                      support: base_legacy.Array) -> base_legacy.Array:
  """Converts a scalar target to a 2-hot encoding of the nearest support."""
  target = jnp.clip(target, support.min(), support.max())
  high_idx = jnp.sum(support < target)
  num_bins = len(support)

  low_value = support[high_idx - 1]
  high_value = support[high_idx]
  prob = (target - high_value) / (low_value - high_value)

  lower_one_hot = prob * rlax.one_hot(high_idx - 1, num_bins)
  upper_one_hot = (1 - prob) * rlax.one_hot(high_idx, num_bins)

  return  lower_one_hot + upper_one_hot


@dataclasses.dataclass
class Cat2HotRegression(single_index.SingleIndexLossFn):
  """Apply categorical loss to 2-hot regression target."""

  def __call__(self, apply: base_legacy.ApplyFn, params: hk.Params,
               batch: base_legacy.Batch,
               index: base_legacy.Index) -> base_legacy.Array:
    chex.assert_shape(batch.y, (None, 1))
    chex.assert_shape(batch.data_index, (None, 1))

    # Forward network and check type
    net_out = apply(params, batch.x, index)
    assert isinstance(net_out, networks.CatOutputWithPrior)

    # Form the target values in real space
    target_val = batch.y - net_out.prior

    # Convert values to 2-hot target probabilities
    probs = jax.vmap(transform_to_2hot, in_axes=[0, None])(
        jnp.squeeze(target_val), net_out.extra['atoms'])
    probs = jnp.expand_dims(probs, 1)
    xent_loss = -jnp.sum(probs * jax.nn.log_softmax(net_out.train), axis=-1)
    if batch.weights is None:
      batch_weights = jnp.ones_like(batch.data_index)
    else:
      batch_weights = batch.weights
    chex.assert_equal_shape([batch_weights, xent_loss])
    return jnp.mean(batch_weights * xent_loss), {}
