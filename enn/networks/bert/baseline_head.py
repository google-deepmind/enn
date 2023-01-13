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
"""Baseline last layer ENN which is a small mlp with dropout."""
import typing as tp

import chex
from enn import base as enn_base
from enn import networks
import haiku as hk
import jax.numpy as jnp


def make_enn(num_classes: int, is_training: bool) -> networks.EnnArray:
  """Makes an enn of a small mlp with dropout."""

  def net_fn(inputs: chex.Array) -> networks.OutputWithPrior:
    """Forwards the network."""
    classification_layer = CommonOutputLayer(
        num_classes=num_classes,
        use_extra_projection=False,
        dropout_rate=0.1,
    )
    outputs = classification_layer(inputs, is_training=is_training)

    # Wrap in ENN output layer
    return networks.OutputWithPrior(outputs, prior=jnp.zeros_like(outputs))

  # Transformed has the rng input, which we need to change --> index.
  transformed = hk.transform_with_state(net_fn)
  def apply(
      params: hk.Params,
      state: hk.State,
      inputs: chex.Array,
      index: enn_base.Index,
  ) -> tp.Tuple[networks.OutputWithPrior, hk.State]:
    return transformed.apply(params, state, index, inputs)
  def init(rng_key: chex.PRNGKey,
           inputs: chex.Array,
           index: enn_base.Index) -> tp.Tuple[hk.Params, hk.State]:
    del index  # rng_key is duplicated in this case.
    return transformed.init(rng_key, inputs)

  return networks.EnnArray(apply, init, networks.PrngIndexer())


class CommonOutputLayer(hk.Module):
  """Finetuning layer for downstream tasks.

  This is the Haiku module of the classifier implemented in tensorflow here:
  https://github.com/google-research/bert/blob/master/run_classifier.py#L574
  """

  def __init__(
      self,
      num_classes: int,
      dropout_rate: float = 0.1,
      use_extra_projection: bool = True,
      name: str = 'output_layer',
  ):
    """Initialises the module.

    Args:
      num_classes: Number of classes in the downstream task.
      dropout_rate: Dropout rate.
      use_extra_projection: Whether to add an extra linear layer with a tanh
        activation is added before computing the output value.
      name: Haiku module name.
    """
    super().__init__(name=name)
    self._num_classes = num_classes
    self._dropout_rate = dropout_rate
    self._use_extra_projection = use_extra_projection

  def __call__(
      self,
      inputs: jnp.DeviceArray,
      is_training: bool = True,
  ) -> jnp.DeviceArray:
    """Compute the classification logits.

    Args:
      inputs: A tensor of shape (B, d_model) containing a summary of the
        sequence to regress
      is_training: `bool` if True dropout is applied.

    Returns:
      A tensor of shape (B,) containing the regressed values.
    """

    output = inputs
    if self._use_extra_projection:
      d_model = output.shape[-1]
      output = hk.Linear(
          d_model,
          w_init=hk.initializers.RandomNormal(stddev=0.02))(output)
      output = jnp.tanh(output)

    if is_training:
      output = hk.dropout(
          rng=hk.next_rng_key(), rate=self._dropout_rate, x=output)

    output = hk.Linear(
        self._num_classes,
        w_init=hk.initializers.RandomNormal(stddev=0.02))(output)
    return output
