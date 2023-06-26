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
"""Last layer classification heads for BERT model."""
import dataclasses
from typing import Optional, Sequence, Tuple

import chex
from enn import base as enn_base
from enn.networks import base as networks_base
from enn.networks import dropout
from enn.networks import einsum_mlp
from enn.networks import epinet
from enn.networks import indexers
import haiku as hk
import jax
import jax.numpy as jnp


@dataclasses.dataclass
class AgentConfig:
  """Agent configuration."""
  hiddens: Sequence[int] = (50, 50)
  # For ensemble agent
  num_ensemble: int = 10
  # For dropout agent
  dropout_rate: float = 0.1
  # For epinet agent
  index_dim: int = 30
  prior_scale: float = 1


def make_head_enn(
    agent: str,
    num_classes: int,
    agent_config: Optional[AgentConfig] = None) -> networks_base.EnnArray:
  """Returns a last layer (head) enn."""
  if agent_config is None:
    agent_config = AgentConfig()
  output_sizes = list(agent_config.hiddens) + [num_classes]
  if agent == 'epinet':
    # We don't want to expose any layers. This means that only the inputs are
    # passed to epinet.
    expose_layers = [False] * len(output_sizes)
    return epinet.make_mlp_epinet(output_sizes=output_sizes,
                                  epinet_hiddens=agent_config.hiddens,
                                  index_dim=agent_config.index_dim,
                                  expose_layers=expose_layers,
                                  prior_scale=agent_config.prior_scale,
                                  stop_gradient=True)
  elif agent == 'ensemble':
    return einsum_mlp.make_einsum_ensemble_mlp_enn(
        output_sizes=output_sizes,
        num_ensemble=agent_config.num_ensemble,
    )
  elif agent == 'dropout':
    return dropout.MLPDropoutENN(
        output_sizes=output_sizes,
        dropout_rate=agent_config.dropout_rate,
        dropout_input=False,
    )
  else:
    raise ValueError(f'Invalid agent: {agent}!')


def make_baseline_head_enn(
    num_classes: int, is_training: bool
) -> networks_base.EnnArray:
  """Makes an enn of the baseline classifier head."""

  def net_fn(inputs: chex.Array) -> networks_base.OutputWithPrior:
    """Forwards the network."""
    classification_layer = CommonOutputLayer(
        num_classes=num_classes,
        use_extra_projection=False,
        dropout_rate=0.1,
    )
    outputs = classification_layer(inputs, is_training=is_training)

    # Wrap in ENN output layer
    return networks_base.OutputWithPrior(outputs, prior=jnp.zeros_like(outputs))

  # Transformed has the rng input, which we need to change --> index.
  transformed = hk.transform_with_state(net_fn)
  def apply(
      params: hk.Params,
      state: hk.State,
      inputs: chex.Array,
      index: enn_base.Index,
  ) -> Tuple[networks_base.OutputWithPrior, hk.State]:
    return transformed.apply(params, state, index, inputs)
  def init(rng_key: chex.PRNGKey,
           inputs: chex.Array,
           index: enn_base.Index) -> Tuple[hk.Params, hk.State]:
    del index  # rng_key is duplicated in this case.
    return transformed.init(rng_key, inputs)

  return networks_base.EnnArray(apply, init, indexers.PrngIndexer())


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
      inputs: jax.Array,
      is_training: bool = True,
  ) -> jax.Array:
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

