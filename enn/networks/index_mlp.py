# python3
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

"""Implementing some types of epistemic neural networks in JAX."""

from typing import Sequence

from enn import base
from enn import utils
from enn.networks import indexers
from enn.networks import priors
import haiku as hk
import jax
import jax.numpy as jnp


class ConcatIndexMLP(base.EpistemicModule):
  """An MLP that has an d-dimensional index concatenated to every layer."""
  # TODO(author2): Rationalize the type behaviour of network outputs.

  def __init__(self,
               output_sizes: Sequence[int],
               index_dim: int,
               variance_dim: int,
               name: str = 'concat_index_mlp'):
    super().__init__(name=name)
    self.index_dim = index_dim
    self.hiddens = output_sizes[:-1]
    self.output_dim = output_sizes[-1]
    self.variance_dim = variance_dim

  def __call__(
      self, inputs: base.Array, index: base.Index) -> base.OutputWithPrior:
    """Index must be of shape (index_dim,) intended to be Gaussian."""
    batch_size = inputs.shape[0]
    batched_index = jnp.repeat(jnp.expand_dims(index, 0), batch_size, axis=0)
    flat_inputs = hk.Flatten()(inputs)
    out = flat_inputs
    out_no_index = out
    for hidden in self.hiddens:
      out_no_index = out
      out = hk.Linear(hidden)(jnp.concatenate([out, batched_index], axis=1))
      out = jax.nn.relu(out)

    # Make the variance predictor
    input_projection = hk.nets.MLP(
        [self.variance_dim], activate_final=True)(flat_inputs)
    var_embedding = jnp.concatenate([out_no_index, input_projection], axis=1)
    var_pred = hk.nets.MLP([self.variance_dim, self.output_dim])(var_embedding)
    return base.OutputWithPrior(
        train=hk.Linear(self.output_dim)(out),
        extra={'log_var': var_pred},
    )


class IndexMLPWithGpPrior(base.EpistemicNetwork):
  """An Index MLP with GP prior as an ENN."""

  def __init__(self,
               output_sizes: Sequence[int],
               input_dim: int,
               num_prior: int,
               num_feat: int,
               variance_dim: int = 20,
               gamma: priors.GpGamma = 1.,
               prior_scale: float = 1,
               seed: int = 0):
    """An Index MLP with GP prior as an ENN."""
    rng = hk.PRNGSequence(seed)
    def net_fn(x, z):
      net = ConcatIndexMLP(
          output_sizes=output_sizes,
          index_dim=num_prior+1,
          variance_dim=variance_dim,
      )
      return net(x, z)
    transformed = hk.without_apply_rng(hk.transform(net_fn))

    output_dim = output_sizes[-1]
    prior_fns = [priors.make_null_prior(output_dim)]
    for _ in range(num_prior):
      prior_fns.append(priors.make_random_feat_gp(
          input_dim, output_dim, num_feat, next(rng), gamma))

    def apply(params: hk.Params,
              inputs: base.Array,
              index: base.Index) -> base.OutputWithPrior:
      """Forward the SpecialMLP and also the prior network with index."""
      net_out = transformed.apply(params, inputs, index)
      all_priors = [prior(inputs) for prior in prior_fns]
      prior_fn = jnp.sum(jnp.stack(all_priors, axis=-1) * index, axis=-1)
      return net_out._replace(prior=prior_scale * prior_fn)

    super().__init__(
        apply=apply,
        init=transformed.init,
        indexer=indexers.GaussianWithUnitIndexer(num_prior + 1),
    )


class IndexMLPEnn(base.EpistemicNetwork):
  """An MLP with index appended to each layer as ENN."""

  def __init__(self,
               output_sizes: Sequence[int],
               index_dim: int,
               variance_dim: int = 20):
    """An MLP with index appended to each layer as ENN."""
    enn = utils.epistemic_network_from_module(
        enn_ctor=lambda: ConcatIndexMLP(output_sizes, index_dim, variance_dim),
        indexer=indexers.GaussianWithUnitIndexer(index_dim),
    )
    apply = lambda p, x, z: enn.apply(p, x, z).train
    super().__init__(apply, enn.init, enn.indexer)
