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

"""Implementing some types of ENN ensembles in JAX."""
from typing import Callable, Optional, Sequence

import chex
from enn import base
from enn import utils
from enn.networks import indexers
from enn.networks import priors
import haiku as hk
import jax
import jax.numpy as jnp


class Ensemble(base.EpistemicModule):
  """Naive discrete ensemble of Haiku modules.

  This implementation assumes *one* integer index per batch, and forwards the
  entire batch with that index.
  """

  def __init__(self, ensemble: Sequence[hk.Module]):
    super().__init__(name='ensemble')
    self.ensemble = list(ensemble)
    self.num_ensemble = len(ensemble)

  def __call__(self, inputs: base.Array, index: base.Index) -> base.Array:
    """Index must be a single integer indicating which net to forward."""
    # TODO(author2): during init ensure all module parameters are created.
    _ = [model(inputs) for model in self.ensemble]  # pytype:disable=not-callable
    return hk.switch(index, self.ensemble, inputs)


class EinsumEnsembleEnn(base.EpistemicNetwork):
  """Ensemble ENN that uses a dot product in param space.

  Repeats parameters by an additional *ensemble* dimension in axis=0.
  Applying the parameters selects one single component of parameters per index.
  """

  def __init__(self,
               model: hk.Transformed,
               num_ensemble: int):
    self.model = model
    self.num_ensemble = num_ensemble

    def init_fn(key: chex.PRNGKey, inputs: chex.Array, index: int):
      del index  # Unused
      batched_init = jax.vmap(model.init, in_axes=[0, None], out_axes=0)
      return batched_init(jax.random.split(key, num_ensemble), inputs)

    def apply_fn(params: hk.Params, inputs: chex.Array, index: int):
      one_hot_index = jax.nn.one_hot(index, num_ensemble)
      param_selector = lambda p: jnp.einsum('i...,i->...', p, one_hot_index)
      sub_params = jax.tree_map(param_selector, params)
      return model.apply(sub_params, inputs)

    indexer = indexers.EnsembleIndexer(num_ensemble)

    super().__init__(apply_fn, init_fn, indexer)


class MLPEnsembleEnn(base.EpistemicNetwork):
  """An ensemble of MLP (with flatten) and without any prior."""

  def __init__(
      self,
      output_sizes: Sequence[int],
      num_ensemble: int,
      w_init: Optional[hk.initializers.Initializer] = None,
      b_init: Optional[hk.initializers.Initializer] = None
  ):
    """An ensemble of MLP (with flatten) and without any prior."""
    def flat_mlp():
      return hk.Sequential([
          hk.Flatten(),
          hk.nets.MLP(
              output_sizes,
              w_init=w_init,
              b_init=b_init)
      ])

    enn = utils.epistemic_network_from_module(
        enn_ctor=lambda: Ensemble([flat_mlp() for _ in range(num_ensemble)]),
        indexer=indexers.EnsembleIndexer(num_ensemble),
    )
    super().__init__(enn.apply, enn.init, enn.indexer)


def make_mlp_ensemble_prior_fns(
    output_sizes: Sequence[int],
    dummy_input: base.Array,
    num_ensemble: int,
    seed: int = 0,
    w_init: Optional[hk.initializers.Initializer] = None,
    b_init: Optional[hk.initializers.Initializer] = None
) -> Sequence[Callable[[base.Array], base.Array]]:
  """Factory method for creating ensemble of prior functions."""
  rng = hk.PRNGSequence(seed)
  def net_fn(x):
    layers = [
        hk.Flatten(),
        hk.nets.MLP(output_sizes, w_init=w_init, b_init=b_init)
    ]
    return hk.Sequential(layers)(x)

  transformed = hk.without_apply_rng(hk.transform(net_fn))

  prior_fns = []
  for _ in range(num_ensemble):
    params = transformed.init(next(rng), dummy_input)
    prior_fns.append(lambda x, params=params: transformed.apply(params, x))
  return prior_fns


def wrap_sequence_as_prior(
    prior_fns: Sequence[Callable[[base.Array], base.Array]],
) -> Callable[[base.Array, base.Index], base.Array]:
  return lambda x, z: jax.lax.switch(z, prior_fns, x)


def make_random_gp_ensemble_prior_fns(
    input_dim: int,
    output_dim: int,
    num_feat: int,
    gamma: priors.GpGamma,
    num_ensemble: int,
    seed: int = 0,
) -> Sequence[Callable[[base.Array], base.Array]]:
  """Factory method for creating an ensemble of random GPs."""
  rng = hk.PRNGSequence(seed)
  prior_fns = []
  for _ in range(num_ensemble):
    prior_fns.append(priors.make_random_feat_gp(
        input_dim, output_dim, num_feat, next(rng), gamma, scale=1.))
  return prior_fns


class MLPEnsembleMatchedPrior(base.EpistemicNetwork):
  """Ensemble of MLPs with matched prior functions."""

  def __init__(
      self,
      output_sizes: Sequence[int],
      dummy_input: base.Array,
      num_ensemble: int,
      prior_scale: float = 1.,
      seed: int = 0,
      w_init: Optional[hk.initializers.Initializer] = None,
      b_init: Optional[hk.initializers.Initializer] = None
  ):
    """Ensemble of MLPs with matched prior functions."""
    mlp_priors = make_mlp_ensemble_prior_fns(
        output_sizes, dummy_input, num_ensemble, seed)
    enn = priors.EnnWithAdditivePrior(
        enn=MLPEnsembleEnn(
            output_sizes, num_ensemble, w_init=w_init, b_init=b_init),
        prior_fn=wrap_sequence_as_prior(mlp_priors),
        prior_scale=prior_scale,
    )
    super().__init__(enn.apply, enn.init, enn.indexer)


class MLPEnsembleGpPrior(base.EpistemicNetwork):
  """Ensemble of MLPs with GP random kitchen sink prior functions."""

  def __init__(self,
               output_sizes: Sequence[int],
               input_dim: int,
               num_ensemble: int,
               num_feat: int,
               gamma: priors.GpGamma = 1.,
               prior_scale: float = 1,
               seed: int = 0):
    """Ensemble of MLPs with GP random kitchen sink prior functions."""
    gp_priors = make_random_gp_ensemble_prior_fns(
        input_dim, output_sizes[-1], num_feat, gamma, num_ensemble, seed)
    enn = priors.EnnWithAdditivePrior(
        enn=MLPEnsembleEnn(output_sizes, num_ensemble),
        prior_fn=wrap_sequence_as_prior(gp_priors),
        prior_scale=prior_scale,
    )
    super().__init__(enn.apply, enn.init, enn.indexer)


class MLPEnsembleArbitraryPrior(base.EpistemicNetwork):
  """Ensemble of MLPs with arbitrary prior functions."""

  def __init__(
      self,
      prior_fns: Sequence[Callable[[base.Array], base.Array]],
      output_sizes: Sequence[int],
      num_ensemble: int,
      prior_scale: float = 1.,
      w_init: Optional[hk.initializers.Initializer] = None,
      b_init: Optional[hk.initializers.Initializer] = None
  ):
    """Ensemble of MLPs with arbitrary prior functions."""
    enn = priors.EnnWithAdditivePrior(
        enn=MLPEnsembleEnn(
            output_sizes, num_ensemble, w_init=w_init, b_init=b_init),
        prior_fn=wrap_sequence_as_prior(prior_fns),
        prior_scale=prior_scale,
    )
    super().__init__(enn.apply, enn.init, enn.indexer)
