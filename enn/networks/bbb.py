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

"""Implementing Bayes-by-backprop (BBB) in JAX."""

from typing import Callable, Optional, Sequence

import chex
from enn import base
from enn import utils
from enn.networks import hypermodels
from enn.networks import indexers
import haiku as hk
import jax
import numpy as np


# TODO(author2): Current implementation will produce a bbb based on a linear
# diagonal hypermodel that can *only* work for a single index at a time.
# However, note that jax.vmap means this can easily be converted into a form
# that works with batched index.


def make_bbb_enn(base_output_sizes: Sequence[int],
                 dummy_input: chex.Array,
                 sigma_0: float,
                 scale: bool = True):
  """Makes a Bayes-by-backprop (BBB) aganet."""

  def make_transformed_base(output_sizes: Sequence[int]) -> hk.Transformed:
    """Factory method for creating base net function."""
    def net_fn(x):
      return hk.Sequential([hk.Flatten(), hk.nets.MLP(output_sizes)])(x)

    transformed = hk.without_apply_rng(hk.transform(net_fn))
    return transformed

  transformed_base = make_transformed_base(base_output_sizes)

  # VI loss computed by vi_losses.get_linear_hypermodel_elbo_fn assumes the
  # index to be Gaussian with the same variance as the latent prior variance.
  def indexer_ctor(index_dim):
    return indexers.ScaledGaussianIndexer(index_dim, index_scale=sigma_0)
  enn = DiagonalLinearHypermodel(
      transformed_base=transformed_base,
      dummy_input=dummy_input,
      indexer_ctor=indexer_ctor,
      return_generated_params=True,
      scale=scale,
  )

  return enn


class DiagonalLinearHypermodel(base.EpistemicNetwork):
  """Diagonal linear hypermodel for transformed_base as EpistemicNetwork."""

  def __init__(
      self,
      transformed_base: hk.Transformed,
      dummy_input: base.Array,
      indexer_ctor: Callable[[int], base.EpistemicIndexer],
      return_generated_params: bool = False,
      scale: bool = True,
      w_init: Optional[hk.initializers.Initializer] = None,
      b_init: Optional[hk.initializers.Initializer] = None,
      ):

    base_params = transformed_base.init(jax.random.PRNGKey(0), dummy_input)
    num_base_params = np.sum(
        jax.tree_leaves(jax.tree_map(np.size, base_params)))
    indexer = indexer_ctor(num_base_params)

    enn = utils.epistemic_network_from_module(
        enn_ctor=hypermodels.hypermodel_module(
            transformed_base,
            dummy_input,
            hyper_torso=lambda x: x,
            diagonal_linear_hyper=True,
            return_generated_params=return_generated_params,
            scale=scale),
        indexer=indexer,
    )
    super().__init__(enn.apply, enn.init, enn.indexer)
