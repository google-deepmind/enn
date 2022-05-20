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

"""Implementing some mechanisms for prior functions with ENNs in JAX."""
import dataclasses
from typing import Callable, Iterable, Optional, Tuple, Union

from absl import logging
import chex
from enn import base
from enn import utils
import haiku as hk
import jax
import jax.numpy as jnp

PriorFn = Callable[[base.Array, base.Index], base.Array]


class EnnWithAdditivePrior(base.EpistemicNetwork):
  """Create an ENN with additive prior_fn applied to outputs."""

  def __init__(self,
               enn: base.EpistemicNetwork,
               prior_fn: PriorFn,
               prior_scale: float = 1.):
    """Create an ENN with additive prior_fn applied to outputs."""
    enn_state = utils.wrap_enn_as_enn_with_state(enn)
    enn_state_p = EnnStateWithAdditivePrior(enn_state, prior_fn, prior_scale)
    enn_prior = utils.wrap_enn_with_state_as_enn(enn_state_p)
    super().__init__(enn_prior.apply, enn_prior.init, enn_prior.indexer)


class EnnStateWithAdditivePrior(base.EpistemicNetworkWithState):
  """Create an ENN with additive prior_fn applied to outputs."""

  def __init__(self,
               enn: base.EpistemicNetworkWithState,
               prior_fn: PriorFn,
               prior_scale: float = 1.):
    """Create an ENN with additive prior_fn applied to outputs."""
    def apply_fn(params: hk.Params,
                 state: hk.State,
                 inputs: base.Array,
                 index: base.Index) -> Tuple[base.OutputWithPrior, hk.State]:
      net_out, state_out = enn.apply(params, state, inputs, index)
      prior = prior_scale * prior_fn(inputs, index)
      if isinstance(net_out, base.OutputWithPrior):
        net_out: base.OutputWithPrior = net_out
        out = net_out._replace(prior=net_out.prior + prior)
      else:
        out = base.OutputWithPrior(train=net_out, prior=prior)
      return out, state_out
    super().__init__(
        apply=apply_fn,
        init=enn.init,
        indexer=enn.indexer,
    )


def convert_enn_to_prior_fn(enn: base.EpistemicNetwork,
                            dummy_input: base.Array,
                            key: base.RngKey) -> PriorFn:
  index_key, init_key, _ = jax.random.split(key, 3)
  index = enn.indexer(index_key)
  prior_params = enn.init(init_key, dummy_input, index)
  def prior_fn(x: base.Array, z: base.Index) -> base.Output:
    return enn.apply(prior_params, x, z)
  return prior_fn


def make_null_prior(output_dim: int) -> Callable[[base.Array], base.Array]:
  def null_prior(inputs: base.Array) -> base.Array:
    return jnp.zeros([inputs.shape[0], output_dim], dtype=jnp.float32)
  return null_prior

# Configure GP kernel via float=gamma or uniform between gamma_min, gamma_max.
GpGamma = Union[float, Tuple[float, float]]


def _parse_gamma(gamma: GpGamma,
                 num_feat: int,
                 key: base.RngKey) -> Union[float, base.Array]:
  if isinstance(gamma, float) or isinstance(gamma, int):
    return float(gamma)
  else:
    gamma_min, gamma_max = gamma
    return gamma_min + (gamma_max - gamma_min) * jax.random.uniform(
        key, shape=[1, num_feat, 1])


def make_random_feat_gp(
    input_dim: int,
    output_dim: int,
    num_feat: int,
    key: base.RngKey,
    gamma: GpGamma = 1.,
    scale: float = 1.,
) -> Callable[[base.Array], base.Array]:
  """Generate a random features GP realization via random features.

  This is based on the "random kitchen sink" approximation from Rahimi,Recht.
  See blog post/paper: http://www.argmin.net/2017/12/05/kitchen-sinks/.

  Args:
    input_dim: dimension of input.
    output_dim: dimension of output.
    num_feat: number of random features used to approximate GP.
    key: jax random number key.
    gamma: gaussian kernel variance = gamma^2 (higher = more wiggle).
      If you pass a tuple we generate uniform between gamma_min, gamma_max.
    scale: scale of the output in each dimension.

  Returns:
    A callable gp_instance: inputs -> outputs in jax.
  """
  weights_key, bias_key, alpha_key, gamma_key = jax.random.split(key, num=4)
  weights = jax.random.normal(
      weights_key, shape=[num_feat, input_dim, output_dim])
  bias = 2 * jnp.pi * jax.random.uniform(
      bias_key, shape=[1, num_feat, output_dim])
  alpha = jax.random.normal(alpha_key, shape=[num_feat]) / jnp.sqrt(num_feat)
  gamma = _parse_gamma(gamma, num_feat, gamma_key)

  def gp_instance(inputs: base.Array) -> base.Array:
    """Assumes one batch dimension and flattens input to match that."""
    flat_inputs = jax.vmap(jnp.ravel)(inputs)
    input_embedding = jnp.einsum('bi,kio->bko', flat_inputs, weights)
    random_feats = jnp.cos(gamma * input_embedding + bias)
    return scale * jnp.einsum('bko,k->bo', random_feats, alpha)

  return gp_instance


def get_random_mlp_with_index(
    x_sample: base.Array,
    z_sample: base.Array,
    rng: chex.PRNGKey,
    prior_output_sizes: Optional[Iterable[int]] = None,
    prior_weight_std: float = 3,
    prior_bias_std: float = 0.1
) -> PriorFn:
  """Construct a prior func f(x, z) based on a random MLP.

  The returned function assumes the data input, x, to include a batch dimension
  but the index input, z, to not include a batch dimension.

  Args:
    x_sample: a sample data input.
    z_sample: a sample index input.
    rng: PRNG key.
    prior_output_sizes: output sizes for the MLP.
    prior_weight_std: unscaled std of the random weights. The actual std is
      scaled by 1/sqrt{n} where n is the fan-in (truncated noraml at 2 sigma).
    prior_bias_std: std of the random biases (truncated noraml at 2 sigma).

  Returns:
    a random function of two inputs x, and z.
  """

  if prior_output_sizes is None:
    prior_output_sizes = [10, 10, 1]

  def net_fn(x: base.Array, z: base.Array):
    # repeat z along the batch dimension of x.
    z = jnp.tile(z, reps=(x.shape[0], 1))
    xz = jnp.concatenate([x, z], axis=1)
    mlp = hk.nets.MLP(
        prior_output_sizes,
        w_init=hk.initializers.VarianceScaling(scale=prior_weight_std),
        b_init=hk.initializers.TruncatedNormal(stddev=prior_bias_std))
    return mlp(xz)

  transformed_fn = hk.without_apply_rng(hk.transform(net_fn))
  params = transformed_fn.init(rng, x_sample, z_sample)
  return lambda x, z: transformed_fn.apply(params, x, z)


# TODO(author2): Forming Modules with Prior is prone to bugs as the "prior"
# parameters will get returned in the init. In general, you should try and
# use the EnnWithAdditivePrior above instead.
WARN = ('WARNING: prior parameters will be included as hk.Params for module.'
        'If possible, you should use EnnWithAdditivePrior instead.')


@dataclasses.dataclass
class NetworkWithAdditivePrior(hk.Module):
  """Combines network and a prior using a specified function."""
  net: hk.Module
  prior_net: hk.Module
  prior_scale: float = 1.

  def __call__(self, *args, **kwargs) -> base.Array:
    logging.warning(WARN)
    prior = jax.lax.stop_gradient(self.prior_net(*args, **kwargs))  # pytype:disable=not-callable
    net_out = self.net(*args, **kwargs)  # pytype:disable=not-callable
    return net_out + prior * self.prior_scale
