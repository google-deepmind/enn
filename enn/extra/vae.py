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

"""Jax implementation of Variational Autoencoder (VAE)."""

from typing import Callable, Sequence, Tuple

from enn import base
from enn import networks
import haiku as hk
import jax
from jax import numpy as jnp


ActivationFn = Callable[[base.Array], base.Array]


class MLPVae(base.EpistemicNetwork):
  """MLP VAE, based on (Kingma & Welling, 2014), as an ENN."""

  def __init__(self,
               input_size: int,
               hidden_sizes: Sequence[int] = (64, 32),
               latent_size: int = 10,
               activation: ActivationFn = jax.nn.tanh,
               bernoulli_decoder: bool = True):
    """Creates the VAE as an ENN.

    Args:
      input_size: Size of the input 1D array to the network.
      hidden_sizes: Size of the hidden layers.
      latent_size: Number of latent variables.
      activation: An activation function.
      bernoulli_decoder: A boolean specifying whether the decoder is Bernoulli.
        If it is False, the the decoder is considered to be Gaussian.
    """
    self._input_size = input_size
    self._hidden_sizes = hidden_sizes
    self._latent_size = latent_size
    self._activation = activation
    self._bernoulli_decoder = bernoulli_decoder

    def encoder(x: base.Array) -> Tuple[base.Array, base.Array]:
      """Implements the encoder for VAE.

      Args:
        x: Input to the encoder.

      Returns:
        A Tuple including the mean and log of variance of the latent variable.
      """
      x = hk.Flatten()(x)
      for hidden_size in self._hidden_sizes:
        x = hk.Linear(hidden_size)(x)
        x = self._activation(x)

      mean = hk.Linear(self._latent_size)(x)
      log_variance = hk.Linear(self._latent_size)(x)

      return mean, log_variance

    def decoder(x: base.Array) -> Tuple[base.Array, base.Array]:
      """Implements the decoder for VAE.

      Generates the output(s) of the decoder. If the decoder is Gaussian, the
      outputs are mean and log_variance of the Gaussian distribution. If the
      decoder is Bernoulli, there is only one output which is the logits, but
      for consistency with the case of Gaussian decoder which returns two
      outputs, we return zero output as the second output of Bernoulli decoder.
      Note that this output is ignored by the forward funciton of VAE.

      Args:
        x: Input to the decoder.

      Returns:
        The generated outputs.
      """
      for hidden_size in self._hidden_sizes[::-1]:
        x = hk.Linear(hidden_size)(x)
        x = self._activation(x)

      if self._bernoulli_decoder:
        y = hk.Linear(self._input_size)(x)
        return y, jnp.zeros_like(y)
      else:
        mean = hk.Linear(self._input_size)(x)
        log_variance = hk.Linear(self._input_size)(x)
        return mean, log_variance

    def enn_fn(inputs: base.Array, z: base.Index) -> base.Output:
      # Encoder
      latent_mean, latent_log_variance = encoder(inputs)

      # Reparameterization trick
      # Note that z is basically a random key.
      epsilon = jax.random.normal(z, latent_mean.shape)
      latent_stddev = jnp.exp(0.5 * latent_log_variance)
      latent = latent_mean + latent_stddev * epsilon

      # Decoder
      out_mean, out_log_variance = decoder(latent)
      vae_outputs = {'latent_mean': latent_mean,
                     'latent_log_variance': latent_log_variance,
                     'out_mean': out_mean,
                     'out_log_variance': out_log_variance}

      return base.OutputWithPrior(train=out_mean, extra=vae_outputs)

    # Note that our enn_fn is stochastic because we generate a random number in
    # it. But, since we pass a base.RngKey directly to it, we can still wrap
    # transformed function with hk.without_apply_rng.
    transformed = hk.without_apply_rng(hk.transform(enn_fn))

    # We use a simple indexer which is basically an identity map.
    indexer = networks.PrngIndexer()

    # Apply function for enn_fn requires a rng key to generate random N(0, 1)
    # sample. We use the index z in f(x,z) as the rng key.
    def apply(params: hk.Params,
              x: base.Array,
              z: base.Index) -> base.Output:
      net_out = transformed.apply(params, x, z)
      return net_out

    super().__init__(apply, transformed.init, indexer)
