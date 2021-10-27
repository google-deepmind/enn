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
import dataclasses
from typing import Callable, NamedTuple, Optional, Sequence, Tuple

import chex
from enn import base
from enn import losses
from enn import networks
from enn import supervised
from enn import utils
import haiku as hk
import jax
from jax import numpy as jnp

import optax

# TODO(vikranthd): Work in Progress. Finalize the file.


class MeanLogVariance(NamedTuple):
  mean: chex.Array
  log_variance: chex.Array


EncoderFn = Callable[[chex.Array], MeanLogVariance]
DecoderFn = Callable[[chex.Array], MeanLogVariance]


@dataclasses.dataclass
class VAE:
  """Variational autoencoder."""
  encoder: EncoderFn
  decoder: DecoderFn
  latent_size: int
  enn: Optional[base.EpistemicNetwork] = None

  def __post_init__(self):
    # if enn is not specified create an enn from encoder, decoder
    if self.enn is None:
      self.enn = make_enn_from_encoder_decoder(self.encoder, self.decoder)


def make_enn_from_encoder_decoder(
    encoder: Callable[[chex.Array], MeanLogVariance],
    decoder: Callable[[chex.Array], MeanLogVariance]) -> base.EpistemicNetwork:
  """Outputs an enn taking encoder and decoder as inputs."""

  def enn_fn(x: chex.Array, key: chex.Array) -> base.Output:
    # Encoder
    latent_mean, latent_log_variance = encoder(x)

    # Generate a random vector based on encoder outputs
    z = jax.random.normal(key, latent_mean.shape)
    latent_stddev = jnp.exp(0.5 * latent_log_variance)
    latent = latent_mean + latent_stddev * z

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

  enn = base.EpistemicNetwork(apply=apply,
                              init=transformed.init,
                              indexer=indexer)
  return enn


def train_vae(vae: VAE,
              data_x: chex.Array,
              bernoulli_decoder: bool = True,
              num_batches: int = 10_000,
              batch_size: int = 1_000,
              learning_rate: float = 1e-3,
              train_log_freq: int = 1_000) -> Tuple[EncoderFn, DecoderFn]:
  """Given a vae and data, this function outputs trained encoder, decoder."""
  input_size = data_x.shape[1]
  latent_size = vae.latent_size

  dummy_y = jnp.zeros(shape=(data_x.shape[0],))
  dataset = utils.make_batch_iterator(
      base.Batch(data_x, dummy_y), batch_size=batch_size)

  # create loss function
  log_likelihood_fn = losses.get_log_likelihood_fn(bernoulli_decoder)
  latent_kl_fn = losses.get_latent_kl_fn()
  single_loss = losses.VaeLoss(log_likelihood_fn, latent_kl_fn)
  loss_fn = losses.average_single_index_loss(single_loss, num_index_samples=1)

  optimizer = optax.adam(learning_rate=learning_rate)
  experiment = supervised.Experiment(vae.enn, loss_fn, optimizer,
                                     dataset, train_log_freq=train_log_freq)
  # train the vae
  experiment.train(num_batches)

  # get the trained params
  enn_params = experiment.state.params

  # create the encoder function
  transformed_encoder = hk.without_apply_rng(hk.transform(vae.encoder))
  def encoder_fn(x: chex.Array) -> MeanLogVariance:
    latent_output = transformed_encoder.apply(enn_params, x)

    chex.assert_shape(latent_output.mean, (x.shape[0], latent_size))
    chex.assert_shape(latent_output.log_variance, latent_output.mean.shape)
    return latent_output

  # create a decoder function
  transformed_decoder = hk.without_apply_rng(hk.transform(vae.decoder))
  def decoder_fn(x: chex.Array) -> MeanLogVariance:
    decoder_output = transformed_decoder.apply(enn_params, x)

    chex.assert_shape(decoder_output.mean, (x.shape[0], input_size))
    chex.assert_shape(decoder_output.log_variance, decoder_output.mean.shape)
    return decoder_output

  return encoder_fn, decoder_fn


ActivationFn = Callable[[chex.Array], chex.Array]


@dataclasses.dataclass
class MLPVAEConfig:
  """All hyperparameters for an VAE with mlp encoder, decoder."""
  input_size: int
  hidden_sizes: Sequence[int] = (64, 32)
  latent_size: int = 2
  activation: ActivationFn = jax.nn.tanh
  bernoulli_decoder: bool = True
  num_batches: int = 10_000
  batch_size: int = 1_000
  learning_rate: float = 1e-3
  train_log_freq: int = 1_000


def get_mlp_vae_encoder_decoder(
    config: MLPVAEConfig,
    data_x: chex.Array) -> Tuple[EncoderFn, DecoderFn]:
  """Takes the config file, data and outputs a trained encoder, decoder pair."""

  chex.assert_scalar_positive(config.latent_size)

  def mlp_encoder(x: chex.Array) -> MeanLogVariance:
    """Encoder for VAE. Outputs mean and log_variance in latent space."""

    x = hk.Flatten()(x)
    chex.assert_equal(x.shape[1], config.input_size)

    for hidden_size in config.hidden_sizes:
      x = hk.Linear(hidden_size, name='encoder')(x)
      x = config.activation(x)

    mean = hk.Linear(config.latent_size, name='encoder_mean')(x)
    log_variance = hk.Linear(config.latent_size,
                             name='encoder_log_variance')(x)
    return MeanLogVariance(mean, log_variance)

  def mlp_decoder(x: chex.Array) -> MeanLogVariance:
    """Decoder for VAE. Outputs mean, log_var for an input in latent space."""

    chex.assert_equal(x.shape[1], config.latent_size)

    for hidden_size in config.hidden_sizes[::-1]:
      x = hk.Linear(hidden_size, name='decoder')(x)
      x = config.activation(x)

    if config.bernoulli_decoder:
      y = hk.Linear(config.input_size, name='decoder_mean')(x)
      return MeanLogVariance(y, jnp.zeros_like(y))
    else:
      mean = hk.Linear(config.input_size, name='decoder_out_mean')(x)
      log_variance = hk.Linear(config.input_size,
                               name='decoder_out_log_variance')(x)
      return MeanLogVariance(mean, log_variance)

  # create vae based on encoder and decoder
  vae_mlp = VAE(encoder=mlp_encoder,
                decoder=mlp_decoder,
                latent_size=config.latent_size)

  # train the vae and return trained encoder_fn, decoder tuple
  return train_vae(vae=vae_mlp,
                   data_x=data_x,
                   bernoulli_decoder=config.bernoulli_decoder,
                   num_batches=config.num_batches,
                   batch_size=config.batch_size,
                   learning_rate=config.learning_rate,
                   train_log_freq=config.train_log_freq)


