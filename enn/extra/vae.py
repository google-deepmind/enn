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
from typing import Callable, NamedTuple, Sequence

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


class MeanLogVariance(NamedTuple):
  mean: chex.Array  # Mean value output
  log_var: chex.Array  # Log of variance (same shape as mean).

PreTransformFn = Callable[[chex.Array], MeanLogVariance]
PostTransformFn = Callable[[chex.Array], MeanLogVariance]


class TrainedVAE(NamedTuple):
  encoder: PostTransformFn  # Maps inputs to mean, log_var in latent
  decoder: PostTransformFn  # Maps latent to mean, log_var in reconstruction


@dataclasses.dataclass
class MLPVAEConfig:
  """Configures training for an MLP VAE."""
  hidden_sizes: Sequence[int] = (256, 64)
  latent_dim: int = 2
  activation: Callable[[chex.Array], chex.Array] = jax.nn.tanh
  bernoulli_decoder: bool = True
  num_batches: int = 10_000
  batch_size: int = 1_000
  learning_rate: float = 1e-3


def get_mlp_vae_encoder_decoder(
    data_x: chex.Array,
    config: MLPVAEConfig = MLPVAEConfig(),
) -> TrainedVAE:
  """Trains an MLP VAE on given data according to config."""
  _, input_dim = data_x.shape

  def mlp_encoder(x: chex.Array) -> MeanLogVariance:
    """Encoder for VAE. Outputs mean and log_variance in latent space."""
    x = hk.Flatten()(x)
    for hidden_size in config.hidden_sizes:
      x = hk.Linear(hidden_size, name='encoder')(x)
      x = config.activation(x)

    mean = hk.Linear(config.latent_dim, name='encoder_mean')(x)
    log_var = hk.Linear(config.latent_dim, name='encoder_log_var')(x)
    return MeanLogVariance(mean, log_var)

  def mlp_decoder(x: chex.Array) -> MeanLogVariance:
    """Decoder for VAE. Outputs mean, log_var for an input in latent space."""
    for hidden_size in config.hidden_sizes[::-1]:
      x = hk.Linear(hidden_size, name='decoder')(x)
      x = config.activation(x)

    mean = hk.Linear(input_dim, name='decoder_mean')(x)
    if config.bernoulli_decoder:
      log_var = jnp.zeros_like(mean)
    else:
      log_var = hk.Linear(input_dim, name='decoder_log_var')(x)
    return MeanLogVariance(mean, log_var)

  # Train the VAE
  return train_vae(
      encoder=mlp_encoder,
      decoder=mlp_decoder,
      latent_dim=config.latent_dim,
      data_x=data_x,
      log_likelihood_fn=losses.get_log_likelihood_fn(config.bernoulli_decoder),
      optimizer=optax.adam(config.learning_rate),
      num_batches=config.num_batches,
      batch_size=config.batch_size,
  )


def make_vae_enn(encoder: PreTransformFn, decoder: PreTransformFn,
                 latent_dim: int) -> networks.EnnArray:
  """Factory method to create and transform ENN from encoder/decoder."""

  def net_fn(x: chex.Array, z: base.Index) -> networks.OutputWithPrior:
    # Encoder
    latent_mean, latent_log_var = encoder(x)
    chex.assert_shape([latent_mean, latent_log_var], [x.shape[0], latent_dim])

    # Generate a random vector based on encoder outputs
    latent_std = jnp.exp(0.5 * latent_log_var)
    latent = latent_mean + jnp.einsum('bi,i->bi', latent_std, z)

    # Decoder
    out_mean, out_log_var = decoder(latent)
    vae_outputs = {'latent_mean': latent_mean, 'latent_log_var': latent_log_var,
                   'out_mean': out_mean, 'out_log_var': out_log_var}
    return networks.OutputWithPrior(train=out_mean, extra=vae_outputs)

  transformed = hk.without_apply_rng(hk.transform_with_state(net_fn))
  indexer = networks.GaussianIndexer(latent_dim)
  return networks.EnnArray(transformed.apply, transformed.init, indexer)


def train_vae(encoder: PreTransformFn,
              decoder: PreTransformFn,
              latent_dim: int,
              data_x: chex.Array,
              log_likelihood_fn: losses.LogLikelihoodFn,
              optimizer: optax.GradientTransformation,
              num_batches: int = 10_000,
              batch_size: int = 1_000) -> TrainedVAE:
  """Given a vae and data, this function outputs trained encoder, decoder."""
  num_train, input_dim = data_x.shape
  dummy_y = jnp.zeros(shape=(num_train,))
  dataset = utils.make_batch_iterator(base.Batch(data_x, dummy_y), batch_size)

  # Create loss function
  single_loss = losses.VaeLoss(log_likelihood_fn, losses.latent_kl_fn)
  loss_fn = losses.average_single_index_loss(single_loss, num_index_samples=1)

  # Train VAE by gradient descent for num_batches and extract parameters.
  experiment = supervised.Experiment(
      enn=make_vae_enn(encoder, decoder, latent_dim),
      loss_fn=loss_fn,
      optimizer=optimizer,
      dataset=dataset,
      train_log_freq=max(int(num_batches / 100), 1),
  )
  experiment.train(num_batches)
  params = experiment.state.params

  # Form an encoder function from these parameters
  transformed_encoder = hk.without_apply_rng(hk.transform(encoder))
  def encoder_fn(x: chex.Array) -> MeanLogVariance:
    latent = transformed_encoder.apply(params, x)
    chex.assert_shape([latent.mean, latent.log_var], (x.shape[0], latent_dim))
    return latent

  # Form an encoder function from these parameters
  transformed_decoder = hk.without_apply_rng(hk.transform(decoder))
  def decoder_fn(x: chex.Array) -> MeanLogVariance:
    reconstruction = transformed_decoder.apply(params, x)
    chex.assert_shape([reconstruction.mean, reconstruction.log_var],
                      (x.shape[0], input_dim))
    return reconstruction

  return TrainedVAE(jax.jit(encoder_fn), jax.jit(decoder_fn))
