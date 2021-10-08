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

import functools
from typing import Callable, Optional, Sequence, Tuple

from acme.utils import loggers
from enn import base
from enn import supervised
from enn import utils
from enn.extra import vae_losses
import haiku as hk
import jax
from jax import numpy as jnp
import optax


ActivationFn = Callable[[base.Array], base.Array]


# TODO(smaghari): Make this trainer use the enn.supervised.sgd_experiment
class VaeTrainer:
  """Trainer class for training a VAE on the provided data."""

  def __init__(self,
               data: base.Batch,
               hidden_sizes: Sequence[int] = (64, 32),
               latent_size: int = 10,
               activation: ActivationFn = jax.nn.tanh,
               bernoulli_decoder: bool = True,
               batch_size: int = 1000,
               learning_rate: float = 1e-3,
               logger: Optional[loggers.Logger] = None,
               train_log_freq: int = 1,
               seed: int = 0):
    """Initializes the Trainer for VAE."""
    self.rng = hk.PRNGSequence(seed)

    # Create an iterator over data
    self.dataset = utils.make_batch_iterator(data, batch_size)

    # Define the vae network
    input_size = data.x.shape[-1]
    def vae_net(x: base.Array) -> vae_losses.VaeOutput:
      return VaeNet(
          input_size=input_size,
          hidden_sizes=hidden_sizes,
          latent_size=latent_size,
          activation=activation,
          bernoulli_decoder=bernoulli_decoder)(x)

    self._net = hk.transform(vae_net)

    # Initialize vae network
    batch = next(self.dataset)
    params = self._net.init(next(self.rng), batch.x)
    self._optimizer = optax.adam(learning_rate)
    opt_state = self._optimizer.init(params)
    self.state = supervised.TrainingState(params, opt_state)
    self.step = 0
    self.logger = logger or loggers.make_default_logger(
        'vae_tarining', time_delta=0)
    self._train_log_freq = train_log_freq

    # Forward vae network at random key
    def forward(
        params: hk.Params, inputs: base.Array, key: base.RngKey) -> base.Array:
      return self._net.apply(params, inputs, key)
    self._forward = jax.jit(forward)

    # Internalize vae loss_fn
    loss_fn = vae_losses.VaeLoss(bernoulli_decoder)
    self._loss_fn = jax.jit(functools.partial(loss_fn, self._net.apply))

    # Define the SGD step on the loss
    def sgd_step(
        state: supervised.TrainingState,
        x: base.Array,
        key: base.RngKey,
    ) -> Tuple[supervised.TrainingState, base.LossMetrics]:
      # Calculate the loss, metrics and gradients
      (loss, metrics), grads = jax.value_and_grad(
          self._loss_fn, has_aux=True)(state.params, x, key)
      metrics.update({'loss': loss})
      updates, new_opt_state = self._optimizer.update(grads, state.opt_state)
      new_params = optax.apply_updates(state.params, updates)
      new_state = supervised.TrainingState(
          params=new_params,
          opt_state=new_opt_state,
      )
      return new_state, metrics

    self._sgd_step = jax.jit(sgd_step)

  def train(self, num_batches: int):
    """Train the VAE on dataset for num_batches."""
    for _ in range(num_batches):
      self.step += 1
      self.state, loss_metrics = self._sgd_step(
          self.state, next(self.dataset).x, next(self.rng))

      # Periodically log this performance
      if self.step % self._train_log_freq == 0:
        loss_metrics.update(
            {'step': self.step})
        self.logger.write(loss_metrics)

  def predict(self, x: base.Array) -> base.Array:
    """Generate output based on test input.

    Args:
      x: A batch of inputs.

    Returns:
      Generated output.
    """
    logits, _, _, _ = self._forward(self.state.params, next(self.rng), x)
    return logits

  def loss(self, x: base.Array, key: base.RngKey) -> base.Array:
    """Evaluate the loss for one batch of data."""
    return self._loss_fn(self.state.params, x, key)

  def get_encoder_fn(
      self) -> Callable[[base.Array], Tuple[base.Array, base.Array]]:
    """Returns the encoder function of the VAE."""

    def encoder(x: base.Array) -> Tuple[base.Array, base.Array]:
      """Forwards input x through the encoder."""
      # key does not affect mean, log_variance (outputs of encoder part).
      # They affect the outputs of decoder which we don't use here.
      key = jax.random.PRNGKey(0)
      _, _, mean, log_variance = self._forward(self.state.params, key, x)
      return mean, log_variance

    return encoder


class VaeNet(hk.Module):
  """Implementation of a VAE based on (Kingma & Welling, 2014)."""

  def __init__(self,
               input_size: int,
               hidden_sizes: Sequence[int] = (64, 32),
               latent_size: int = 10,
               activation: ActivationFn = jax.nn.tanh,
               bernoulli_decoder: bool = True):
    """Creates the VAE architecture.

    Args:
      input_size: Size of the input 1D array to the network.
      hidden_sizes: Size of the hidden layers.
      latent_size: Number of latent variables.
      activation: An activation function.
      bernoulli_decoder: A boolean specifying whether the decoder is Bernoulli.
        If it is False, the the decoder is considered to be Gaussian.
    """
    super().__init__(name='VaeNet')
    self._input_size = input_size
    self._hidden_sizes = hidden_sizes
    self._latent_size = latent_size
    self._activation = activation
    self._bernoulli_decoder = bernoulli_decoder

  def __call__(self, x: base.Array) -> vae_losses.VaeOutput:
    """Implements the network based on the specified architecture.

    Args:
      x: Input to the network.

    Returns:
      A Tuple including the outputs of the decoder, mean and stddev of the
      latent variable.
    """
    # Encoder
    mean, log_variance = self.encoder(x)

    # Reparameterization trick
    # Note that since this function is transformed by hk.transform(), we get a
    # unique key by calling hk.next_rng_key().
    epsilon = jax.random.normal(hk.next_rng_key(), mean.shape)
    stddev = jnp.exp(0.5 * log_variance)
    z = mean + stddev * epsilon

    # Decoder
    out_mean, out_log_variance = self.decoder(z)

    return out_mean, out_log_variance, mean, log_variance

  def encoder(self, x: base.Array) -> Tuple[base.Array, base.Array]:
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

  def decoder(self, x: base.Array) -> Tuple[base.Array, base.Array]:
    """Implements the decoder for VAE.

    Generates the output(s) of the decoder. If the decoder is Gaussian, the
    outputs are mean and log_variance of the Gaussian distribution. If the
    decoder is Bernoulli, there is only one output which is the logits, but for
    consistency with the case of Gaussian decoder which returns two outputs, we
    return a zero output as the second output of Bernoulli decoder. Note that
    this output is ignored by the forward funciton of VAE.

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
