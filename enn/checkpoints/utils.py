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
"""Utility functions for loading from entries."""

import os
import tempfile
from typing import Callable, Tuple

import chex
import dill
from enn import base_legacy as enn_base
from enn import utils as enn_utils
from enn.checkpoints import base as checkpoint_base
from enn.checkpoints import epinet as checkpoint_epinet
import haiku as hk
import jax
import jax.numpy as jnp
import requests
from typing_extensions import Protocol


class EnnSampler(Protocol):

  def __call__(self, inputs: chex.Array, key: chex.PRNGKey) -> chex.Array:
    """Takes in a batch of inputs, a key and outputs a number of ENN samples.

    Args:
      inputs: of shape [batch_size, ...]
      key: random key

    Returns:
      Multiple ENN samples e.g. [num_enn_samples, batch_size, num_classes].
    """


def load_from_file(
    file_name: str,
    url_path: str = 'https://storage.googleapis.com/dm-enn',
) -> checkpoint_base.ParamsStateLoadFn:
  """Utility wrapper to create a load function from a file."""
  def load_fn() -> Tuple[hk.Params, hk.State]:
    return init_from_file(file_name, url_path)
  return load_fn


def init_from_file(
    file_name: str,
    url_path: str = 'https://storage.googleapis.com/dm-enn',
) -> Tuple[hk.Params, hk.State]:
  """Returns state and params from a file stored on `url_path`."""
  url = f'{url_path}/{file_name}.npzs'
  with tempfile.TemporaryDirectory() as tmpdir:
    response = requests.get(url, verify=False)

    # Make a temporary file for downloading
    filepath = os.path.join(tmpdir, f'/tmp/{file_name}.npzs')
    open(filepath, 'wb').write(response.content)

    # Read data from temporary file
    with open(filepath, 'rb') as f:
      data = dill.load(f)

  # Extracting params and state from data
  params, state = data['params'], data['state']

  # Map params and state from np.array to jnp.array
  state = jax.tree_map(jnp.array, state)
  params = jax.tree_map(jnp.array, params)

  return params, state


def average_logits(array: chex.Array) -> chex.Array:
  """Takes average of logits over num_enn_sample."""
  assert array.ndim == 3
  unused_num_enn_samples, batch_size, num_classes = array.shape

  # Convert logits to probabilities and take average
  probs = jnp.mean(jax.nn.softmax(array), axis=0)

  # Convert average probabilities back to logits
  logits = jnp.log(probs)
  chex.assert_shape(logits, (batch_size, num_classes))

  return logits


def make_enn_sampler_from_checkpoint(
    checkpoint: checkpoint_base.EnnCheckpoint,
    num_enn_samples: int,
    temperature_rescale: bool = False,
) -> EnnSampler:
  """Makes a sampler that samples multiple logits given inputs and key.

  Args:
    checkpoint: an ENN checkpoint.
    num_enn_samples: number of index samples for ENN.
    temperature_rescale: whether to apply the tuned evaluation temperature.

  Returns:
    Callable: inputs, key --> logits of shape [num_enn_samples, batch, class].
  """
  enn = checkpoint.enn_ctor()
  params, state = checkpoint.load_fn()
  if temperature_rescale and checkpoint.tuned_eval_temperature:
    temperature = checkpoint.tuned_eval_temperature
  else:
    temperature = 1.

  def sample_logits(inputs: chex.Array, key: chex.PRNGKey,) -> chex.Array:
    index_fwd = lambda z: enn.apply(params, state, inputs, z)
    indices = jax.vmap(enn.indexer)(jax.random.split(key, num_enn_samples))
    enn_out, _ = jax.lax.map(index_fwd, indices)
    logits = enn_utils.parse_net_output(enn_out)
    chex.assert_shape(logits, [num_enn_samples, None, None])
    return logits / temperature

  return jax.jit(sample_logits)


def load_checkpoint_as_logit_fn(
    checkpoint: checkpoint_base.EnnCheckpoint,
    num_enn_samples: int = 1,
    temperature_rescale: bool = False,
    seed: int = 0,
) -> Callable[[chex.Array], enn_base.OutputWithPrior]:
  """Loads an ENN as a simple forward function: images --> logits."""
  enn_sampler = make_enn_sampler_from_checkpoint(
      checkpoint, num_enn_samples, temperature_rescale)

  def forward_fn(inputs: chex.Array) -> chex.Array:
    logits = enn_sampler(inputs, jax.random.PRNGKey(seed))
    ave_logits = average_logits(logits)
    # Wrap the output with prior
    return enn_base.OutputWithPrior(ave_logits)

  return jax.jit(forward_fn)


################################################################################
# Optimized Epinet forward functions and samplers


def make_epinet_sampler_from_checkpoint(
    epinet_cpt: checkpoint_epinet.EpinetCheckpoint,
    num_enn_samples: int = 1000,
    temperature_rescale: bool = False,
) -> EnnSampler:
  """Forms a callable that samples multiple logits based on inputs and key."""
  base_enn = epinet_cpt.base_cpt.enn_ctor()

  if epinet_cpt.base_index is None:
    base_index = base_enn.indexer(jax.random.PRNGKey(0))
  else:
    base_index = epinet_cpt.base_index

  epinet = epinet_cpt.epinet_ctor()

  # Pull out the parameters
  base_params, base_state = epinet_cpt.base_cpt.load_fn()
  epi_params, epi_state = epinet_cpt.load_fn()

  if temperature_rescale and epinet_cpt.tuned_eval_temperature:
    temperature = epinet_cpt.tuned_eval_temperature
  else:
    temperature = 1.

  def sample_logits(inputs: chex.Array, key: chex.PRNGKey,) -> chex.Array:
    # Forward the base network once
    base_out, unused_base_state = base_enn.apply(
        base_params, base_state, inputs, base_index)
    hidden = epinet_cpt.parse_hidden(base_out)
    base_logits = enn_utils.parse_net_output(base_out) * epinet_cpt.base_scale

    # Forward the enn over all the different indices
    keys = jax.random.split(key, num_enn_samples)
    indices = jax.vmap(epinet.indexer)(keys)

    def index_fwd(index: enn_base.Index) -> enn_base.Array:
      return epinet.apply(epi_params, epi_state, inputs, index, hidden)

    enn_out, unused_epi_state = jax.lax.map(index_fwd, indices)
    enn_logits = enn_utils.parse_net_output(enn_out)

    # Combined logits
    combined_logits = jnp.expand_dims(base_logits, 0) + enn_logits
    chex.assert_equal_shape([combined_logits, enn_logits])
    return combined_logits / temperature

  return jax.jit(sample_logits)


def make_epinet_forward_fn(
    epinet_cpt: checkpoint_epinet.EpinetCheckpoint,
    num_enn_samples: int = 1000,
    temperature_rescale: bool = False,
    seed: int = 44,
) -> Callable[[chex.Array], chex.Array]:
  """Forms a callable that averages epinet over num_enn_samples indices."""
  epinet_sampler = make_epinet_sampler_from_checkpoint(
      epinet_cpt, num_enn_samples, temperature_rescale)
  key = jax.random.PRNGKey(seed)

  def forward_fn(inputs: chex.Array) -> chex.Array:
    logits_samples = epinet_sampler(inputs, key)
    return average_logits(logits_samples)

  return jax.jit(forward_fn)
