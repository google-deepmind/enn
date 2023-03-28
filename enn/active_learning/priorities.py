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
"""Functions to prioritize batches of data based on ENN forward pass."""

import typing as tp

import chex
from enn import datasets
from enn import networks
from enn.active_learning import base
import haiku as hk
import jax
import jax.numpy as jnp


def make_priority_fn_ctor(
    per_example_priority: base.PerExamplePriority,) -> base.PriorityFnCtor:
  """Makes a priority function constructor from a per example priority."""

  def make_priority_fn(
      enn_batch_fwd: networks.EnnBatchFwd[datasets.ArrayBatch],
  ) -> base.PriorityFn:
    """Makes a priority function."""
    def priority_fn(
        params: hk.Params,
        state: hk.State,
        batch: datasets.ArrayBatch,
        key: chex.PRNGKey,
    ) -> base.PriorityOutput:
      logits = enn_batch_fwd(params, state, batch.x)  # pytype: disable=wrong-arg-types  # numpy-scalars
      # Make sure labels have shape [num_data, 1] as expected by priority.
      labels = batch.y
      if labels.ndim == 1:
        labels = jnp.expand_dims(labels, axis=1)
      values = per_example_priority(logits, labels, key)
      return values, {}
    return priority_fn

  return make_priority_fn  # pytype: disable=bad-return-type  # numpy-scalars


def uniform_per_example(
    logits: chex.Array,
    labels: chex.Array,
    key: chex.PRNGKey,
) -> chex.Array:
  """Returns uniformly random scores per example."""
  del logits
  labels = jnp.squeeze(labels)
  return jax.random.uniform(key, shape=labels.shape)


def variance_per_example(
    logits: chex.Array,
    labels: chex.Array,
    key: chex.PRNGKey,
) -> chex.Array:
  """Calculates variance per example."""
  del labels, key
  unused_enn_samples, data_size, unused_num_classes = logits.shape
  probs = jax.nn.softmax(logits)
  variances = jnp.sum(jnp.var(probs, axis=0), axis=-1)
  chex.assert_shape(variances, (data_size,))
  return variances


def nll_per_example(
    logits: chex.Array,
    labels: chex.Array,
    key: chex.PRNGKey,
) -> chex.Array:
  """Calculates negative log-likelihood (nll) per example."""
  del key
  unused_enn_samples, data_size, unused_num_classes = logits.shape
  sample_probs = jax.nn.softmax(logits)
  probs = jnp.mean(sample_probs, axis=0)

  # Penalize with log loss
  labels = labels.astype(jnp.int32)
  labels = jnp.squeeze(labels)
  true_probs = probs[jnp.arange(data_size), labels]
  losses = -jnp.log(true_probs)
  chex.assert_shape(losses, (data_size,))
  return losses


def joint_nll_per_example(
    logits: chex.Array,
    labels: chex.Array,
    key: chex.PRNGKey,
) -> chex.Array:
  """Calculates joint negative log-likelihood (nll) per example."""
  del key
  num_enn_samples, data_size, unused_num_classes = logits.shape
  sample_probs = jax.nn.softmax(logits)

  # Penalize with log loss
  labels = labels.astype(jnp.int32)
  labels = jnp.squeeze(labels)
  true_probs = sample_probs[:, jnp.arange(data_size), labels]
  tau = 10
  repeated_lls = tau * jnp.log(true_probs)
  chex.assert_shape(repeated_lls, (num_enn_samples, data_size))

  # Take average of joint lls over num_enn_samples
  joint_lls = jnp.mean(repeated_lls, axis=0)
  chex.assert_shape(joint_lls, (data_size,))

  return -1 * joint_lls


def entropy_per_example(
    logits: chex.Array,
    labels: chex.Array,
    key: chex.PRNGKey,
) -> chex.Array:
  """Calculates entropy per example."""
  del labels, key
  unused_enn_samples, data_size, num_classes = logits.shape
  sample_probs = jax.nn.softmax(logits)
  probs = jnp.mean(sample_probs, axis=0)
  chex.assert_shape(probs, (data_size, num_classes))
  entropies = -1 * jnp.sum(probs * jnp.log(probs), axis=1)
  chex.assert_shape(entropies, (data_size,))

  return entropies


def margin_per_example(
    logits: chex.Array,
    labels: chex.Array,
    key: chex.PRNGKey,
) -> chex.Array:
  """Calculates margin between top and second probabilities per example."""
  # See e.g. use in PLEX paper: https://arxiv.org/abs/2207.07411
  del labels, key
  unused_enn_samples, data_size, num_classes = logits.shape
  sample_probs = jax.nn.softmax(logits)
  probs = jnp.mean(sample_probs, axis=0)
  chex.assert_shape(probs, (data_size, num_classes))
  sorted_probs = jnp.sort(probs)
  margins = sorted_probs[:, -1] - sorted_probs[:, -2]
  chex.assert_shape(margins, (data_size,))
  # Return the *negative* margin
  return -margins


def bald_per_example(
    logits: chex.Array,
    labels: chex.Array,
    key: chex.PRNGKey,
) -> chex.Array:
  """Calculates BALD mutual information per example."""
  del labels, key
  num_enn_samples, data_size, num_classes = logits.shape
  sample_probs = jax.nn.softmax(logits)
  # Function to compute entropy
  compute_entropy = lambda p: -1 * jnp.sum(p * jnp.log(p), axis=1)

  # Compute entropy for average probabilities
  mean_probs = jnp.mean(sample_probs, axis=0)
  chex.assert_shape(mean_probs, (data_size, num_classes))
  mean_entropy = compute_entropy(mean_probs)
  chex.assert_shape(mean_entropy, (data_size,))

  # Compute entropy for each sample probabilities
  sample_entropies = jax.vmap(compute_entropy)(sample_probs)
  chex.assert_shape(sample_entropies, (num_enn_samples, data_size))

  models_disagreement = mean_entropy - jnp.mean(sample_entropies, axis=0)
  chex.assert_shape(models_disagreement, (data_size,))
  return models_disagreement


def var_ratios_per_example(
    logits: chex.Array,
    labels: chex.Array,
    key: chex.PRNGKey,
) -> chex.Array:
  """Calculates the highest probability per example."""
  del labels, key
  unused_enn_samples, data_size, num_classes = logits.shape
  sample_probs = jax.nn.softmax(logits)
  probs = jnp.mean(sample_probs, axis=0)
  chex.assert_shape(probs, (data_size, num_classes))
  max_probs = jnp.max(probs, axis=1)
  variation_ratio = 1 - max_probs
  assert len(variation_ratio) == data_size

  return variation_ratio


def make_ucb_per_example(
    ucb_factor: float = 1.,
    class_values: tp.Optional[chex.Array] = None,
) -> base.PerExamplePriority:
  """Creates a UCB-style priority metric."""

  def compute_ucb(
      logits: chex.Array,
      labels: chex.Array,
      key: chex.PRNGKey,
  ) -> chex.Array:
    del labels, key
    unused_enn_samples, data_size, num_classes = logits.shape

    # Either use class values or default to just the first class
    scale_values = class_values
    if scale_values is None:
      scale_values = jnp.zeros(num_classes).at[0].set(1)

    probs = jax.nn.softmax(logits)
    value = jnp.einsum('zbc,c->zb', probs, scale_values)
    mean_values = jnp.mean(value, axis=0)
    std_values = jnp.std(value, axis=0)
    ucb_value = mean_values + ucb_factor * std_values
    chex.assert_shape(ucb_value, (data_size,))
    return ucb_value

  return compute_ucb


def make_scaled_mean_per_example(
    class_values: tp.Optional[chex.Array] = None,
) -> base.PerExamplePriority:
  """Creates a priority metric based on mean probs scaled by class_values."""

  def compute_scaled_mean(
      logits: chex.Array,
      labels: chex.Array,
      key: chex.PRNGKey,
  ) -> chex.Array:
    del labels, key
    unused_enn_samples, data_size, num_classes = logits.shape

    # Either use class values or default to just the first class
    scale_values = class_values
    if scale_values is None:
      scale_values = jnp.zeros(num_classes).at[0].set(1)

    probs = jax.nn.softmax(logits)
    values = jnp.einsum('zbc,c->zb', probs, scale_values)
    mean_values = jnp.mean(values, axis=0)
    chex.assert_shape(mean_values, (data_size,))
    return mean_values

  return compute_scaled_mean


def make_scaled_std_per_example(
    class_values: tp.Optional[chex.Array] = None,
) -> base.PerExamplePriority:
  """Creates a priority metric based on std of probs scaled by class_values."""

  def compute_scaled_std(
      logits: chex.Array,
      labels: chex.Array,
      key: chex.PRNGKey,
  ) -> chex.Array:
    del labels, key
    unused_enn_samples, data_size, num_classes = logits.shape

    # Either use class values or default to just the first class
    scale_values = class_values
    if scale_values is None:
      scale_values = jnp.zeros(num_classes).at[0].set(1)

    probs = jax.nn.softmax(logits)
    values = jnp.einsum('zbc,c->zb', probs, scale_values)
    std_values = jnp.std(values, axis=0)
    chex.assert_shape(std_values, (data_size,))
    return std_values

  return compute_scaled_std


_PerExamplePriorities = {
    'uniform': uniform_per_example,
    'variance': variance_per_example,
    'nll': nll_per_example,
    'joint_nll': joint_nll_per_example,
    'entropy': entropy_per_example,
    'margin': margin_per_example,
    'bald': bald_per_example,
    'var_ratios': var_ratios_per_example,
    'ucb': make_ucb_per_example(),
    'scaled_mean': make_scaled_mean_per_example(),
    'scaled_std': make_scaled_std_per_example(),
}


_PriorityFnCtors = {
    key: make_priority_fn_ctor(value)
    for key, value in _PerExamplePriorities.items()
}


def get_implemented_priority_fn_ctors() -> tp.Sequence[str]:
  """Returns the list of all supported priority function constructors."""
  return list(_PriorityFnCtors.keys())


def get_priority_fn_ctor(name: str) -> base.PriorityFnCtor:
  """Returns a priority function constructor for the priority specified by `name`."""
  assert name in get_implemented_priority_fn_ctors()
  return _PriorityFnCtors[name]


def get_implemented_per_example_priorities() -> tp.Sequence[str]:
  """Returns the list of all supported per example priority functions."""
  return list(_PerExamplePriorities.keys())


def get_per_example_priority(name: str) -> base.PerExamplePriority:
  """Returns a per example priority function for the priority specified by `name`."""
  assert name in get_implemented_per_example_priorities()
  return _PerExamplePriorities[name]
