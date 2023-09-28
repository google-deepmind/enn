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
"""Metrics used to evaluate joint predictions."""

from typing import Tuple

import chex
from enn.metrics import base as metrics_base
from enn.metrics import marginal
import jax
import jax.numpy as jnp


def make_nll_polyadic_calculator(
    tau: int = 10,
    kappa: int = 2,
) -> metrics_base.MetricCalculator:
  """Returns a MetricCalculator that computes d_{KL}^{tau, kappa} metric."""

  def joint_ll_repeat(logits: chex.Array,
                      labels: chex.Array,
                      key: chex.PRNGKey) -> float:
    """Calculates joint NLL tau inputs resampled from anchor points."""
    # Shape checking
    chex.assert_shape(logits, [kappa, None])
    chex.assert_shape(labels, [kappa, 1])

    # Compute log-likehood at the kappa anchor points
    probs = jax.nn.softmax(logits)
    assigned_probs = probs[jnp.arange(kappa), jnp.squeeze(labels)]
    log_probs = jnp.log(assigned_probs)

    # Sample with replacement from the anchor points and sum for joint ll
    selected = jax.random.randint(key, shape=[tau], minval=0, maxval=kappa)
    return jnp.sum(log_probs[selected])  # pytype: disable=bad-return-type  # jnp-type

  def enn_nll(logits: chex.Array,
              labels: chex.Array,
              key: chex.Array) -> float:
    """Averages joint_ll_repeat over multiple ENN samples."""
    # Shape checking
    chex.assert_shape(logits, [None, kappa, None])
    chex.assert_shape(labels, [kappa, 1])

    # Averaging over ENN samples
    batched_ll = jax.vmap(joint_ll_repeat, in_axes=[0, None, None])
    lls = batched_ll(logits, labels, key)
    return -1 * metrics_base.average_sampled_log_likelihood(lls)  # pytype: disable=wrong-arg-types  # numpy-scalars

  def polyadic_nll(logits: chex.Array, labels: chex.Array) -> float:
    """Returns polyadic NLL based on repeated inputs.

    Internally this function works by taking the batch of logits and then
    "melting" it to add an extra dimension so that the batches we evaluate
    likelihood are of size=kappa. This means that one batch_size=N*kappa becomes
    N batches of size=kappa. For each of these batches of size kappa, we then
    resample tau observations replacement from these two anchor points. The
    function then returns the joint nll evaluated over this synthetic batch.

    Args:
      logits: [num_enn_samples, batch_size, num_classes]
      labels: [batch_size, 1]
    """
    # TODO(author2): Revisit metric/performance and sampling solution.
    # Shape checking
    chex.assert_rank(logits, 3)
    chex.assert_shape(labels, [logits.shape[1], 1])

    # We use the values of the sampled labels to specify a seed for the anchor
    # point resampling. This is not necessary if the evaluation batches are
    # sample i.i.d. but is a precaution against some other factor in sampling.
    offset = jnp.arange(labels.shape[0])[:, None] * jnp.max(labels) * 10
    seed = jnp.sum(labels * offset, dtype=jnp.int32)

    # Creating synthetic batches of size=kappa then use vmap.
    batched_logits, batched_labels = reshape_to_smaller_batches(
        logits, labels, batch_size=kappa)
    keys = jax.random.split(jax.random.PRNGKey(seed), batched_logits.shape[0])
    nlls = jax.vmap(enn_nll, in_axes=0)(batched_logits, batched_labels, keys)
    return jnp.mean(nlls)  # pytype: disable=bad-return-type  # jnp-type

  return jax.jit(polyadic_nll)


def make_nll_joint_calculator(tau: int = 10) -> metrics_base.MetricCalculator:
  """Returns a MetricCalculator that computes d_{KL}^{tau} metric."""

  def calculate_nll_joint(logits: chex.Array, labels: chex.Array) -> float:
    """Calculates joint nll."""
    num_data = labels.shape[0]
    assert num_data >= tau, f'num_data={num_data} should be at least tau!'

    batched_logits, batched_labels = reshape_to_smaller_batches(
        logits, labels, batch_size=tau)
    num_batches = batched_labels.shape[0]

    lls = jax.vmap(calculate_joint_ll)(
        batched_logits, batched_labels)
    chex.assert_shape(lls, (num_batches,))
    return -1 * jnp.mean(lls)  # pytype: disable=bad-return-type  # jnp-type

  return calculate_nll_joint


def calculate_joint_ll(logits: chex.Array, labels: chex.Array) -> float:
  """Computes joint log likelihood (ll) aggregated over enn samples.

  Depending on data batch_size (can be inferred from logits and labels), this
  function computes joint ll for tau=batch_size aggregated over enn samples. If
  num_data is one, this function computes marginal ll.

  Args:
    logits: [num_enn_sample, num_data, num_classes]
    labels: [num_data, 1]

  Returns:
    marginal log likelihood
  """
  num_enn_samples, tau, num_classes = logits.shape
  chex.assert_shape(labels, (tau, 1))

  class_probs = jax.nn.softmax(logits)
  chex.assert_shape(class_probs, (num_enn_samples, tau, num_classes))

  batched_ll = jax.vmap(marginal.categorical_log_likelihood, in_axes=[0, None])
  sampled_ll = batched_ll(class_probs, labels)
  return metrics_base.average_sampled_log_likelihood(sampled_ll)  # pytype: disable=wrong-arg-types  # numpy-scalars


def reshape_to_smaller_batches(
    logits: chex.Array,
    labels: chex.Array,
    batch_size: int,
) -> Tuple[chex.Array, chex.Array]:
  """Reshapes logits,labels to add leading batch_size dimension.

  In case the size of logits and labels are such that they cannot be equally
  divided into batches of size batch_size, extra data is discarded.

  Args:
    logits: has shape [num_enn_samples, num_data, num_classes]
    labels: has shape [num_data, 1]
    batch_size: desired output batch size.

  Returns:
    A tuple of batched_logits and batched_labels with shapes
      batched_logits: (num_batches, num_enn_samples, batch_size, num_classes)
      batched_labels: (num_batches, batch_size, 1)
  """
  # Shape checking
  assert logits.ndim == 3
  num_enn_samples, num_data, num_classes = logits.shape
  chex.assert_shape(labels, [num_data, 1])
  assert num_data >= batch_size

  ##############################################################################
  # 1. We split num_data to batches of size batch_size. To ensure that the split
  # is possible, we might need to discard extra data.
  num_batches = num_data // batch_size
  num_extra_data = num_data % batch_size
  num_data -= num_extra_data

  # 1.1. Discard extra data if needed.
  logits = logits[:, :num_data, :]
  labels = labels[:num_data, :]
  chex.assert_shape(logits, [num_enn_samples, num_data, num_classes])
  chex.assert_shape(labels, [num_data, 1])

  # 1.2. Split num_data to batches of size batch_size
  batched_logits = logits.reshape(
      [num_enn_samples, num_batches, batch_size, num_classes])
  batched_labels = labels.reshape([num_batches, batch_size, 1])

  ##############################################################################
  # 2. We want num_batches to be the leading axis. It is already the case for
  # batched_labels, but we need to change axes for batched_logits.
  batched_logits = batched_logits.swapaxes(0, 1)
  chex.assert_shape(batched_logits,
                    [num_batches, num_enn_samples, batch_size, num_classes])

  return batched_logits, batched_labels
