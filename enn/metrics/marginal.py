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

"""Metrics used to evaluate marginal predictions."""

import chex
from enn.metrics import base as metrics_base
import jax
import jax.numpy as jnp


def make_nll_marginal_calculator() -> metrics_base.MetricCalculator:
  """Returns a MetricCalculator for marginal negative log likelihood (nll)."""
  return lambda x, y: -1 * calculate_marginal_ll(x, y)


def make_accuracy_calculator() -> metrics_base.MetricCalculator:
  """Returns a MetricCalculator that calculate accuracy."""
  return calculate_accuracy


def calculate_marginal_ll(logits: chex.Array, labels: chex.Array) -> float:
  """Computes marginal log likelihood (ll) aggregated over enn samples."""
  unused_num_enn_samples, num_data, num_classes = logits.shape
  chex.assert_shape(labels, (num_data, 1))

  probs = jnp.mean(jax.nn.softmax(logits), axis=0)
  chex.assert_shape(probs, [num_data, num_classes])

  return categorical_log_likelihood(probs, labels) / num_data


def calculate_accuracy(logits: chex.Array, labels: chex.Array) -> float:
  """Computes classification accuracy (acc) aggregated over enn samples."""
  chex.assert_rank(logits, 3)
  unused_num_enn_samples, num_data, num_classes = logits.shape
  chex.assert_shape(labels, [num_data, 1])

  class_probs = jax.nn.softmax(logits)
  mean_class_prob = jnp.mean(class_probs, axis=0)
  chex.assert_shape(mean_class_prob, [num_data, num_classes])

  predictions = jnp.argmax(mean_class_prob, axis=1)[:, None]
  chex.assert_shape(predictions, [num_data, 1])

  return jnp.mean(predictions == labels)


def categorical_log_likelihood(probs: chex.Array, labels: chex.Array) -> float:
  """Computes joint log likelihood based on probs and labels."""
  num_data, unused_num_classes = probs.shape
  assert len(labels) == num_data
  assigned_probs = probs[jnp.arange(num_data), jnp.squeeze(labels)]
  return jnp.sum(jnp.log(assigned_probs))
