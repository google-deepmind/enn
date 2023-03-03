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
"""Base for defining metrics."""

from typing import Optional, Dict, NamedTuple

import chex
import jax
import jax.numpy as jnp
import typing_extensions


class MetricCalculator(typing_extensions.Protocol):
  """Interface for evaluation of multiple posterior samples based on a metric."""

  def __call__(self, logits: chex.Array, labels: chex.Array) -> float:
    """Calculates a metric based on logits and labels.

    Args:
      logits: An array of shape [A, B, C] where B is the batch size of data, C
        is the number of outputs per data (for classification, this is
        equal to number of classes), and A is the number of random samples for
        each data.
      labels: An array of shape [B, 1] where B is the batch size of data.

    Returns:
      A float number specifies the value of the metric.
    """


class PerExampleMetric(typing_extensions.Protocol):
  """Interface for metric per example."""

  def __call__(self, logits: chex.Array, labels: chex.Array) -> chex.Array:
    """Calculates a metric based on logits and labels.

    Args:
      logits: An array of shape [A, B, C] where B is the batch size of data, C
        is the number of outputs per data (for classification, this is
        equal to number of classes), and A is the number of random samples for
        each data.
      labels: An array of shape [B, 1] where B is the batch size of data.

    Returns:
      A metric per example of shape [B,].
    """


class MetricsState(NamedTuple):
  """State for metrics aggregation, default value should work for init."""
  value: float = 0.  # Should keep track of final metric value post aggregation
  count: int = 0  # The number of times the aggregator has been called
  extra: Optional[Dict[str, chex.Array]] = None  # Extra sufficient statistics.


class AggregateMetricCalculator(typing_extensions.Protocol):

  def __call__(
      self,
      logits: chex.Array,
      labels: chex.Array,
      state: Optional[MetricsState] = None,
  ) -> MetricsState:
    """Aggregates metric calculated over logits and labels with state."""


def make_average_aggregator(
    metric: MetricCalculator) -> AggregateMetricCalculator:
  """Keeps a running average of metric evaluated per batch."""

  def agg_metric(
      logits: chex.Array,
      labels: chex.Array,
      state: Optional[MetricsState] = None,
  ) -> MetricsState:
    value = metric(logits, labels)
    if state is None:
      # Initialize state
      state = MetricsState()
    new_count = state.count + 1
    new_value = (value + state.value * state.count) / new_count
    return MetricsState(new_value, new_count)

  return agg_metric


def average_sampled_log_likelihood(x: chex.Array) -> float:
  """Computes average log likelihood from samples.

  This method takes several samples of log-likelihood, converts
  them to likelihood (by exp), then takes the average, then
  returns the logarithm over the average  LogSumExp
  trick is used for numerical stability.

  Args:
    x: chex.Array
  Returns:
    log-mean-exponential
  """
  return jax.lax.cond(
      jnp.isneginf(jnp.max(x)),
      lambda x: -jnp.inf,
      lambda x: jnp.log(jnp.mean(jnp.exp(x - jnp.max(x)))) + jnp.max(x),
      operand=x,
  )
