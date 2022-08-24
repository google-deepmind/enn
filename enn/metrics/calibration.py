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
"""Utility functions for calculating calibration error."""

import dataclasses
from typing import Dict, Optional, Tuple

import chex
from enn.metrics import base as metrics_base
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp


class ExpectedCalibrationError(metrics_base.AggregateMetricCalculator):
  """Computes expected calibration error (ECE) aggregated over enn samples.

  Expected calibration error (Guo et al., 2017, Naeini et al., 2015) is a scalar
  measure of calibration for probabilistic models. Calibration is defined as the
  level to which the accuracy over a set of predicted decisions and true
  outcomes associated with a given predicted probability level matches the
  predicted probability. A perfectly calibrated model would be correct `p`% of
  the time for all examples for which the predicted probability was `p`%, over
  all values of `p`.

  This metric can be computed as follows. First, convert the logits into
  probablities and take average of the probabilities over enn samples. Second,
  cut up the probability space interval [0, 1] into some number of bins. Then,
  for each example, store the predicted class (based on a threshold of 0.5 in
  the binary case and the max probability in the multiclass case), the predicted
  probability corresponding to the predicted class, and the true label into the
  corresponding bin based on the predicted probability. Then, for each bin,
  compute the average predicted probability ("confidence"), the accuracy of the
  predicted classes, and the absolute difference between the confidence and the
  accuracy ("calibration error"). Expected calibration error can then be
  computed as a weighted average calibration error over all bins, weighted based
  on the number of examples per bin.

  Perfect calibration under this setup is when, for all bins, the average
  predicted probability matches the accuracy, and thus the expected calibration
  error equals zero. In the limit as the number of bins goes to infinity, the
  predicted probability would be equal to the accuracy for all possible
  probabilities.

  References:
    1. Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q. On Calibration of Modern
       Neural Networks. in International Conference on Machine Learning (ICML)
       cs.LG, (Cornell University Library, 2017).
    2. Naeini, M. P., Cooper, G. F. & Hauskrecht, M. Obtaining Well Calibrated
       Probabilities Using Bayesian Binning. Proc Conf AAAI Artif Intell 2015,
       2901-2907 (2015).
  """

  def __init__(self, num_bins: int):
    self.num_bins = num_bins

  def _get_init_stats(self,) -> metrics_base.MetricsState:
    """Returns initial sufficient statistics for ece."""
    init_ece_stats = {
        'correct_sums': jnp.zeros(self.num_bins),
        'prob_sums': jnp.zeros(self.num_bins),
        'counts': jnp.zeros(self.num_bins),
    }
    return metrics_base.MetricsState(
        value=0,
        count=0,
        extra=init_ece_stats,
    )

  def __call__(
      self,
      logits: chex.Array,
      labels: chex.Array,
      state: Optional[metrics_base.MetricsState] = None,
  ) -> metrics_base.MetricsState:
    """Returns ece state."""
    chex.assert_rank(logits, 3)
    unused_num_enn_samples, num_data, num_classes = logits.shape
    chex.assert_shape(labels, [num_data, 1])

    class_probs = jax.nn.softmax(logits)
    mean_class_prob = jnp.mean(class_probs, axis=0)
    chex.assert_shape(mean_class_prob, [num_data, num_classes])

    batch_stats = _compute_per_batch_ece_stat(
        probs=mean_class_prob, labels=labels, num_bins=self.num_bins)

    if state is None:
      # Initialize state
      state = self._get_init_stats()

    # Update state
    new_stats = jax.tree_util.tree_map(jnp.add, state.extra, batch_stats)
    new_count = state.count + 1
    new_value = _map_stats_to_ece(new_stats)
    return metrics_base.MetricsState(
        value=new_value,
        count=new_count,
        extra=new_stats,
    )


@dataclasses.dataclass
class SingleBatchECE(metrics_base.MetricCalculator):
  """Computes expected calibration error (ECE) aggregated over enn samples.

  Note: this calculator can be used only in the case where all data is provided
  in ONE batch.
  """
  num_bins: int

  def __call__(self, logits: chex.Array, labels: chex.Array) -> float:
    """Returns ece."""
    chex.assert_rank(logits, 3)
    unused_num_enn_samples, num_data, num_classes = logits.shape
    chex.assert_shape(labels, [num_data, 1])

    class_probs = jax.nn.softmax(logits)
    mean_class_prob = jnp.mean(class_probs, axis=0)
    chex.assert_shape(mean_class_prob, [num_data, num_classes])

    predictions = jnp.argmax(mean_class_prob, axis=1)[:, None]
    chex.assert_shape(predictions, labels.shape)

    # ece
    mean_class_logits = jnp.log(mean_class_prob)
    chex.assert_shape(mean_class_logits, (num_data, num_classes))
    labels_true = jnp.squeeze(labels, axis=-1)
    chex.assert_shape(labels_true, (num_data,))
    labels_predicted = jnp.squeeze(predictions, axis=-1)
    chex.assert_shape(labels_predicted, (num_data,))
    return tfp.stats.expected_calibration_error(
        num_bins=self.num_bins,
        logits=mean_class_logits,
        labels_true=labels_true,
        labels_predicted=labels_predicted,
    )


# JAX implementation of tf.histogram_fixed_width_bins
def _histogram_fixed_width_bins(values: chex.Array,
                                value_range: Tuple[float, float],
                                num_bins: int,) -> chex.Array:
  """Bins the given values for use in a histogram.

  Args:
    values: An array.
    value_range: A tuple of the form (min_value, max_value). value <= min_value
      will be mapped to the first bin and value >= max_value will be mapped to
      the last bin.
    num_bins: Number of histogram bins.

  Returns:
    An array holding the indices of the binned values whose shape matches
    values.
  """
  _, bin_edges = jnp.histogram(values, bins=num_bins, range=value_range)
  return jnp.digitize(values, bins=bin_edges[1:])


# JAX implementation of tf.math.unsorted_segment_sum
def _unsorted_segment_sum(values: chex.Array,
                          segment_ids: chex.Array,
                          num_segments: int):
  """Computes the sum within segments of an array.

  Args:
    values: an array with the values to be summed.
    segment_ids: an array with integer dtype that indicates the segments of
      `values` (along its leading axis) to be summed. Values can be repeated and
      need not be sorted.
    num_segments: An int with nonnegative value indicating the number of
      segments.

  Returns:
    An array representing the segment sums.
  """
  return jax.ops.segment_sum(
      values, segment_ids=segment_ids, num_segments=num_segments)


def _compute_per_batch_ece_stat(
    probs: chex.Array,
    labels: chex.Array,
    num_bins: int,
) -> Dict[str, chex.Array]:
  """Computes sufficient statistics of Expected Calibration Error (ECE).

  Args:
    probs: An array of shape [num_data, num_classes].
    labels: An array of shape [num_data, 1].
    num_bins: Number of bins to maintain over the interval [0, 1].

  Returns:
    A dict of sufficient statistics.
  """
  chex.assert_rank(probs, 2)
  num_data, unused_num_classes = probs.shape
  chex.assert_shape(labels, [num_data, 1])

  # Compute predicted labels per example given class probabilities
  pred_labels = jnp.argmax(probs, axis=-1)
  # Compute maximum predicted probs per example given class probabilities
  pred_probs = jnp.max(probs, axis=-1)

  # Flatten labels to [num_data, ].
  labels = jnp.squeeze(labels)
  correct_preds = jnp.equal(pred_labels, labels)
  correct_preds = jnp.asarray(correct_preds, dtype=jnp.float32)

  bin_indices = _histogram_fixed_width_bins(
      values=pred_probs, value_range=(0., 1.), num_bins=num_bins)
  correct_sums = _unsorted_segment_sum(
      values=correct_preds,
      segment_ids=bin_indices,
      num_segments=num_bins,
  )
  prob_sums = _unsorted_segment_sum(
      values=pred_probs,
      segment_ids=bin_indices,
      num_segments=num_bins,
  )
  counts = _unsorted_segment_sum(
      values=jnp.ones_like(bin_indices),
      segment_ids=bin_indices,
      num_segments=num_bins,
  )

  ece_state = {
      'correct_sums': correct_sums,
      'prob_sums': prob_sums,
      'counts': counts,
  }
  return ece_state


def _map_stats_to_ece(ece_stats: Dict[str, chex.Array]) -> float:
  """Maps ece sufficient statistics to the ece value.

  ECE = Sum over bins (|bin-acc - bin-conf| * bin-count / total-count)

  Args:
    ece_stats: A dict of sufficient statistics for calculating ece.

  Returns:
    ECE value.
  """

  assert 'counts' in ece_stats
  assert 'correct_sums' in ece_stats
  assert 'prob_sums' in ece_stats
  counts = ece_stats['counts']
  accs = jnp.nan_to_num(ece_stats['correct_sums'] / counts)
  confs = jnp.nan_to_num(ece_stats['prob_sums'] / counts)
  total_count = jnp.sum(counts)
  return jnp.sum(counts/ total_count * jnp.abs(accs - confs))
