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
"""Tests for enn.metrics."""

from absl.testing import absltest
from absl.testing import parameterized
from enn import metrics
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


class MetricsTest(parameterized.TestCase):

  @parameterized.parameters([1, 3, 100])
  def test_average_sampled_log_likelihood_all_neginf(self, num_sample: int):
    """Test that average of negative infinity log likelihood is neg infinity."""
    log_likelihood = jnp.concatenate([jnp.array([-jnp.inf] * num_sample)])
    avg_log_likelihood = metrics.average_sampled_log_likelihood(
        log_likelihood)
    self.assertTrue(jnp.isneginf(avg_log_likelihood))

  @parameterized.parameters([3, 100])
  def test_average_sampled_log_likelihood_single_neginf(self, num_sample: int):
    """Test that avg with one negative infinity log likelihood is correct."""
    log_likelihood = jnp.concatenate([jnp.array([-jnp.inf]),
                                      jnp.zeros(shape=(num_sample - 1,))])
    avg_log_likelihood = metrics.average_sampled_log_likelihood(
        log_likelihood)
    expected_log_likelihood = jnp.log((num_sample -1) / num_sample)
    self.assertAlmostEqual(
        avg_log_likelihood, expected_log_likelihood,
        msg=(f'Expected log likelihood to be {expected_log_likelihood} ',
             f'but received {avg_log_likelihood}'),
        delta=0.1/num_sample)

  @parameterized.product(
      ll_val=[-1000, -100, 10, 0],
      num_sample=[1, 3, 100])
  def test_average_sampled_log_likelihood_const_values(
      self, ll_val: float, num_sample: int):
    """Test that average of equal log likelihood values is correct."""
    log_likelihood = ll_val * jnp.ones(shape=(num_sample,))
    avg_log_likelihood = metrics.average_sampled_log_likelihood(
        log_likelihood)
    self.assertAlmostEqual(
        avg_log_likelihood, ll_val,
        msg=(f'Expected log likelihood to be {ll_val} ',
             f'but received {avg_log_likelihood}'),
        delta=1e-5)

  @parameterized.product(seed=[1, 2, 3, 4, 5])
  def test_dyadic_matches(self, seed: int):
    rng = hk.PRNGSequence(seed)
    batch_size = 23
    num_classes = 7
    num_enn_samples = 13

    # Form trial data
    logits = jax.random.normal(
        next(rng), [num_enn_samples, batch_size, num_classes])
    labels = jax.random.randint(next(rng), [batch_size, 1], 0, num_classes)

    # Make sure not huge NLL
    new_calc = metrics.make_nll_polyadic_calculator(10, 2)
    new_nll = new_calc(logits, labels)
    assert np.isfinite(jnp.abs(new_nll))

  @parameterized.product(
      seed=[1000],
      num_enn_samples=[1, 10,],
      batch_size=[10],
      num_classes=[2, 10,],
      num_bins=[2, 10],
      num_batches=[1, 5])
  def test_ece_calculator(
      self,
      seed: int,
      num_enn_samples: int,
      batch_size: int,
      num_classes: int,
      num_bins: int,
      num_batches: int,
  ):
    """Tests that CalibrationErrorCalculator is correct by comparing it with BatchCalibrationErrorCalculator."""
    # We set this to `allow` (instead of the default `set`), because some
    # internal broadcasting is being done in tfp_ece_calculator.
    jax.config.update('jax_numpy_rank_promotion', 'allow')

    # Generate a set of random logits and labels
    rng = hk.PRNGSequence(seed)
    logits_ls = []
    labels_ls = []
    for _ in range(num_batches):
      logits = jax.random.normal(
          next(rng), [num_enn_samples, batch_size, num_classes])
      labels = jax.random.randint(
          next(rng), shape=[batch_size, 1], minval=0, maxval=num_classes)

      logits_ls.append(logits)
      labels_ls.append(labels)
    # Combining batches into one batch
    stacked_logits = jnp.concatenate(logits_ls, axis=1)
    stacked_labels = jnp.concatenate(labels_ls, axis=0)

    # Compute ece using tfp ece calculator which can only work when all data is
    # provided in one batch
    tfp_ece_calculator = metrics.SingleBatchECE(
        num_bins=num_bins)
    tfp_ece = tfp_ece_calculator(logits=stacked_logits, labels=stacked_labels)
    tfp_ece_value = float(tfp_ece)

    # Compute ece using our ece calculator which can also work when data is
    # is provided in multiple batches
    our_ece_calculator = metrics.ExpectedCalibrationError(
        num_bins=num_bins)
    ece_state = None
    for logits, labels in zip(logits_ls, labels_ls):
      ece_state = our_ece_calculator(
          logits=logits, labels=labels, state=ece_state)
    if ece_state is not None:
      our_ece_value = float(ece_state.value)

    # Check that ece results by our calculator and tfp calculator are the same.
    self.assertAlmostEqual(
        our_ece_value, tfp_ece_value,
        msg=f'our_ece_value={our_ece_value} not close enough to tfp_ece_value',
        delta=5e-2,
    )


if __name__ == '__main__':
  absltest.main()
