# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Classification log likelihoods retain finite tails and correct mixtures."""

from absl.testing import absltest
from absl.testing import parameterized
from enn.metrics import joint
from enn.metrics import marginal
import jax
import jax.numpy as jnp
import numpy as np
from scipy import special


class LogSpaceLikelihoodTest(parameterized.TestCase):

  @parameterized.product(gap=[2., 120., 1000.], samples=[1, 3],
                         metric=['marginal', 'joint', 'polyadic'], compiled=[False, True])
  def test_finite_scores_and_gradients(self, gap, samples, metric, compiled):
    logits = jnp.broadcast_to(jnp.array([0., -gap]), (samples, 2, 2))
    labels = jnp.ones((2, 1), dtype=jnp.int32)
    if metric == 'marginal':
      fn, repeats = marginal.make_nll_marginal_calculator(), 1
    elif metric == 'joint':
      fn, repeats = joint.make_nll_joint_calculator(tau=2), 2
    else:
      fn, repeats = joint.make_nll_polyadic_calculator(tau=5, kappa=2), 5
    evaluated = jax.jit(fn) if compiled else fn
    actual = evaluated(logits, labels)
    expected = repeats * np.logaddexp(0., gap)
    self.assertTrue(np.isfinite(actual))
    np.testing.assert_allclose(actual, expected, rtol=2e-6)
    gradient = jax.grad(lambda x: evaluated(x, labels))(logits)
    self.assertTrue(np.isfinite(gradient).all())
    # Shift the labelled class everywhere: expected d(NLL)/d(shift).
    np.testing.assert_allclose(np.sum(gradient[..., 1]),
                               -repeats * special.expit(gap), rtol=3e-6)

  @parameterized.parameters(0., 1000.)
  def test_unequal_samples_average_probabilities_not_log_likelihoods(self, gap):
    values = np.array([[[0., -gap - 2.], [1., 0.]],
                       [[0., -gap - 4.], [0., 2.]],
                       [[0., -gap - 7.], [0., -1.]]], dtype=np.float32)
    labels = jnp.array([[1], [0]])
    selected = special.log_softmax(values.astype(np.float64), axis=-1)[:, np.arange(2), [1, 0]]
    expected_marginal = np.mean(special.logsumexp(selected, axis=0) - np.log(3))
    expected_joint = special.logsumexp(selected.sum(axis=1)) - np.log(3)
    np.testing.assert_allclose(marginal.calculate_marginal_ll(jnp.array(values), labels),
                               expected_marginal, rtol=2e-6)
    np.testing.assert_allclose(joint.calculate_joint_ll(jnp.array(values), labels),
                               expected_joint, rtol=2e-6)

  def test_zero_probability_and_probability_input_api_are_unchanged(self):
    labels = jnp.array([[1]])
    logits = jnp.array([[[0., -jnp.inf]]])
    self.assertEqual(marginal.calculate_marginal_ll(logits, labels), -jnp.inf)
    self.assertEqual(joint.calculate_joint_ll(logits, labels), -jnp.inf)
    self.assertEqual(marginal.categorical_log_likelihood(jnp.array([[1., 0.]]), labels), -jnp.inf)


if __name__ == '__main__':
  absltest.main()
