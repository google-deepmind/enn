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
"""A minimalist wrapper around ENN experiment for testbed submission."""
import dataclasses
from typing import Dict, Optional

from acme.utils import loggers
from enn import base as enn_base
from enn import supervised
from enn import utils
from enn.experiments.neurips_2021 import base as testbed_base
from enn.experiments.neurips_2021 import enn_losses
import jax
import optax


@dataclasses.dataclass
class VanillaEnnConfig:
  """Configuration options for the VanillaEnnAgent."""
  enn_ctor: enn_losses.EnnCtor
  loss_ctor: enn_losses.LossCtor = enn_losses.default_enn_loss()
  optimizer: optax.GradientTransformation = optax.adam(1e-3)
  num_batches: int = 1000
  seed: int = 0
  batch_size: int = 100
  eval_batch_size: Optional[int] = None
  logger: Optional[loggers.Logger] = None
  train_log_freq: Optional[int] = None
  eval_log_freq: Optional[int] = None


def extract_enn_sampler(
    experiment: supervised.Experiment) -> testbed_base.EpistemicSampler:
  def enn_sampler(x: enn_base.Array, seed: int = 0) -> enn_base.Array:
    """Generate a random sample from posterior distribution at x."""
    net_out = experiment.predict(x, seed)
    return utils.parse_net_output(net_out)
  return jax.jit(enn_sampler)


@dataclasses.dataclass
class VanillaEnnAgent(testbed_base.TestbedAgent):
  """Wraps an ENN as a testbed agent, using sensible loss/bootstrapping."""
  config: VanillaEnnConfig
  eval_datasets: Optional[Dict[str, enn_base.BatchIterator]] = None
  experiment: Optional[supervised.Experiment] = None

  def __call__(
      self,
      data: testbed_base.Data,
      prior: testbed_base.PriorKnowledge,
  ) -> testbed_base.EpistemicSampler:
    """Wraps an ENN as a testbed agent, using sensible loss/bootstrapping."""
    enn = self.config.enn_ctor(prior)
    enn_data = enn_base.Batch(data.x, data.y)
    self.experiment = supervised.Experiment(
        enn=enn,
        loss_fn=self.config.loss_ctor(prior, enn),
        optimizer=self.config.optimizer,
        dataset=utils.make_batch_iterator(
            enn_data, self.config.batch_size, self.config.seed),
        seed=self.config.seed,
        logger=self.config.logger,
        train_log_freq=logging_freq(
            self.config.num_batches, log_freq=self.config.train_log_freq),
        eval_datasets=self.eval_datasets,
        eval_log_freq=logging_freq(
            self.config.num_batches, log_freq=self.config.eval_log_freq),
    )
    self.experiment.train(self.config.num_batches)
    return extract_enn_sampler(self.experiment)


def _round_to_integer(x: float) -> int:
  """Utility function to round a float to integer, or 1 if it would be 0."""
  assert x > 0
  x = int(x)
  if x == 0:
    return 1
  else:
    return x


def logging_freq(num_batches: int,
                 num_points: int = 100,
                 log_freq: Optional[int] = None) -> int:
  """Computes a logging frequency from num_batches, optionally log_freq."""
  if log_freq is None:
    log_freq = _round_to_integer(num_batches / num_points)
  return log_freq
