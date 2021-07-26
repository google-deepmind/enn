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
"""Collecting factory methods for the best ENN agent configs."""
import dataclasses
from typing import Any, Callable, Dict, List, Sequence

from acme.utils import loggers
from enn import base as enn_base
from enn import networks
from enn.experiments.neurips_2021 import agents
from enn.experiments.neurips_2021 import base as testbed_base
from enn.experiments.neurips_2021 import enn_losses
import jax.numpy as jnp
import optax
import pandas as pd


ConfigCtor = Callable[[], agents.VanillaEnnConfig]


def make_ensemble_ctor(num_ensemble: int,
                       noise_scale: float,
                       prior_scale: float,
                       hidden_size: int = 50,
                       num_layers: int = 2,
                       seed: int = 0) -> ConfigCtor:
  """Generate an ensemble agent config."""
  def make_enn(prior: testbed_base.PriorKnowledge) -> enn_base.EpistemicNetwork:
    output_sizes = list([hidden_size] * num_layers) + [prior.num_classes]
    return networks.make_ensemble_mlp_with_prior_enn(
        output_sizes=output_sizes,
        dummy_input=jnp.ones([prior.num_train, prior.input_dim]),
        num_ensemble=num_ensemble,
        prior_scale=prior_scale,
    )

  def make_agent_config() -> agents.VanillaEnnConfig:
    """Factory method to create agent_config, swap this for different agents."""
    return agents.VanillaEnnConfig(
        enn_ctor=make_enn,
        loss_ctor=enn_losses.gaussian_regression_loss(
            num_ensemble, noise_scale, l2_weight_decay=0),
        num_batches=1000,  # Irrelevant for bandit
        logger=loggers.make_default_logger('experiment', time_delta=0),
        seed=seed,
    )
  return make_agent_config


def make_dropout_ctor(dropout_rate: float,
                      regularization_scale: float,
                      hidden_size: int = 50,
                      num_layers: int = 2,
                      dropout_input: bool = True,
                      regularization_tau: float = 1,
                      seed: int = 0) -> ConfigCtor:
  """Generate a dropout agent config."""
  def make_enn(prior: testbed_base.PriorKnowledge) -> enn_base.EpistemicNetwork:
    output_sizes = list([hidden_size] * num_layers) + [prior.num_classes]
    return networks.MLPDropoutENN(
        output_sizes=output_sizes,
        dropout_rate=dropout_rate,
        dropout_input=dropout_input,
        seed=seed,
    )

  def make_agent_config() -> agents.VanillaEnnConfig:
    """Factory method to create agent_config, swap this for different agents."""
    return agents.VanillaEnnConfig(
        enn_ctor=make_enn,
        loss_ctor=enn_losses.regularized_dropout_loss(
            num_index_samples=1,
            dropout_rate=dropout_rate,
            scale=regularization_scale,
            tau=regularization_tau,
        ),
        num_batches=1000,  # Irrelevant for bandit
        logger=loggers.make_default_logger('experiment', time_delta=0),
        seed=seed,
    )
  return make_agent_config


def make_hypermodel_ctor(index_dim: int,
                         noise_scale: float,
                         prior_scale: float,
                         hidden_size: int = 50,
                         num_layers: int = 2,
                         seed: int = 0) -> ConfigCtor:
  """Generate an ensemble agent config."""
  def make_enn(prior: testbed_base.PriorKnowledge) -> enn_base.EpistemicNetwork:
    output_sizes = list([hidden_size] * num_layers) + [prior.num_classes]
    return networks.MLPHypermodelPriorIndependentLayers(
        base_output_sizes=output_sizes,
        prior_scale=prior_scale,
        dummy_input=jnp.ones([prior.num_train, prior.input_dim]),
        indexer=networks.ScaledGaussianIndexer(index_dim),
        prior_base_output_sizes=output_sizes,
        hyper_hidden_sizes=[],
        seed=seed,
        scale=False)

  def make_agent_config() -> agents.VanillaEnnConfig:
    """Factory method to create agent_config, swap this for different agents."""
    return agents.VanillaEnnConfig(
        enn_ctor=make_enn,
        loss_ctor=enn_losses.gaussian_regression_loss(
            num_index_samples=index_dim * 20,
            noise_scale=noise_scale,
            l2_weight_decay=0,
        ),
        num_batches=1000,  # Irrelevant for bandit
        logger=loggers.make_default_logger('experiment', time_delta=0),
        seed=seed,
    )
  return make_agent_config


def make_bbb_ctor(sigma_0: float,
                  learning_rate: float,
                  hidden_size: int = 50,
                  num_layers: int = 2,
                  num_index_samples: int = 64,
                  seed: int = 0) -> ConfigCtor:
  """Generate an ensemble agent config."""
  def make_enn(prior: testbed_base.PriorKnowledge) -> enn_base.EpistemicNetwork:
    """Makes ENN."""
    output_sizes = list([hidden_size] * num_layers) + [prior.num_classes]
    enn = networks.make_bbb_enn(
        dummy_input=jnp.ones(shape=(prior.input_dim,)),
        base_output_sizes=output_sizes,
        sigma_0=sigma_0,
        scale=True)

    return enn

  def make_agent_config() -> agents.VanillaEnnConfig:
    """Factory method to create agent_config, swap this for different agents."""
    return agents.VanillaEnnConfig(
        enn_ctor=make_enn,
        loss_ctor=enn_losses.bbb_loss(
            sigma_0=sigma_0, num_index_samples=num_index_samples),
        optimizer=optax.adam(learning_rate),
        num_batches=1000,  # Irrelevant for bandit
        logger=loggers.make_default_logger('experiment', time_delta=0),
        seed=seed,
    )

  return make_agent_config


@dataclasses.dataclass
class AgentCtorConfig:
  settings: Dict[str, Any]  # Hyperparameters to work out which agent it is
  config_ctor: ConfigCtor  # Constructor for the agent config.


def make_ensemble_sweep() -> List[AgentCtorConfig]:
  """Generates the benchmark sweep for paper results."""
  sweep = []

  # Adding reasonably interesting ensemble agents
  for num_ensemble in [1, 3, 10, 30]:
    for noise_scale in [0, 1]:
      for prior_scale in [0, 1]:
        for num_layers in [2, 3]:
          for hidden_size in [50]:
            settings = {
                'agent': 'ensemble',
                'num_ensemble': num_ensemble,
                'noise_scale': noise_scale,
                'prior_scale': prior_scale,
                'num_layers': num_layers,
                'hidden_size': hidden_size
            }
            config_ctor = make_ensemble_ctor(num_ensemble, noise_scale,
                                             prior_scale, hidden_size,
                                             num_layers)
            sweep.append(AgentCtorConfig(settings, config_ctor))

  return sweep


def make_dropout_sweep() -> List[AgentCtorConfig]:
  """Generates the benchmark sweep for paper results."""
  sweep = []

  # Adding reasonably interesting dropout agents
  for dropout_rate in [0.05, 0.1, 0.2]:
    for regularization_scale in [0, 1e-6, 1e-4]:
      for num_layers in [2, 3]:
        for hidden_size in [50, 100]:
          settings = {
              'agent': 'dropout',
              'dropout_rate': dropout_rate,
              'regularization_scale': regularization_scale,
              'num_layers': num_layers,
              'hidden_size': hidden_size
          }
          config_ctor = make_dropout_ctor(dropout_rate, regularization_scale,
                                          hidden_size, num_layers)
          sweep.append(AgentCtorConfig(settings, config_ctor))

  return sweep


def make_hypermodel_sweep() -> List[AgentCtorConfig]:
  """Generates the benchmark sweep for paper results."""
  sweep = []

  # Adding reasonably interesting hypermodel agents
  for index_dim in [5, 10, 20]:
    for noise_scale in [0, 1]:
      for prior_scale in [0, 5]:
        for num_layers in [2, 3]:
          for hidden_size in [50]:
            settings = {
                'agent': 'hypermodel',
                'index_dim': index_dim,
                'noise_scale': noise_scale,
                'prior_scale': prior_scale,
                'num_layers': num_layers,
                'hidden_size': hidden_size
            }
            config_ctor = make_hypermodel_ctor(index_dim, noise_scale,
                                               prior_scale, hidden_size,
                                               num_layers)
            sweep.append(AgentCtorConfig(settings, config_ctor))

  return sweep


def make_bbb_sweep() -> List[AgentCtorConfig]:
  """Generates the benchmark sweep for paper results."""
  sweep = []

  # Adding reasonably interesting bbb agents
  for sigma_0 in [1, 10, 100, 200]:
    for learning_rate in [1e-3, 3e-4, 1e-4]:
      for num_layers in [2, 3]:
        for hidden_size in [50, 100]:
          settings = {
              'agent': 'bbb',
              'sigma_0': sigma_0,
              'learning_rate': learning_rate,
              'num_layers': num_layers,
              'hidden_size': hidden_size
          }
          config_ctor = make_bbb_ctor(sigma_0, learning_rate, hidden_size,
                                      num_layers)
          sweep.append(AgentCtorConfig(settings, config_ctor))

  return sweep


def make_agent_sweep(agent: str = 'all') -> Sequence[AgentCtorConfig]:
  """Generates the benchmark sweep for paper results."""
  if agent == 'all':
    agent_sweep = make_ensemble_sweep() + make_dropout_sweep(
    ) + make_hypermodel_sweep() + make_bbb_sweep()
  elif agent == 'ensemble':
    agent_sweep = make_ensemble_sweep()
  elif agent == 'hypermodel':
    agent_sweep = make_hypermodel_sweep()
  elif agent == 'dropout':
    agent_sweep = make_dropout_sweep()
  elif agent == 'bbb':
    agent_sweep = make_bbb_sweep()
  else:
    raise ValueError(f'agent={agent} is not valid!')

  return tuple(agent_sweep)


def load_agent_config(agent_id: int,
                      agent: str = 'all') -> agents.VanillaEnnConfig:
  """Use this in run.py to load an agent config from FLAGS.agent_id."""
  sweep = make_agent_sweep(agent)
  return sweep[agent_id].config_ctor()


def xm_agent_sweep(agent: str = 'all') -> Sequence[int]:
  """Use this in xm.py to create a sweep over all desired agent_id."""
  sweep = make_agent_sweep(agent)
  return range(len(sweep))


def join_metadata(df: pd.DataFrame) -> pd.DataFrame:
  """Joins data including 'agent_id' to work out agent settings."""
  assert 'agent_id' in df.columns
  sweep = make_agent_sweep()

  data = []
  for agent_id, agent_ctor_config in enumerate(sweep):
    agent_params = {'agent_id': agent_id}
    agent_params.update(agent_ctor_config.settings)
    data.append(agent_params)
  agent_df = pd.DataFrame(data)

  # Suffixes should not be needed... but added to be safe in case of clash.
  return pd.merge(df, agent_df, on='agent_id', suffixes=('', '_agent'))
