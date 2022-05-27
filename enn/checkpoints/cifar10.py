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
"""Entries on Cifar10."""

from enn import base as enn_base
from enn import datasets
from enn import networks
from enn import utils as enn_utils
from enn.checkpoints import base as checkpoint_base
from enn.checkpoints import epinet as checkpoint_epinet
from enn.checkpoints import utils
from enn.networks.epinet import priors
from enn.networks.epinet import resnet as resnet_epinet_lib


def _make_resnet_ctor(
    config: networks.ResNetConfig,
    temperature: float = 1.,
) -> checkpoint_base.EnnCtor:
  """Creates a resnet constructor for appropriate config."""
  def enn_ctor() -> enn_base.EpistemicNetworkWithState:
    enn = networks.EnsembleResNetENN(
        num_output_classes=datasets.Cifar10().num_classes,
        num_ensemble=1,
        is_training=False,
        enable_double_transpose=False,
        config=config,
    )
    return enn_utils.scale_enn_output(enn=enn, scale=1/temperature)
  return enn_ctor


def resnet_18() -> checkpoint_base.EnnCheckpoint:
  """Benchmark baseline for ResNet18 on Cifar10."""
  return checkpoint_base.EnnCheckpoint(
      name='cifar10_resnet18',
      load_fn=utils.load_from_file(file_name='resnet18_cifar10'),
      enn_ctor=_make_resnet_ctor(
          networks.CanonicalResNets.RESNET_18.value,
          temperature=1.1,
      ),
      dataset=datasets.Cifar10(),
  )


def resnet_32() -> checkpoint_base.EnnCheckpoint:
  """Benchmark baseline for ResNet32 on Cifar10."""
  return checkpoint_base.EnnCheckpoint(
      name='cifar10_resnet32',
      load_fn=utils.load_from_file(file_name='resnet32_cifar10'),
      enn_ctor=_make_resnet_ctor(
          networks.CanonicalResNets.RESNET_32.value,
          temperature=1.1,
      ),
      dataset=datasets.Cifar10(),
  )


def resnet_44() -> checkpoint_base.EnnCheckpoint:
  """Benchmark baseline for ResNet44 on Cifar10."""
  return checkpoint_base.EnnCheckpoint(
      name='cifar10_resnet44',
      load_fn=utils.load_from_file(file_name='resnet44_cifar10'),
      enn_ctor=_make_resnet_ctor(
          networks.CanonicalResNets.RESNET_44.value,
          temperature=1.1,
      ),
      dataset=datasets.Cifar10(),
  )


def resnet_56() -> checkpoint_base.EnnCheckpoint:
  """Benchmark baseline for ResNet56 on Cifar10."""
  return checkpoint_base.EnnCheckpoint(
      name='cifar10_resnet56',
      load_fn=utils.load_from_file(file_name='resnet56_cifar10'),
      enn_ctor=_make_resnet_ctor(
          networks.CanonicalResNets.RESNET_56.value,
          temperature=1.1,
      ),
      dataset=datasets.Cifar10(),
  )


def resnet_110() -> checkpoint_base.EnnCheckpoint:
  """Benchmark baseline for ResNet110 on Cifar10."""
  return checkpoint_base.EnnCheckpoint(
      name='cifar10_resnet110',
      load_fn=utils.load_from_file(file_name='resnet110_cifar10'),
      enn_ctor=_make_resnet_ctor(
          networks.CanonicalResNets.RESNET_110.value,
          temperature=1.1,
      ),
      dataset=datasets.Cifar10(),
  )


def _make_epinet_config(
    base_checkpoint: checkpoint_base.EnnCheckpoint
) -> resnet_epinet_lib.ResnetFinalEpinetConfig:
  """Creates an epinet config given a base net checkpoint."""
  def prior_fn_ctor() -> networks.PriorFn:
    return priors.make_cifar_conv_prior(num_ensemble=20, num_classes=10)

  return resnet_epinet_lib.ResnetFinalEpinetConfig(
      base_checkpoint=base_checkpoint,
      index_dim=20,
      num_classes=10,
      epinet_hiddens=[50,],
      epi_prior_scale=4.,
      add_prior_scale=0.0,
      prior_fn_ctor=prior_fn_ctor,
      freeze_base=True,
      temperature=0.5,
  )


def resnet_18_final_epinet() -> checkpoint_epinet.EpinetCheckpoint:
  """Final-layer epinet with Resnet18 base model on CIFAR10."""
  return resnet_epinet_lib.make_checkpoint_from_config(
      name='cifar10_final_epinet_resnet18',
      load_fn=utils.load_from_file(file_name='resnet18_epinet_cifar10'),
      config=_make_epinet_config(resnet_18()),
  )


def resnet_32_final_epinet() -> checkpoint_epinet.EpinetCheckpoint:
  """Final-layer epinet with Resnet32 base model on CIFAR10."""
  return resnet_epinet_lib.make_checkpoint_from_config(
      name='cifar10_final_epinet_resnet32',
      load_fn=utils.load_from_file(file_name='resnet32_epinet_cifar10'),
      config=_make_epinet_config(resnet_32()),
  )


def resnet_44_final_epinet() -> checkpoint_epinet.EpinetCheckpoint:
  """Final-layer epinet with Resnet44 base model on CIFAR10."""
  return resnet_epinet_lib.make_checkpoint_from_config(
      name='cifar10_final_epinet_resnet44',
      load_fn=utils.load_from_file(file_name='resnet44_epinet_cifar10'),
      config=_make_epinet_config(resnet_44()),
  )


def resnet_56_final_epinet() -> checkpoint_epinet.EpinetCheckpoint:
  """Final-layer epinet with Resnet56 base model on CIFAR10."""
  return resnet_epinet_lib.make_checkpoint_from_config(
      name='cifar10_final_epinet_resnet56',
      load_fn=utils.load_from_file(file_name='resnet56_epinet_cifar10'),
      config=_make_epinet_config(resnet_56()),
  )


def resnet_110_final_epinet() -> checkpoint_epinet.EpinetCheckpoint:
  """Final-layer epinet with Resnet110 base model on CIFAR10."""
  return resnet_epinet_lib.make_checkpoint_from_config(
      name='cifar10_final_epinet_resnet110',
      load_fn=utils.load_from_file(file_name='resnet110_epinet_cifar10'),
      config=_make_epinet_config(resnet_110()),
  )

