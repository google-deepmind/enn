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
"""Entries on ImageNet."""

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
        num_output_classes=datasets.Imagenet().num_classes,
        num_ensemble=1,
        is_training=False,
        enable_double_transpose=True,
        config=config,
    )
    return enn_utils.scale_enn_output(enn=enn, scale=1/temperature)
  return enn_ctor


def resnet_50() -> checkpoint_base.EnnCheckpoint:
  """Benchmark baseline for ResNet50 on ImageNet."""
  return checkpoint_base.EnnCheckpoint(
      name='imagenet_resnet50',
      load_fn=utils.load_from_file(file_name='resnet50_imagenet'),
      enn_ctor=_make_resnet_ctor(
          config=networks.CanonicalResNets.RESNET_50.value,
          temperature=0.8,
      ),
      dataset=datasets.Imagenet(),
  )


def resnet_101() -> checkpoint_base.EnnCheckpoint:
  """Benchmark baseline for ResNet101 on ImageNet."""
  return checkpoint_base.EnnCheckpoint(
      name='imagenet_resnet101',
      load_fn=utils.load_from_file(file_name='resnet101_imagenet'),
      enn_ctor=_make_resnet_ctor(
          config=networks.CanonicalResNets.RESNET_101.value,
          temperature=0.8,
      ),
      dataset=datasets.Imagenet(),
  )


def resnet_152() -> checkpoint_base.EnnCheckpoint:
  """Benchmark baseline for ResNet152 on ImageNet."""
  return checkpoint_base.EnnCheckpoint(
      name='imagenet_resnet152',
      load_fn=utils.load_from_file(file_name='resnet152_imagenet'),
      enn_ctor=_make_resnet_ctor(
          config=networks.CanonicalResNets.RESNET_152.value,
          temperature=0.8,
      ),
      dataset=datasets.Imagenet(),
  )


def resnet_200() -> checkpoint_base.EnnCheckpoint:
  """Benchmark baseline for ResNet200 on ImageNet."""
  return checkpoint_base.EnnCheckpoint(
      name='imagenet_resnet200',
      load_fn=utils.load_from_file(file_name='resnet200_imagenet'),
      enn_ctor=_make_resnet_ctor(
          config=networks.CanonicalResNets.RESNET_200.value,
          temperature=0.8,
      ),
      dataset=datasets.Imagenet(),
  )


def _make_epinet_config(
    base_checkpoint: checkpoint_base.EnnCheckpoint
) -> resnet_epinet_lib.ResnetFinalEpinetConfig:
  """Creates an epinet config given a base net checkpoint."""
  return resnet_epinet_lib.ResnetFinalEpinetConfig(
      base_checkpoint=base_checkpoint,
      index_dim=30,
      num_classes=1000,
      base_logits_scale=1.,
      epinet_hiddens=[50],
      epi_prior_scale=1.,
      add_prior_scale=1.,
      prior_fn_ctor=lambda: priors.make_imagenet_conv_prior(num_ensemble=30),
      freeze_base=True,
      temperature=0.7,
  )


def resnet_50_final_epinet() -> checkpoint_epinet.EpinetCheckpoint:
  """Final-layer epinet with Resnet50 base model on Imagenet."""
  return resnet_epinet_lib.make_checkpoint_from_config(
      name='imagenet_final_epinet_resnet50',
      load_fn=utils.load_from_file(file_name='resnet50_epinet_imagenet'),
      config=_make_epinet_config(resnet_50()),
  )


def resnet_101_final_epinet() -> checkpoint_epinet.EpinetCheckpoint:
  """Final-layer epinet with Resnet101 base model on Imagenet."""
  return resnet_epinet_lib.make_checkpoint_from_config(
      name='imagenet_final_epinet_resnet101',
      load_fn=utils.load_from_file(file_name='resnet101_epinet_imagenet'),
      config=_make_epinet_config(resnet_101()),
  )


def resnet_152_final_epinet() -> checkpoint_epinet.EpinetCheckpoint:
  """Final-layer epinet with Resnet152 base model on Imagenet."""
  return resnet_epinet_lib.make_checkpoint_from_config(
      name='imagenet_final_epinet_resnet152',
      load_fn=utils.load_from_file(file_name='resnet152_epinet_imagenet'),
      config=_make_epinet_config(resnet_152()),
  )


def resnet_200_final_epinet() -> checkpoint_epinet.EpinetCheckpoint:
  """Final-layer epinet with Resnet200 base model on Imagenet."""
  return resnet_epinet_lib.make_checkpoint_from_config(
      name='imagenet_final_epinet_resnet200',
      load_fn=utils.load_from_file(file_name='resnet200_epinet_imagenet'),
      config=_make_epinet_config(resnet_200()),
  )
