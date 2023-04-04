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
"""Public methods for epinet."""
# Base
from enn.networks.epinet.base import BaseHiddenParser
from enn.networks.epinet.base import combine_base_epinet_as_enn
from enn.networks.epinet.base import EpinetApplyWithState
from enn.networks.epinet.base import EpinetInitWithState
from enn.networks.epinet.base import EpinetWithState

# last_layer
from enn.networks.epinet.last_layer import MLPEpinetWithPrior
from enn.networks.epinet.last_layer import parse_base_hidden

# ResNet
from enn.networks.epinet.mlp import make_mlp_epinet

# Prior
from enn.networks.epinet.priors import combine_epinet_and_prior
from enn.networks.epinet.priors import make_cifar_conv_prior
from enn.networks.epinet.priors import make_imagenet_conv_prior
from enn.networks.epinet.priors import make_imagenet_mlp_prior

# ResNet
from enn.networks.epinet.resnet import make_checkpoint_from_config
from enn.networks.epinet.resnet import ResnetFinalEpinet
from enn.networks.epinet.resnet import ResnetFinalEpinetConfig
