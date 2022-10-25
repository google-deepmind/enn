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

"""Exposing the public methods of the networks."""
# Base
from enn.networks.base import ApplyArray
from enn.networks.base import ApplyNoState
from enn.networks.base import EnnArray
from enn.networks.base import EnnNoState
from enn.networks.base import EpistemicModule
from enn.networks.base import InitArray
from enn.networks.base import InitNoState
from enn.networks.base import Output
from enn.networks.base import OutputWithPrior
# BBB
from enn.networks.bbb import make_bbb_enn
# Categorical regression ensemble
from enn.networks.categorical_ensembles import CategoricalRegressionMLP
from enn.networks.categorical_ensembles import CatMLPEnsembleGpPrior
from enn.networks.categorical_ensembles import CatMLPEnsembleMlpPrior
from enn.networks.categorical_ensembles import CatOutputWithPrior
# Dropout
from enn.networks.dropout import MLPDropoutENN
# Einsum MLP
from enn.networks.einsum_mlp import EnsembleMLP
from enn.networks.einsum_mlp import make_einsum_ensemble_mlp_enn
from enn.networks.einsum_mlp import make_ensemble_mlp_with_prior_enn
from enn.networks.einsum_mlp import make_ensemble_prior
# Ensemble
from enn.networks.ensembles import combine_functions_choice_via_index
from enn.networks.ensembles import combine_functions_linear_in_index
from enn.networks.ensembles import Ensemble
from enn.networks.ensembles import EnsembleWithState
from enn.networks.ensembles import make_mlp_ensemble_prior_fns
from enn.networks.ensembles import MLPEnsembleMatchedPrior
# Gaussian ENN
from enn.networks.gaussian_enn import GaussianNoiseEnn
from enn.networks.gaussian_enn import GaussianNoiseMLP
# Hypermodels
from enn.networks.hypermodels import hypermodel_module
from enn.networks.hypermodels import MLPHypermodel
from enn.networks.hypermodels import MLPHypermodelPriorIndependentLayers
from enn.networks.hypermodels import MLPHypermodelWithHypermodelPrior
from enn.networks.hypermodels import PriorMLPIndependentLayers
# Index MLP
from enn.networks.index_mlp import ConcatIndexMLP
from enn.networks.index_mlp import IndexMLPEnn
from enn.networks.index_mlp import IndexMLPWithGpPrior
# Indexers
from enn.networks.indexers import DirichletIndexer
from enn.networks.indexers import EnsembleIndexer
from enn.networks.indexers import GaussianIndexer
from enn.networks.indexers import GaussianWithUnitIndexer
from enn.networks.indexers import PrngIndexer
from enn.networks.indexers import ScaledGaussianIndexer
# LeNet (MNIST)
from enn.networks.lenet import EnsembleLeNet5ENN
from enn.networks.lenet import LeNet5
# MLP
from enn.networks.mlp import ExposedMLP
from enn.networks.mlp import ProjectedMLP
# Priors
from enn.networks.priors import convert_enn_to_prior_fn
from enn.networks.priors import EnnNoStateWithAdditivePrior
from enn.networks.priors import EnnWithAdditivePrior
from enn.networks.priors import get_random_mlp_with_index
from enn.networks.priors import make_null_prior
from enn.networks.priors import make_random_feat_gp
from enn.networks.priors import NetworkWithAdditivePrior
from enn.networks.priors import PriorFn
# ResNet (Imagenet)
from enn.networks.resnet.base import EnsembleResNetENN
from enn.networks.resnet.base import resnet_model
# ResNet Configs (Imagenet)
from enn.networks.resnet.lib import CanonicalResNets
from enn.networks.resnet.lib import ForwardFn
from enn.networks.resnet.lib import ResBlockV1
from enn.networks.resnet.lib import ResBlockV2
from enn.networks.resnet.lib import ResNet
from enn.networks.resnet.lib import ResNetConfig
# ResNet (Imagenet)
from enn.networks.resnet.priors import ResnetCnnPrior
from enn.networks.resnet.priors import ResnetMlpPrior
# Utils
from enn.networks.utils import epistemic_network_from_module
from enn.networks.utils import make_centered_enn
from enn.networks.utils import make_centered_enn_no_state
from enn.networks.utils import parse_net_output
from enn.networks.utils import parse_to_output_with_prior
from enn.networks.utils import scale_enn_output
from enn.networks.utils import wrap_apply_no_state_as_apply
from enn.networks.utils import wrap_enn_as_enn_no_state
from enn.networks.utils import wrap_enn_no_state_as_enn
from enn.networks.utils import wrap_init_no_state_as_init
from enn.networks.utils import wrap_transformed_as_enn
from enn.networks.utils import wrap_transformed_as_enn_no_state
# VGG (Cifar10)
from enn.networks.vgg import EnsembleVGGENN
from enn.networks.vgg import VGG
