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

"""Exposing the public methods."""
# Kmeans
from enn.extra.kmeans import KMeansCluster
from enn.extra.kmeans import KMeansOutput

# VAE
from enn.extra.vae import VaeNet
from enn.extra.vae import VaeTrainer

# VAE losses
from enn.extra.vae_losses import binary_log_likelihood
from enn.extra.vae_losses import gaussian_log_likelihood
from enn.extra.vae_losses import latent_kl_divergence
from enn.extra.vae_losses import VaeApplyFn
from enn.extra.vae_losses import VaeLoss
from enn.extra.vae_losses import VaeOutput
