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
"""Exposing the public methods of datasets."""

# Base
from enn.datasets.base import ArrayBatch
from enn.datasets.base import ArrayBatchIterator
from enn.datasets.base import DataIndex
from enn.datasets.base import Dataset
from enn.datasets.base import DatasetGenerator
from enn.datasets.base import DatasetTransformer
from enn.datasets.base import DatasetWithTransform
from enn.datasets.base import EVAL_TRANSFORMERS_DEFAULT
from enn.datasets.base import OodVariant


# CIFAR10
from enn.datasets.cifar import Cifar
from enn.datasets.cifar import Cifar10
from enn.datasets.cifar import Cifar100
from enn.datasets.cifar import CifarVariant
from enn.datasets.cifar import Split as Cifar10Split

# ImageNet
from enn.datasets.imagenet import Imagenet
from enn.datasets.imagenet import MAX_NUM_TRAIN as IMAGENET_MAX_NUM_TRAIN
from enn.datasets.imagenet import Split as ImagenetSplit

# MNIST
from enn.datasets.mnist import Mnist
from enn.datasets.mnist import Split as MnistSplit

## Dataset Transformations

# Local Sample
from enn.datasets.transforms.local_sample import make_dyadic_transform
from enn.datasets.transforms.local_sample import make_repeat_sample_transform
from enn.datasets.transforms.local_sample import PerturbFn

# OOD
from enn.datasets.transforms.ood import get_dataset_transform_from_type
from enn.datasets.transforms.ood import make_ood_transformers
from enn.datasets.transforms.ood import sample_classes

# Utils
from enn.datasets.utils import add_data_index_to_dataset
from enn.datasets.utils import change_ds_dict_to_enn_batch
from enn.datasets.utils import get_per_device_batch_size
from enn.datasets.utils import OverrideTrainDataset
from enn.datasets.utils import slice_dataset_to_batches
