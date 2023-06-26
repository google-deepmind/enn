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
"""Install script for setuptools."""

import imp  # pylint: disable=deprecated-module

import setuptools

# Additional requirements for testing.
# Discussion on `pytype==2021.8.11`: pytype latest version raises some issues
# tracked in https://github.com/google/pytype/issues/1359. This version is
# borrowed from https://github.com/deepmind/acme/blob/master/setup.py#L71
testing_require = [
    'pytest-xdist',
    'pytype',  # to be compatible with dm-acme
]

setuptools.setup(
    name='enn',
    description=(
        'Epistemic neural networks. '
        'A library for probabilistic inference via neural networks.'
    ),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/deepmind/enn',
    author='DeepMind',
    author_email='enn-eng+os@google.com',
    license='Apache License, Version 2.0',
    version=imp.load_source('_metadata', 'enn/_metadata.py').__version__,
    keywords='probabilistic-inference python machine-learning',
    packages=setuptools.find_packages(),
    # Discussion on pinning versions: As of 2023-03-31, `dm-acme==0.4.0` is the
    # latest version that supports Python 3.7, 3.8, and 3.9. However, this
    # version works only with `tensorflow==2.8.0`, `tensorflow-datasets==4.6.0`,
    # and `tensorflow_probability==0.15.0` as specified in
    # https://github.com/deepmind/acme/blob/master/setup.py#L39.
    # Moreover, our library does not require loading `tensorflow_probability`,
    # it is just loaded to pin to the specific version required by dm-acme.
    install_requires=[
        'absl-py',
        'chex',
        'dill',
        'dm-haiku',
        'jax',
        'jaxline',
        'numpy',
        'optax',
        'pandas',
        'rlax',
        'plotnine',
        'scikit-learn',
        'tensorflow',  # to be compatible with dm-acme
        'tensorflow-datasets',  # to be compatible with dm-acme
        'tensorflow_probability',  # to be compatible with dm-acme
        'typing-extensions',
    ],
    extras_require={
        'testing': testing_require,
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
