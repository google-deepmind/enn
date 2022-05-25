# Epistemic Neural Networks

A library for uncertainty representation and training in neural networks.

## Introduction
Many applications in deep learning requires or benefit from going beyond a
point estimte and representing uncertainty about the model. The coherent use
of Bayes’ rule and probability theory are the gold standard for updating
beliefs and estimating uncertainty. But exact computation quickly becomes
infeasible for even simple problems. Modern machine learning has developed an
effective toolkit for learning in high-dimensional using a simple and
coherent convention. Epistemic neural network (ENN) is a library that
provides a similarly simple and coherent convention for defining and training
neural networks that represent uncertainty over a hypothesis class of models.

## Technical overview
In a supervised setting, For input `x_i ∈ X` and
outputs `y_i ∈ Y` a point estimate `f_θ(x)` is trained by fitting the
observed data `D = {(xi, yi) for i = 1, ..., N}` by minimizing a loss
function `l(θ, D) ∈ R`. In epistemic neural networks we
introduce the concept of an epistemic index `z ∈ I ⊆ R^{n_z}` distributed
according to some reference distribution `p_z(·)`. An augmented epistemic
function approximator then takes the form `f_θ(x, z)`; where the function
class `fθ(·, z)` is a neural network. The index `z` allows unambiguous
identification of a corresponding function value and sampling `z` corresponds
to sampling from the hypothesis class of functions.

On some level, ENNs are purely a notational convenience and most existing
approaches to dealing with uncertainty in deep learning can be rephrased in
this way. For example, an ensemble of point estimates `{f_θ1, ..., f_θK }`
can be viewed as an ENN with `θ = (θ1, .., θK)`, `z ∈ {1, .., K}`, and
`f_θ(x, z) := f_θz(x)`. However, this simplicity hides a deeper insight: that
the process of epistemic update itself can be tackled through the tools of
machine learning typically reserved for point estimates, through the addition
of this epistemic index. Further, since these machine learning tools were
explicitly designed to scale to large and complex problems, they might
provide tractable approximations to large scale Bayesian inference even where
the exact computations are intractable.

For a more comprehensive overview, see the accompanying [paper].

## Reproducing NeurIPS experiments

To reproduce the experiments from our paper please see `experiments/neurips_2021`.

## Getting started

You can get started in our [colab tutorial] without installing anything on your
machine.

### Installation

We have tested `ENN` on Python 3.7. To install the dependencies:

1.  **Optional**: We recommend using a
    [Python virtual environment](https://docs.python.org/3/tutorial/venv.html)
    to manage your dependencies, so as not to clobber your system installation:

    ```bash
    python3 -m venv enn
    source enn/bin/activate
    pip install --upgrade pip setuptools
    ```

2.  Install `ENN` directly from [github](https://github.com/deepmind/enn):

    ```bash
    pip install git+https://github.com/deepmind/enn
    ```
3.  Test that you can load `ENN` by training a simple ensemble ENN.

    ```python
    from acme.utils.loggers.terminal import TerminalLogger

    from enn import losses
    from enn import networks
    from enn import supervised
    from enn.supervised import regression_data
    import optax

    # A small dummy dataset
    dataset = regression_data.make_dataset()

    # Logger
    logger = TerminalLogger('supervised_regression')

    # ENN
    enn = networks.MLPEnsembleMatchedPrior(
        output_sizes=[50, 50, 1],
        num_ensemble=10,
    )

    # Loss
    loss_fn = losses.average_single_index_loss(
        single_loss=losses.L2LossWithBootstrap(),
        num_index_samples=10
    )

    # Optimizer
    optimizer = optax.adam(1e-3)

    # Train the experiment
    experiment = supervised.Experiment(
        enn, loss_fn, optimizer, dataset, seed=0, logger=logger)
    experiment.train(FLAGS.num_batch)
    ```

More examples can be found in the [colab tutorial].

4. **Optional**: run the tests by executing `./test.sh` from ENN root directory.

## Citing

If you use `ENN` in your work, please cite the accompanying [paper]:

```bibtex
@article{osband2022epistemic,
  title={Epistemic neural networks},
  author={Osband, Ian and Wen, Zheng and Asghari, Seyed Mohammad and Dwaracherla, Vikranth and Ibrahimi, Morteza and Lu, Xiuyuan and Van Roy, Benjamin},
  journal={arXiv preprint arXiv:2107.08924},
  year={2022}
}
```

[colab tutorial]: https://colab.research.google.com/github/deepmind/enn/blob/master/enn/colabs/enn_demo.ipynb
[paper]: https://arxiv.org/abs/2107.08924
