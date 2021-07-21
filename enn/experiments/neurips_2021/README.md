# Opensourcing code for NeurIPS paper

This folder includes the core experiment code to reproduce the results in 2021
NeurIPS submission: "Epistemic Neural Networks".

The main entry point to *run* the experiments comes in three places:

-   `run_testbed`: to reproduce the results of Section 4.1 on testbed problems.
-   `distillation/run`: to reproduce the results of Section 4.2 of
    "distillation" ENN.
-   `run_thompson`: to reproduce the results of Section 4.3 on Thompson sampling
    problems.

The detailed hyperparameter sweeps for each agent are collected in
`agent_factories.py`. In this file, we collect the default settings for each of
our "benchmark" agents (Sections 4.1, 4.3), as well as the complete sweeps
performed and outlined in Appendix A.
