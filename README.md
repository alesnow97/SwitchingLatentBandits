# Switching Latent Bandits

-------------------------------------

This repository contains the code for the methods introduced in [1].

The repository is organized into separate directories for the proposed simulations, baseline policies, and data generation scripts.
Specifically:
- the *environment* directory defines the different sets of bandit instances and the Markov chain controlling their dynamics. Observation and transition models are built as described in Appendix A in [1],
- the *experiments* directory contains data generated from the different runs of the algorithm,
- the *run_files* directory contains scripts for executing experiments. Each script requires specific hyperparameters, which vary depending on the experiment,
- the *simulations* directory includes the files handling the execution of the experiments and the storage of the results in the *experiments* directory,
- the *policies* directory contains the implementation of our estimation approach, the SL-EC algorithm and the other used baselines,
- the *plots* directory contains files used for plotting the results stored in the *experiments* directory. 

The provided code primarily supports the experiments presented in Figures 1, 2, and 5.  

To replicate the remaining experiments, minor modifications to the available scripts should be done.

------------------------------------

[1] Russo Alessio, Metelli Alberto Maria and Restelli Marcello. Switching Latent Bandits, in Transaction of Machine Learning Research, 2024.