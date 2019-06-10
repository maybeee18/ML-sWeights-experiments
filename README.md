# Experiments with training NNs and Catboost on sWeights
## Setup
1. Setup conda environment `conda env create -f environment.yml`
2. Get the catboost with the losses implemented and negative weights check disabled from [the repository](https://github.com/kazeevn/catboost/tree/constrained_regression), build wheel according to the [instructions](https://catboost.ai/docs/installation/python-installation-method-build-from-source-linux-macos.html).
3. Install the compiled catboost with `pip`
4. Install the utility library for neural networks: `pip install git+https://gitlab.com/mborisyak/craynn.git@5ee1057bbc9bc9a9d2a16d826196cc28044cebb9`
5. By default, the precompiled CPU version of tensorflow is installed. If you plan to run the Neural Network experiments and have a GPU, you might want to install a GPU version

You now should be able to run the experiments in the repository
