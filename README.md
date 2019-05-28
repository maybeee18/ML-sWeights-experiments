# Experiments with training NNs and Catboost on sWeights
## Setup
1. Setup conda environment `conda env create -f environment.yml`
2. Get the catboost with the losses implemented and negative weights check disabled from https://github.com/kazeevn/catboost/tree/constrained_regression , build wheel according to the [[https://catboost.ai/docs/installation/python-installation-method-build-from-source-linux-macos.html][instructions]].
3. Install the compiled catboost with `pip`
You now should be able to run the experiments in the repository
