import numpy as np
import pandas as pd
from hep_ml import splot

def invent_sWeights(labels, signal_distr, background_distr, return_probas=False):
    """
    Given true labels, and "mass" distribution, creates sWeights
    Args:
    labels: np.array[n_examles], 1 for signal, 0 for background
    signal_distr, background_distr: scipy.stats distributions for "mass"
    Returns:
    np.array[n_examles], the signal sWeights
    """
    signal_indices = (labels == 1)
    pseudo_mass = np.empty(labels.shape)
    total_signal = signal_indices.sum()
    total_background = labels.shape[0] - total_signal
    pseudo_mass[signal_indices] = signal_distr.rvs(size=total_signal)
    pseudo_mass[~signal_indices] = background_distr.rvs(size=total_background)
    probs = pd.DataFrame(dict(
        sig=signal_distr.pdf(pseudo_mass)*total_signal/labels.shape[0],
        bck=background_distr.pdf(pseudo_mass)*total_background/labels.shape[0]))
    probs = probs.div(probs.sum(axis=1), axis=0)
    sweights = splot.compute_sweights(probs)
    if return_probas:
        return sweights.sig.values, probs.values[:, 0]
    else:
        return sweights.sig.values