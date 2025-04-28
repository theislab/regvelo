import torch

import pandas as pd
import numpy as np

from scipy.stats import ranksums, ttest_ind
from sklearn.metrics import roc_auc_score

import os, shutil

from .._model import REGVELOVI
from .utils import p_adjust_bh


def abundance_test(
    prob_raw : pd.DataFrame, 
    prob_pert : pd.DataFrame, 
    method : str = "likelihood"
    ) -> pd.DataFrame:
    """
    Perform an abundance test between two probability datasets.

    Parameters
    ----------
    prob_raw : pd.DataFrame
        Raw probabilities dataset.
    prob_pert : pd.DataFrame
        Perturbed probabilities dataset.
    method : str, optional (default="likelihood")
        Method to calculate scores: "likelihood" or "t-statistics".

    Returns
    -------
    pd.DataFrame
        Dataframe with coefficients, p-values, and FDR adjusted p-values.
    """
    y = [1] * prob_raw.shape[0] + [0] * prob_pert.shape[0]
    X = pd.concat([prob_raw, prob_pert], axis=0)

    table = []
    for i in range(prob_raw.shape[1]):
        pred = np.array(X.iloc[:, i])
        if np.sum(pred) == 0:
            score, pval = np.nan, np.nan
        else:
            pval = ranksums(pred[np.array(y) == 0], pred[np.array(y) == 1])[1]
            if method == "t-statistics":
                score = ttest_ind(pred[np.array(y) == 0], pred[np.array(y) == 1])[0]
            elif method == "likelihood":
                score = roc_auc_score(y, pred)
            else:
                raise NotImplementedError("Supported methods are 't-statistics' and 'likelihood'.")

        table.append(np.expand_dims(np.array([score, pval]), 0))

    table = np.concatenate(table, axis=0)
    table = pd.DataFrame(table, index=prob_raw.columns, columns=["coefficient", "p-value"])
    table["FDR adjusted p-value"] = p_adjust_bh(table["p-value"].tolist())
    return table