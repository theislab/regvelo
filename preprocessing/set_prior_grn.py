
from pathlib import Path
from typing import Optional, Union
from urllib.request import urlretrieve

import torch
import numpy as np
import pandas as pd
import scvelo as scv
from anndata import AnnData
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist

def set_prior_grn(adata: AnnData, gt_net: pd.DataFrame, keep_dim: bool = False) -> AnnData:
    """Constructs a gene regulatory network (GRN) based on ground-truth interactions and gene expression data.

    Parameters
    ----------
    adata
        An annotated data matrix where `adata.X` contains gene expression data, and `adata.var` has gene identifiers.
    gt_net
        A DataFrame representing the ground-truth regulatory network with regulators as columns and targets as rows.
    keep_dim
        A boolean variable represeting if keep the output adata has the same dimensions.

    Returns
    -------
    None. Modifies `AnnData` object to include the GRN information, with network-related metadata stored in `uns`.
    """
    regulator_mask = adata.var_names.isin(gt_net.columns)
    regulators = adata.var_names[regulator_mask]

    target_mask = adata.var_names.isin(gt_net.index)
    targets = adata.var_names[target_mask]

    if keep_dim:
        skeleton = pd.DataFrame(0, index=adata.var_names, columns=adata.var_names, dtype=float)
        skeleton.loc[
            list(set(adata.var_names).intersection(gt_net.index)),
            list(set(adata.var_names).intersection(gt_net.columns)),
        ] = gt_net.loc[
            list(set(adata.var_names).intersection(gt_net.index)),
            list(set(adata.var_names).intersection(gt_net.columns)),
        ]
        gt_net = skeleton.copy()

    # Compute correlation matrix for genes
    gex = adata.layers["Ms"]
    correlation = 1 - cdist(gex.T, gex.T, metric="correlation")
    correlation = correlation[np.ix_(target_mask, regulator_mask)]
    correlation[np.isnan(correlation)] = 0

    # Filter ground-truth network and combine with correlation matrix
    grn = gt_net.loc[targets, regulators] * correlation

    # Threshold and clean the network
    grn = (grn.abs() >= 0.01).astype(int)
    np.fill_diagonal(grn.values, 0)  # Remove self-loops

    if keep_dim:
        skeleton = pd.DataFrame(0, index=adata.var_names, columns=adata.var_names, dtype=float)
        skeleton.loc[grn.columns, grn.index] = grn.T
    else:
        grn = grn.loc[grn.sum(axis=1) > 0, grn.sum(axis=0) > 0]

        # Prepare a matrix with all unique genes from the final network
        genes = grn.index.union(grn.columns).unique()
        skeleton = pd.DataFrame(0, index=genes, columns=genes, dtype=float)
        skeleton.loc[grn.columns, grn.index] = grn.T

    # Subset the original data to genes in the network and set final properties
    adata = adata[:, skeleton.index]
    skeleton = skeleton.loc[adata.var_names, adata.var_names]

    adata.uns["regulators"] = adata.var_names.to_numpy()
    adata.uns["targets"] = adata.var_names.to_numpy()
    adata.uns["skeleton"] = skeleton
    adata.uns["network"] = np.ones((adata.n_vars, adata.n_vars))

    return adata