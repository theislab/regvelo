from pathlib import Path
from typing import Optional, Union

import pandas as pd

from scanpy import read

from scvelo.core import cleanup
from scvelo.read_load import load

url_adata = "https://drive.google.com/uc?id=1Nzq1F6dGw-nR9lhRLfZdHOG7dcYq7P0i&export=download"
url_grn = "https://drive.google.com/uc?id=1ci_gCwdgGlZ0xSn6gSa_-LlIl9-aDa1c&export=download/"
url_adata_murine = "https://drive.usercontent.google.com/download?id=19bNQfW3jMKEEjpjNdUkVd7KDTjJfqxa5&export=download&authuser=1&confirm=t&uuid=4fdf3051-229b-4ce2-b644-cb390424570a&at=APcmpoxgcuZ5r6m6Fb6N_2Og6tEO:1745354679573"

def zebrafish_nc(file_path: Union[str, Path] = "data/zebrafish_nc/adata_zebrafish_preprocessed.h5ad"):
    """Zebrafish neural crest cells.

    Single cell RNA-seq datasets of zebrafish neural crest cell development across 
    seven distinct time points using ultra-deep Smart-seq3 technique.

    There are four distinct phases of NC cell development: 1) specification at the NPB, 2) epithelial-to-mesenchymal
    transition (EMT) from the neural tube, 3) migration throughout the periphery, 4) differentiation into distinct cell types

    Arguments:
    ---------
    file_path
        Path where to save dataset and read it from.

    Returns
    -------
    Returns `adata` object
    """
    adata = read(file_path, backup_url=url_adata, sparse=True, cache=True)
    return adata

def zebrafish_grn(file_path: Union[str, Path] = "data/zebrafish_nc/prior_GRN.csv"):
    """Zebrafish neural crest cells.

    Single cell RNA-seq datasets of zebrafish neural crest cell development across 
    seven distinct time points using ultra-deep Smart-seq3 technique.

    There are four distinct phases of NC cell development: 1) specification at the NPB, 2) epithelial-to-mesenchymal
    transition (EMT) from the neural tube, 3) migration throughout the periphery, 4) differentiation into distinct cell types

    Arguments:
    ---------
    file_path
        Path where to save dataset and read it from.

    Returns
    -------
    Returns `adata` object
    """
    grn = pd.read_csv(url_grn, index_col = 0)
    grn.to_csv(file_path)
    return grn

def murine_nc(file_path: Union[str, Path] = "data/murine_nc/adata_preprocessed.h5ad"):
    """Mouse neural crest cells.

    Single cell RNA-seq datasets of mouse neural crest cell development subset from Qiu, Chengxiang, et al. datasets.

    The GRN is saved in `adata.uns["skeleton"]`, which learned via pySCENIC

    Arguments:
    ---------
    file_path
        Path where to save dataset and read it from.

    Returns
    -------
    Returns `adata` object
    """
    adata = read(file_path, backup_url=url_adata_murine, sparse=True, cache=True)
    return adata