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

def get_permutation_scores(save_path: Union[str, Path] = Path("data/")) -> pd.DataFrame:
    """
    Get the reference permutation scores on positive and negative controls.

    Parameters
    ----------
    save_path
        path to save the csv file

    """
    if isinstance(save_path, str):
        save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    if not (save_path / "permutation_scores.csv").is_file():
        URL = "https://figshare.com/ndownloader/files/36658185"
        urlretrieve(url=URL, filename=save_path / "permutation_scores.csv")

    return pd.read_csv(save_path / "permutation_scores.csv")