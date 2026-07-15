import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Sequence

import mplscience
import seaborn as sns
from anndata import AnnData

import regvelo as rgv

from ._utils import SIGNIFICANCE_PALETTE

def visits_diff_per_tf(
    adata: AnnData,
    terminal_states: Sequence[str],
    dd_sig: np.ndarray,
    sig_palette: dict,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Collect per-terminal-state visit difference values and significance colours.

    Parameters
    ----------
    adata : AnnData
        AnnData with 'visits_diff' stored in adata.obs.
    terminal_states : sequence of str
        Terminal state labels to iterate over.
    dd_sig : np.ndarray
        Array of p-values, one per terminal state, from rgv.tl.simulated_visit_diff.
    sig_palette : dict
        Mapping from significance label ('n.s.', '*', '**', '***') to hex colour.

    Returns
    -------
    df : pd.DataFrame
        Long-form DataFrame with columns 'Value' (visit diff) and 'Group' (state).
    palette_rel : list of str
        Hex colour per terminal state based on significance.
    """
    data = []
    palette_rel = []

    for i, ts in enumerate(terminal_states):
        p_value = dd_sig[i]
        terminal_indices_sub = np.where(adata.obs["term_states_fwd"].isin([ts]))[0]

        values = adata.obs["visits_diff"].iloc[terminal_indices_sub]
        subgroups = [ts] * len(values)

        for val, subgrp in zip(values, subgroups):
            data.append({"Value": val, "Group": subgrp})

        significance = rgv.mt.get_significance(p_value)
        palette_rel.append(sig_palette[significance])

    return pd.DataFrame(data), palette_rel
