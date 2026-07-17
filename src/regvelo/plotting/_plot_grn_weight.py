import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scvelo as scv
import mplscience
import regvelo as rgv

def plot_grn_weight(
    adata,
    vae,
    TF,
    target_list,
    device="cpu"):

    """Plot cell-resolved regulatory weights for one TF against several targets.

    For each target, extracts the cell-specific GRN weight of the ``TF -> target``
    edge from ``rgv.tl.inferred_grn(..., cell_specific_grn=True)``, stores it in
    ``adata.obs``, and draws a 3-column UMAP row: TF expression, target
    expression, and the per-cell edge weight.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData with a UMAP embedding; ``var.index`` must contain ``TF`` and every
        gene in ``target_list``.
    vae : regvelo.REGVELOVI
        Trained RegVelo model, used to infer the cell-specific GRN.
    TF : str
        Transcription factor (regulator) whose outgoing edges are plotted.
    target_list : iterable of str
        Target genes to plot, one row each.
    device : str, optional
        Device passed to ``rgv.tl.inferred_grn`` for GRN inference. Default ``"cpu"``.

    Returns
    -------
    None
    """
    GRN = rgv.tl.inferred_grn(vae, adata, cell_specific_grn=True, device=device)

    ncols = 3
    nrows = len(target_list)
    figsize = 6
    wspace = 0.05

    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * figsize + figsize * wspace * (ncols - 1), nrows * figsize),
    )
    plt.subplots_adjust(wspace=wspace)    

    for row, target in enumerate(target_list):
        regulon = GRN[
            :, [i == target for i in adata.var.index], [i == TF for i in adata.var.index]
        ].reshape(-1)
        adata.obs[f"{TF}_{target}_weight"] = regulon

        scv.pl.umap(adata, color=[TF], frameon=False, title=[TF], ax=axs[row, 0], show=False)
        scv.pl.umap(adata, color=[target], frameon=False, title=[target], ax=axs[row, 1], show=False)
        scv.pl.umap(adata, color=[f"{TF}_{target}_weight"], cmap="Reds", ax=axs[row, 2], show=False)
        plt.tight_layout()
        plt.show()
