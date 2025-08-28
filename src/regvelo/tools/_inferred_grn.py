
import numpy as np
import pandas as pd
import torch
from anndata import AnnData

from .._model import REGVELOVI

def inferred_grn(
    vae: REGVELOVI,
    adata: AnnData,
    label: str = None,
    group: str | list[str] = None,
    cell_specific_grn: bool = False,
    data_frame: bool = False,
) -> np.ndarray | pd.DataFrame:
    r"""Infer the gene regulatory network (GRN) using a trained RegVelo VAE model.

    Parameters
    ----------
    vae
        Trained :class:`REGVELOVI` model with a `v_encoder` that supports GRN inference.
    adata
        Annotated data matrix containing cell-specific information and gene expression layers.
    label
        Column name in ``adata.obs`` specifying cell type or grouping labels.
    group
        Groups of cells to analyze.

        - ``"all"`` or ``None``: use all cells.
        - list of str: subset to these groups from ``adata.obs[label]``.
    cell_specific_grn
        If ``True``, compute a cell-specific GRN using individual cell data. 
    data_frame
        If ``True`` and ``cell_specific_grn`` is ``False``, return the GRN as a :class:`pandas.DataFrame`.

    Returns
    -------
    numpy.ndarray or pandas.DataFrame
        The inferred GRN as an array or DataFrame. For cell-specific GRN, always returns an array.
    """
    
    if "Ms" not in adata.layers:
        raise KeyError("Layer 'Ms' not found in adata.layers.")

    if cell_specific_grn is not True:
        # Retrieve unique cell types or groups from the specified label
        cell_types = np.unique(adata.obs[label])

        if group == "all":
            print("Computing global GRN...")
        else:
            # Subset the data to include only specified groups, raising an error for invalid groups
            if all(elem in cell_types for elem in group):
                adata = adata[adata.obs[label].isin(group)]
            else:
                raise TypeError(f"The group label contains elements not present in `adata.obs[{label}]`.")

        # Compute the GRN using the VAE's encoder and global mean gene expression
        GRN = (
            vae.module.v_encoder.GRN_Jacobian(torch.tensor(adata.layers["Ms"].mean(0)).to("cuda:0"))
            .detach()
            .cpu()
            .numpy()
        )
    else:
        # Compute the cell-specific GRN using the VAE's encoder for each cell
        GRN = vae.module.v_encoder.GRN_Jacobian2(torch.tensor(adata.layers["Ms"]).to("cuda:0")).detach().cpu().numpy()

    # Normalize GRN by the mean absolute non-zero values
    GRN = GRN / np.mean(np.abs(GRN)[GRN != 0])

    # Convert to a DataFrame if requested and not cell-specific
    if cell_specific_grn is not True and data_frame:
        GRN = pd.DataFrame(
            GRN,
            index=adata.var.index.tolist(),
            columns=adata.var.index.tolist(),
        )

    return GRN