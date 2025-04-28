
import numpy as np
import pandas as pd
from anndata import AnnData
from typing import Union, Sequence

def perturbation_effect(
    adata_perturb : AnnData,
    adata : AnnData,
    terminal_state : Union[str, Sequence[str]],
    ) -> AnnData:
    """
    Compute change in fate probabilities towards terminal states after perturbation. Negative values correspond to a decrease in
    probabilities, while positive values indicate an increase.

    Parameters
    ----------
    adata_perturb : AnnData
        Annotated data matrix of perturbed GRN.
    adata : AnnData
        Annotated data matrix of unperturbed GRN.
    terminal_state : str or Sequence[str]
        List of terminal states to compute probabilities for.

    Returns
    -------
    AnnData
        Annotated data matrix with the following added:
        - `terminal state perturbation` : Change in fate probabilities towards terminal state after perturbation.
    """

    if isinstance(terminal_state, str):
        terminal_state = [terminal_state]

    if "lineages_fwd" in adata.obsm and "lineages_fwd" in adata_perturb.obsm:
        perturb_df = pd.DataFrame(
            adata_perturb.obsm["lineages_fwd"], columns=adata_perturb.obsm["lineages_fwd"].names.tolist()
            )
        original_df = pd.DataFrame( 
            adata.obsm["lineages_fwd"], columns=adata.obsm["lineages_fwd"].names.tolist()
            )

        for state in terminal_state:
            adata.obs[f"perturbation effect on {state}"] = np.array(perturb_df[state] - original_df[state])

        return adata
    else:
        raise ValueError("Lineages not computed. Please compute lineages before using this function.")