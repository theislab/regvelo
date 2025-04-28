import torch
import pandas as pd
from anndata import AnnData
from typing import Union, Sequence, Any, Optional, Tuple
import cellrank as cr

from .abundance_test import abundance_test

def depletion_score(perturbed : dict[str, AnnData],
                    baseline : AnnData,
                    terminal_state : Union[str, Sequence[str]],
                    **kwargs : Any,
                    ) -> Tuple[pd.DataFrame, dict[str, AnnData]]:
    """
    Compute depletion scores.

    Parameters
    ----------
    perturbed : dict[str, AnnData]
        Dictionary mapping TF candidates to their perturbed AnnData objects.
    baseline : AnnData
        Annotated data matrix. Fate probabilities already computed.
    terminal_state : str or Sequence[str]
        List of terminal states to compute probabilities for.
    kwargs : Any
        Optional
        Additional keyword arguments passed to CellRank and plot functions.

    Returns
    -------
    Tuple[pd.DataFrame, dict[str, AnnData]]
        A tuple containing:

        - **df** – Summary of depletion scores and associated statistics.
        - **adata_perturb_dict** – Dictionary mapping TFs to their perturbed AnnData objects.
    """

    macro_kwargs = {k: kwargs[k] for k in ("n_states", "n_cells", "cluster_key", "method") if k in kwargs}
    compute_fate_probabilities_kwargs = {k: kwargs[k] for k in ("solver", "tol") if k in kwargs}

    if "lineages_fwd" not in baseline.obsm:
        raise KeyError("Lineages not found in baseline.obsm. Please compute lineages first.")

    if isinstance(terminal_state, str):
        terminal_state = [terminal_state]

    # selecting indices of cells that have reached a terminal state
    ct_indices = {
        ct: baseline.obs["term_states_fwd"][baseline.obs["term_states_fwd"] == ct].index.tolist()
        for ct in terminal_state
    }

    fate_prob_perturb = {}
    for TF, adata_target_perturb in perturbed.items():
        vk = cr.kernels.VelocityKernel(adata_target_perturb).compute_transition_matrix()
        vk.write_to_adata()
        estimator = cr.estimators.GPCCA(vk)

        estimator.compute_macrostates(**macro_kwargs)

        estimator.set_terminal_states(ct_indices)
        estimator.compute_fate_probabilities(**compute_fate_probabilities_kwargs)

        perturbed[TF] = adata_target_perturb

        perturbed_prob = pd.DataFrame(
            adata_target_perturb.obsm["lineages_fwd"], 
            columns=adata_target_perturb.obsm["lineages_fwd"].names.tolist()
            )[terminal_state]

        fate_prob_perturb[TF] = perturbed_prob

    fate_prob_raw = pd.DataFrame(
        baseline.obsm["lineages_fwd"], 
        columns=baseline.obsm["lineages_fwd"].names.tolist()
        )

    dfs = []
    for TF, perturbed_prob in fate_prob_perturb.items():
        stats = abundance_test(prob_raw=fate_prob_raw, prob_pert=perturbed_prob)
        df = pd.DataFrame(
            {
                "Depletion score": stats.iloc[:, 0].tolist(),
                "p-value": stats.iloc[:, 1].tolist(),
                "FDR adjusted p-value": stats.iloc[:, 2].tolist(),
                "Terminal state": stats.index.tolist(),
                "TF": [TF] * stats.shape[0],
            }
        )
        dfs.append(df)

    df = pd.concat(dfs)

    df["Depletion score"] = 2 * (0.5 - df["Depletion score"])

    return df, perturbed



