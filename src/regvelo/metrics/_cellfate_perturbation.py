import pandas as pd
import cellrank as cr

from anndata import AnnData
from typing import Sequence, Literal, Union

from ._abundance_test import abundance_test

def cellfate_perturbation(
        perturbed : dict[str, AnnData],
        baseline : AnnData,
        terminal_state : str | Sequence[str] | None = None,
        terminal_indices : dict[str, list[str]] | None = None,
        method : Literal["likelihood", "t-statistics"] = "likelihood",
        solver: Union[str, Literal["direct", "gmres", "lgmres", "bicgstab", "gcrotmk"]] = "gmres",
        ) -> pd.DataFrame:
    r"""Compute depletion likelihood or score for TF perturbation.

    Parameters
    ----------
    perturbed
        Dictionary mapping TF candidate names to their perturbed :class:`~anndata.AnnData` objects,
        each containing precomputed fate probabilities under :attr:`obsm["lineages_fwd"]`.
    baseline
        Unperturbed :class:`~anndata.AnnData` object with precomputed fate probabilities 
        under :attr:`obsm["lineages_fwd"]`.
    terminal_state
        One or more terminal states for which depletion scores are computed.
    terminal_indices
        Dictionary mapping each terminal state to the list of cell indices (obs_names) belonging to it.
    method
        Scoring method to use:
        - "t-statistics": uses t-statistics.
        - "likelihood": uses ROC AUC score.
    solver
        Linear system solver used by CellRank’s GPCCA estimator when computing fate probabilities.
        Supported options include:
        - `"direct"`: direct solver.
        - `"gmres"`: Generalized Minimal Residual method (default).
        - `"lgmres"`: Loose GMRES.
        - `"bicgstab"`: BiConjugate Gradient Stabilized method.
        - `"gcrotmk"`: GCROTMK solver.
        Choice of solver may affect performance and numerical stability.

    Returns
    -------
    DataFrame summarizing depletion scores and significance statistics.
    """

    if "lineages_fwd" not in baseline.obsm and terminal_indices is None:
        raise KeyError("Lineages not found in baseline.obsm. Please compute lineages first.")
    
    if isinstance(terminal_state, str):
        terminal_state = [terminal_state]
        
    if terminal_indices is not None:
        vk = cr.kernels.VelocityKernel(baseline).compute_transition_matrix()
        estimator = cr.estimators.GPCCA(vk)
        estimator.set_terminal_states(terminal_indices)
        estimator.compute_fate_probabilities(solver=solver)

        for adata_p in perturbed.values():
            vk_p = cr.kernels.VelocityKernel(adata_p).compute_transition_matrix()
            estimator_p = cr.estimators.GPCCA(vk_p)
            estimator_p.set_terminal_states(terminal_indices)
            estimator_p.compute_fate_probabilities(solver=solver)
        terminal_state = list(terminal_indices.keys())

    fate_prob_perturb = {}
    for TF, adata_target_perturb in perturbed.items():
        if "lineages_fwd" not in adata_target_perturb.obsm:
            raise KeyError("Lineages not found in adata_target_perturb.obsm. Please compute lineages first.")

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
        stats = abundance_test(prob_raw=fate_prob_raw, prob_pert=perturbed_prob, method=method)
        if method == "likelihood":
            df = pd.DataFrame(
                {
                    "Depletion likelihood": stats.iloc[:, 0].tolist(),
                    "p-value": stats.iloc[:, 1].tolist(),
                    "FDR adjusted p-value": stats.iloc[:, 2].tolist(),
                    "Terminal state": stats.index.tolist(),
                    "TF": [TF] * stats.shape[0],
                }
            )

        elif method == "t-statistics":
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

    return df



