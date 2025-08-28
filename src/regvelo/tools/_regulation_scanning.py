import numpy as np
import pandas as pd
import cellrank as cr
from anndata import AnnData
from scvelo import logging as logg
from typing import Literal, Any

from .._model import REGVELOVI
from ..metrics import abundance_test
from ._in_silico_block_regulation_simulation import in_silico_block_regulation_simulation

def regulation_scanning(
    model: str,
    adata: AnnData,
    n_states: int,
    cluster_label: str,
    terminal_states: list[str] | None,
    TF: list[str],
    target: list[str],
    effect: float = 1e-3,
    method: Literal["likelihood", "t-statistics"] = "likelihood",
    **kwargs: Any,
) -> dict[str, list[str] | list[pd.Series]]:
    r"""Perform transcription factor (TF) scanning and perturbation analysis on a gene regulatory network.

    Parameters
    ----------
    model
        Path to the pretrained :class:`REGVELOVI` model.
    adata
        Annotated data matrix.
    n_states
        Number of macrostates to compute.
    cluster_label
        Key in ``adata.obs`` used for grouping cells into clusters.
    terminal_states
        List of terminal state labels. If ``None``, will be inferred by GPCCA.
    TF
        List of transcription factor genes to perturb.
    target
        List of target genes to perturb downstream of each TF.
    effect
        Value to assign when blocking a regulation (default: ``1e-3``).
    method
        Method to use in :func:`abundance_test` (``"likelihood"`` or ``"t-statistics"``).
    **kwargs
        Additional keyboard parameters passed on to :func:`in_silico_block_regulation_simulation`.

    Returns
    -------
    Dictionary with keys:
    
    - ``"links"``: formatted regulator -> target strings.
    - ``"coefficient"``: perturbation coefficients.
    - ``"pvalue"``: FDR-adjusted p-values.
    """

    n_samples = kwargs.get("n_samples", 30)

    # Load model and add outputs
    reg_vae = REGVELOVI.load(model, adata)
    adata = reg_vae.add_regvelo_outputs_to_adata(adata=adata, n_samples=n_samples)

    # Curate all targets of specific TF
    # Build kernel
    vk = cr.kernels.VelocityKernel(adata).compute_transition_matrix()
    ck = cr.kernels.ConnectivityKernel(adata).compute_transition_matrix()
    g = cr.estimators.GPCCA(0.8 * vk + 0.2 * ck)

    # Evaluate the fate prob on original space
    g.compute_macrostates(n_states=n_states, n_cells=30, cluster_key=cluster_label)

    # Predict cell fate probabilities
    if terminal_states is None:
        g.predict_terminal_states()
        terminal_states = g.terminal_states.cat.categories.tolist()
    g.set_terminal_states(terminal_states)
    g.compute_fate_probabilities(solver="direct")
    fate_prob = pd.DataFrame(
        g.fate_probabilities,
        index=adata.obs.index.tolist(),
        columns=g.fate_probabilities.names.tolist(),
    )
    fate_prob_original = fate_prob.copy()

    # Update n_states
    n_states = len(g.macrostates.cat.categories.tolist())

    # Containers
    coef, pvalue, links = [], [], []

    # Loop over TF-target pairs
    for regulator in TF:
        for gene in target:
            adata_target = in_silico_block_regulation_simulation(
                model, adata, regulator, gene, n_samples=n_samples, effects=effect
            )

            # Perturb the regulations
            vk = cr.kernels.VelocityKernel(adata_target).compute_transition_matrix()
            ck = cr.kernels.ConnectivityKernel(adata_target).compute_transition_matrix()
            g2 = cr.estimators.GPCCA(0.8 * vk + 0.2 * ck)

            # Evaluate the fate probabilities on original space
            ct_indices = {
                ct: adata.obs["term_states_fwd"][adata.obs["term_states_fwd"] == ct].index.tolist()
                for ct in terminal_states
            }

            # Recompute fate probabilities
            g2.set_terminal_states(ct_indices)
            g2.compute_fate_probabilities()
            fb = pd.DataFrame(
                g2.fate_probabilities,
                index=adata.obs.index.tolist(),
                columns=g2.fate_probabilities.names.tolist(),
            )

            # Align to original terminal states
            fate_prob2 = pd.DataFrame(columns=terminal_states, index=adata.obs.index.tolist())
            for i in terminal_states:
                fate_prob2.loc[:, i] = fb.loc[:, i]
            fate_prob2 = fate_prob2.fillna(0)

            arr = np.array(fate_prob2.sum(0))
            arr[arr != 0] = 1
            fate_prob = fate_prob * arr

            # Abundance test
            fate_prob2.index = [i + "_perturb" for i in fate_prob2.index]
            test_result = abundance_test(fate_prob, fate_prob2, method)
            coef.append(test_result.loc[:, "coefficient"])
            pvalue.append(test_result.loc[:, "FDR adjusted p-value"])

            logg.info("Finished " + fr"{regulator} -> {gene}")
            links.append(fr"$\text{{{regulator}}} \to \text{{{gene}}}$")

            # Reset
            fate_prob = fate_prob_original.copy()

    return {"links": links, "coefficient": coef, "pvalue": pvalue}