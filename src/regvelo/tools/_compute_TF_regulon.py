import sys
import warnings

import numpy as np
import pandas as pd

import regvelo as rgv


def compute_TF_regulon(
    adata,
    rgv_model,
    cluster_key,
    TF,
    TERMINAL_STATES,
    threshold=0.4,
    n_states=7,
    n_samples=50,
    n_cells=30,
    solver="direct",
):
    """Scan each candidate TF's regulon (targets and regulators) and save scores.

    For every TF in ``TF``, extracts the inferred GRN weights from the trained
    model, identifies that TF's targets and regulators above ``threshold`` (by
    absolute weight), and runs ``rgv.tl.regulation_scanning`` for both edge sets,
    returning the per-edge coefficient tables keyed by TF.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData used to train ``rgv_model``. Must contain ``uns["skeleton"]``
        (the prior-GRN adjacency) whose row/column labels name targets/regulators.
    rgv_model : str
        Path to a saved, trained RegVelo model directory (as produced by
        ``REGVELOVI.save``). Loaded via ``REGVELOVI.load(rgv_model, adata)`` to
        access the inferred GRN, and passed as-is to ``rgv.tl.regulation_scanning``.
    cluster_key : str
        ``adata.obs`` column used as the cluster label during scanning
    TF : str or list of str
        Candidate transcription factor(s) to scan.
    TERMINAL_STATES : list of str
        Terminal-state labels passed to ``rgv.tl.regulation_scanning``.
    threshold : float, optional
        Absolute GRN-weight cutoff for retaining an edge as a target/regulator.
        Default ``0.4``.
    n_states : int, optional
        Number of macrostates used by ``regulation_scanning``. Default ``7``.
    n_samples : int, optional
        Number of samples drawn per scan. Default ``50``.
    n_cells : int, optional
        Number of cells per macrostate, passed to ``rgv.tl.regulation_scanning``. Default ``30``.
    solver : str, optional
        Linear solver passed to ``rgv.tl.regulation_scanning``. Default ``"direct"``.

    Returns
    ----------
    coef_targets, coef_regulators: dict of str to DataFrame
        Per-TF target and regulator coefficient tables, keyed by TF, to be
        passed individually into :func:`regvelo.pl.plot_regulon`.

    """
    if isinstance(TF, str):
        TF = [TF]

    vae = rgv.REGVELOVI.load(rgv_model, adata)

    GRN = pd.DataFrame(
        vae.module.v_encoder.fc1.weight.detach().cpu().numpy(),
        index=adata.uns["skeleton"].index.tolist(),
        columns=adata.uns["skeleton"].columns.tolist(),
    )

    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    TF_candidate = TF
    coef_targets = {}
    coef_regulators = {}

    for TF in TF_candidate:
        print(f'Candidate factor: {TF}')
        targets = GRN.loc[:,TF][GRN.loc[:,TF].abs() > threshold]
        targets = np.array(targets.index.tolist())[np.array(targets) != 0]

        if targets.size == 0:
            print("No targets found - recommended to try lower threshold")

        else:
            n_targets = len(targets)
            print(f'Found {n_targets} targets')

            perturb_screening = rgv.tl.regulation_scanning(model=rgv_model,
                                                       adata=adata,
                                                       n_states=n_states,
                                                       cluster_label=cluster_key,
                                                       terminal_states=TERMINAL_STATES,
                                                       TF=[TF],
                                                       target=targets,
                                                       effect=0,
                                                       n_samples=n_samples,
                                                       n_cells=n_cells,
                                                       solver=solver)

            coef = pd.DataFrame(np.array(perturb_screening["coefficient"]))
            coef.index = perturb_screening["links"]
            coef.columns = list(perturb_screening["coefficient"][0].keys())
            coef_targets[TF] = coef

        # Filter to top regulators only
        regulators = GRN.loc[TF,:][GRN.loc[TF,:].abs() > threshold]
        regulators = np.array(regulators.index.tolist())[np.array(regulators) != 0]

        if regulators.size == 0:
            print("No regulators found - recommended to try lower threshold")

        else:
            n_regulators = len(regulators)
            print(f'Found {n_regulators} regulators')

            perturb_screening_tf = rgv.tl.regulation_scanning(model=rgv_model,
                                                          adata=adata,
                                                          n_states=n_states,
                                                          cluster_label=cluster_key,
                                                          terminal_states=TERMINAL_STATES,
                                                          TF=regulators,
                                                          target=[TF],
                                                          effect=0,
                                                          n_samples=n_samples,
                                                          n_cells=n_cells,
                                                          solver=solver)

            coef = pd.DataFrame(np.array(perturb_screening_tf["coefficient"]))
            coef.index = perturb_screening_tf["links"]
            coef.columns = list(perturb_screening_tf["coefficient"][0].keys())
            coef_regulators[TF] = coef

    return coef_targets, coef_regulators
