import os
from contextlib import contextmanager
from typing import Sequence

import numpy as np
import pandas as pd

from anndata import AnnData
from scvelo import logging as logg
from tqdm.auto import tqdm

import cellrank as cr
import regvelo as rgv

from ..plotting._utils import SIGNIFICANCE_PALETTE, delta_to_probability, smooth_score


@contextmanager
def _mute_cellrank():
    """Temporarily silence CellRank log messages and progress bars.

    Sets ``cr.settings.verbosity = 0`` (suppresses the ``logg`` info lines) and
    ``TQDM_DISABLE=1`` (suppresses the transition-matrix and softmax-scale
    progress bars, which are not governed by verbosity). Both are restored on
    exit, so progress bars created outside this context (e.g. the TF screening
    loop) are unaffected.
    """
    old_verbosity = cr.settings.verbosity
    old_tqdm_disable = os.environ.get("TQDM_DISABLE")
    cr.settings.verbosity = 0
    os.environ["TQDM_DISABLE"] = "1"
    try:
        yield
    finally:
        cr.settings.verbosity = old_verbosity
        if old_tqdm_disable is None:
            os.environ.pop("TQDM_DISABLE", None)
        else:
            os.environ["TQDM_DISABLE"] = old_tqdm_disable


def visits_diff_per_tf(
    adata: AnnData,
    terminal_states: Sequence[str],
    dd_sig: np.ndarray,
    sig_palette: dict,
) -> tuple[pd.DataFrame, list[str]]:
    """Collect per-terminal-state visit difference values and significance colours.

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


def _assign_visits_diff(
    adata: AnnData,
    adata_perturb: AnnData,
    total_simulations: int,
    key: str = "visits",
    embedding: str = "X_pca",
    n_neighbors: int = 10,
) -> None:
    """Compute ``{key}_diff`` and ``{key}_diff_smooth`` in ``adata.obs`` without plotting.

    Mirrors the computation performed by :func:`rgv.pl.simulated_visit_diff`, so the
    downstream visit-difference statistics remain available when plotting is disabled.
    """
    adata.obs[f"{key}_diff"] = delta_to_probability(
        adata_perturb.obs[key] - adata.obs[key],
        k=1 / np.sqrt(total_simulations),
    )
    smooth_score(adata, key=f"{key}_diff", embedding=embedding, n_neighbors=n_neighbors)


def markov_density_screening(
    adata: AnnData,
    adata_perturb_dict: dict[str, AnnData],
    TERMINAL_STATES: list[str],
    STARTING_POINTS: list[str],
    tf_ko_list: list[str] | None,
    cluster_key: str,
    method: str = "stepwise",
    n_step_to_use: int = 500,
    n_simulations: int = 1000,
    seed: int = 0,
    output_dir: str = ".",
    plot: bool = False,
    save: bool = False,
) -> None:

    r"""Run Markov simulations to score TF perturbation effects on cell fate density.

    For each TF, simulates random walks from :attr:`STARTING_POINTS` to
    :attr:`TERMINAL_STATES` in both the baseline and perturbed transition
    matrices, computes a density difference (dd) score and its significance, then
    collects per-cell visit statistics. After all TFs are processed, produces
    summary CSVs and (optionally) a combined significance-annotated boxplot.

    Parameters
    ----------
    adata
        Baseline AnnData with velocity outputs and terminal state annotations in ``adata.obs['term_states_fwd']``.
    adata_perturb_dict
        Perturbed AnnData objects keyed by TF name, as returned by :func:`rgv.tl.in_silico_block_simulation`.
    TERMINAL_STATES
        Terminal state labels used to define absorption boundaries for the Markov simulation.
    STARTING_POINTS
        Cell-type labels (from ``adata.obs[cluster_key]``) used to seed random walks.
    tf_ko_list
        TFs to simulate. If None, all TFs in ``adata.var['TF']`` are used.
    cluster_key
        `.obs` column name for cell-type annotation, used to identify starting cells.
    method
        Markov simulation method passed to :func:`rgv.tl.markov_density_simulation`, either ``'stepwise'`` or ``'one-step'``.
    n_step_to_use
        Number of steps for the Markov random walk.
    n_simulations
        Number of simulations per starting cell, passed to :func:`rgv.tl.markov_density_simulation`.
    seed
        Random seed passed to :func:`rgv.tl.markov_density_simulation`.
    output_dir
        Directory to write the result CSVs into when ``save=True``. Default ``"."`` (current working directory).
    plot
        Whether to draw the density-difference figures. Default ``False``.
    save
        Whether to also write the result tables to CSV in ``output_dir``. Default ``False``.

    Returns
    -------
    Nothing. Results are stored in ``adata.uns['markov_density_screening']`` as a dict with keys:

    - ``'dd_score_by_TF'``: per-TF density difference scores, significance values, and baseline/perturbed absorption rates.
    - ``'visits_by_TF'``: per-cell visit counts, densities, and differences for each TF.
    - ``'screen_perturbation_rate'``: long-form per-terminal-state visit differences with significance annotation for every TF.

    Saves
    -----
    When ``save=True``, writes the tables above to ``output_dir`` as:

    - ``markov_dd_score_by_TF.csv``
    - ``markov_visits_by_TF.csv``
    - ``markov_screen_perturbation_rate.csv``
    """

    # Compute baseline transition matrix
    with _mute_cellrank():
        vk = cr.kernels.VelocityKernel(adata).compute_transition_matrix(show_progress_bar=False)
    vkt = vk.transition_matrix.A

    terminal_indices = np.where(adata.obs["term_states_fwd"].isin(TERMINAL_STATES))[0]
    start_indices = np.where(adata.obs[cluster_key].isin(STARTING_POINTS))[0]

    res_table = pd.DataFrame()
    visits_table = pd.DataFrame()
    visits_table["cell"] = adata.obs_names

    if tf_ko_list:
        TF_candidate = tf_ko_list
    else:
        TF_candidate = list(adata.var_names[adata.var["TF"]])

    # Run Markov simulation for each TF
    for TF in tqdm(TF_candidate, desc="Markov simulation screening for multiple TFs"):
        adata_perturb = adata_perturb_dict[TF].copy()
        adata_perturb.obs[cluster_key] = adata.obs[cluster_key].copy()

        with _mute_cellrank():
            vk_p = cr.kernels.VelocityKernel(adata_perturb).compute_transition_matrix(show_progress_bar=False)
        vkt_p = vk_p.transition_matrix.A

        total_simulations = rgv.tl.markov_density_simulation(
            adata,
            vkt,
            start_indices,
            terminal_indices,
            TERMINAL_STATES,
            method=method,
            n_steps=n_step_to_use,
            n_simulations=n_simulations,
            seed=seed,
        )

        _ = rgv.tl.markov_density_simulation(
            adata_perturb,
            vkt_p,
            start_indices,
            terminal_indices,
            TERMINAL_STATES,
            method=method,
            n_steps=n_step_to_use,
            n_simulations=n_simulations,
            seed=seed,
        )

        dd_score, dd_sig = rgv.tl.simulated_visit_diff(adata, adata_perturb, TERMINAL_STATES)

        ctrl_rate = np.sum(adata.obs["visits_dens"][~np.isnan(adata.obs["visits_dens"])])
        perturb_rate = np.sum(
            adata_perturb.obs["visits_dens"][~np.isnan(adata_perturb.obs["visits_dens"])]
        )

        # Populate visits_diff / visits_diff_smooth; only render the figure when requested.
        if plot:
            rgv.pl.simulated_visit_diff(
                adata,
                adata_perturb,
                TERMINAL_STATES,
                total_simulations,
                title=f"Density difference {TF}",
            )
        else:
            _assign_visits_diff(adata, adata_perturb, total_simulations)

        res_table.loc[TF, "ctrl_rate"] = ctrl_rate
        res_table.loc[TF, "perturb_rate"] = perturb_rate

        for i, state in enumerate(TERMINAL_STATES):
            res_table.loc[TF, f"dd_score_{state}"] = dd_score[i]
            res_table.loc[TF, f"dd_sig_{state}"] = dd_sig[i]

        # Verify required obs columns exist before accessing
        required_cols = ["visits_diff", "visits_diff_smooth"]
        for col in required_cols:
            if col not in adata.obs:
                raise KeyError(
                    f"Column '{col}' not found in adata.obs after computing visit differences. "
                    f"Available columns: {list(adata.obs.columns)}"
                )

        visits_table[f"visits_{TF}"] = adata.obs["visits"].values
        visits_table[f"visits_dens_{TF}"] = adata.obs["visits_dens"].values
        visits_table[f"visits_diff_{TF}"] = adata.obs["visits_diff"].values
        visits_table[f"visits_diff_smooth_{TF}"] = adata.obs["visits_diff_smooth"].values
        visits_table[f"visits_perturb_{TF}"] = adata_perturb.obs["visits"].values
        visits_table[f"visits_perturb_dens_{TF}"] = adata_perturb.obs["visits_dens"].values

        # Clean up per-TF obs columns before next iteration
        for col in ["visits", "visits_dens", "visits_diff", "visits_diff_smooth"]:
            del adata.obs[col]

    # Handle division by zero gracefully
    res_table["delta_success_rate"] = np.where(
        res_table["ctrl_rate"] != 0,
        res_table["perturb_rate"] / res_table["ctrl_rate"] - 1,
        np.nan
    )

    df_all = pd.DataFrame()

    for TF in TF_candidate:
        adata.obs["visits"] = visits_table[f"visits_{TF}"].values
        adata.obs["visits_dens"] = visits_table[f"visits_dens_{TF}"].values
        adata.obs["visits_diff"] = visits_table[f"visits_diff_{TF}"].values
        adata.obs["visits_diff_smooth"] = visits_table[f"visits_diff_smooth_{TF}"].values

        dd_sig_tf = np.array([
            res_table.loc[TF, f"dd_sig_{state}"] for state in TERMINAL_STATES
        ])

        df, _ = visits_diff_per_tf(adata, TERMINAL_STATES, dd_sig_tf, SIGNIFICANCE_PALETTE)
        df["Factor"] = TF

        for state in TERMINAL_STATES:
            sig = rgv.mt.get_significance(res_table.loc[TF, f"dd_sig_{state}"])
            df.loc[df["Group"] == state, "significance"] = sig

        df_all = pd.concat([df_all, df], ignore_index=True)

        for col in ["visits", "visits_dens", "visits_diff", "visits_diff_smooth"]:
            del adata.obs[col]

    # Store all result tables in adata.uns as a named dict
    adata.uns["markov_density_screening"] = {
        "dd_score_by_TF": res_table,
        "visits_by_TF": visits_table,
        "screen_perturbation_rate": df_all,
    }

    # Optionally persist the tables to CSV
    if save:
        res_table.to_csv(os.path.join(output_dir, "markov_dd_score_by_TF.csv"), sep=",")
        visits_table.to_csv(os.path.join(output_dir, "markov_visits_by_TF.csv"), sep=",")
        df_all.to_csv(os.path.join(output_dir, "markov_screen_perturbation_rate.csv"))

    logg.info("Markov simulation complete")
