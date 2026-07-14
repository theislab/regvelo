import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import mplscience
 
from typing import Sequence
from anndata import AnnData
from scvelo import logging as logg
from tqdm.auto import tqdm
 
import cellrank as cr
import regvelo as rgv

from ..plotting._markov_screen import _visits_diff_per_tf
from ..plotting._markov_screen import _plot_visits_dist
from ..plotting._markov_screen import _plot_visits_dist_combined
from ..plotting._utils import SIGNIFICANCE_PALETTE

def TFscreening(
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
) -> tuple[pd.DataFrame, AnnData]:

    r"""Run Markov simulations to score TF perturbation effects on cell fate density.
 
    For each TF, simulates random walks from :attr:`STARTING_POINTS` to
    :attr:`TERMINAL_STATES` in both the baseline and perturbed transition
    matrices, computes a density difference (dd) score and its significance, then
    collects per-cell visit statistics. After all TFs are processed, produces
    summary CSVs and a combined significance-filtered boxplot.
 
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
        Directory to save the result CSVs into. Default ``"."`` (current working directory).

    Returns
    -------
 
    - DataFrame of per-TF density difference scores, significance values, and baseline/perturbed absorption rates.
    - AnnData used for the simulation.
 
    Saves
    -----
 
    - ``markov_dd_score_by_TF.csv``: per-TF density difference scores and significance per terminal state.
    - ``markov_visits_by_TF.csv``: per-cell visit counts, densities, and differences for each TF.
    """
 
    # Compute baseline transition matrix
    vk = cr.kernels.VelocityKernel(adata).compute_transition_matrix()
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
 
        vk_p = cr.kernels.VelocityKernel(adata_perturb).compute_transition_matrix()
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
 
        rgv.pl.simulated_visit_diff(
            adata,
            adata_perturb,
            TERMINAL_STATES,
            total_simulations,
            title=f"Density difference {TF}",
        )
 
        df, palette_rel = _visits_diff_per_tf(
            adata, TERMINAL_STATES, dd_sig, SIGNIFICANCE_PALETTE
        )
 
        res_table.loc[TF, "ctrl_rate"] = ctrl_rate
        res_table.loc[TF, "perturb_rate"] = perturb_rate
 
        for i, state in enumerate(TERMINAL_STATES):
            res_table.loc[TF, f"dd_score_{state}"] = dd_score[i]
            res_table.loc[TF, f"dd_sig_{state}"] = dd_sig[i]
 
        visits_table[f"visits_{TF}"] = adata.obs["visits"].values
        visits_table[f"visits_dens_{TF}"] = adata.obs["visits_dens"].values
        visits_table[f"visits_diff_{TF}"] = adata.obs["visits_diff"].values
        visits_table[f"visits_diff_smooth_{TF}"] = adata.obs["visits_diff_smooth"].values
        visits_table[f"visits_perturb_{TF}"] = adata_perturb.obs["visits"].values
        visits_table[f"visits_perturb_dens_{TF}"] = adata_perturb.obs["visits_dens"].values
 
        # Clean up per-TF obs columns before next iteration
        for col in ["visits", "visits_dens", "visits_diff", "visits_diff_smooth"]:
            del adata.obs[col]
 
    res_table.to_csv(os.path.join(output_dir, "markov_dd_score_by_TF.csv"), sep=",")
    visits_table.to_csv(os.path.join(output_dir, "markov_visits_by_TF.csv"), sep=",")
 
    # Plot Markov results
    res_table["delta_success_rate"] = (
        res_table["perturb_rate"] / res_table["ctrl_rate"] - 1
    )
    res_sort = res_table.sort_values(by="delta_success_rate", ascending=True)  # noqa: F841
 
    df_all = pd.DataFrame()
 
    for TF in TF_candidate:
        adata.obs["visits"] = visits_table[f"visits_{TF}"].values
        adata.obs["visits_dens"] = visits_table[f"visits_dens_{TF}"].values
        adata.obs["visits_diff"] = visits_table[f"visits_diff_{TF}"].values
        adata.obs["visits_diff_smooth"] = visits_table[f"visits_diff_smooth_{TF}"].values
 
        dd_sig_tf = np.array([
            res_table.loc[TF, f"dd_sig_{state}"] for state in TERMINAL_STATES
        ])
 
        df, _ = _visits_diff_per_tf(adata, TERMINAL_STATES, dd_sig_tf, SIGNIFICANCE_PALETTE)
        df["Factor"] = TF
 
        for state in TERMINAL_STATES:
            sig = rgv.mt.get_significance(res_table.loc[TF, f"dd_sig_{state}"])
            df.loc[df["Group"] == state, "significance"] = sig
 
        df_all = pd.concat([df_all, df], ignore_index=True)
 
        for col in ["visits", "visits_dens", "visits_diff", "visits_diff_smooth"]:
            del adata.obs[col]
 
    df_all.to_csv(os.path.join(output_dir, "markov_screen_perturbation_rate.csv"))
 
    # Make combined plot for all terminal_states if only testing individual TF
    if len(TF_candidate) == 1:
        TF = TF_candidate[0]
        adata_perturb = adata_perturb_dict[TF].copy()
        adata_perturb.obs[cluster_key] = adata.obs[cluster_key].copy()
 
        vk_p = cr.kernels.VelocityKernel(adata_perturb).compute_transition_matrix()
        vkt_p = vk_p.transition_matrix.A
 
        total_simulations = rgv.tl.markov_density_simulation(adata,
                                                     vkt,
                                                     start_indices,
                                                     terminal_indices,
                                                     TERMINAL_STATES,
                                                     method=method,
                                                     n_steps=n_step_to_use,
                                                     n_simulations=n_simulations,
                                                     seed=seed)

        _ = rgv.tl.markov_density_simulation(adata_perturb,
                                             vkt_p,
                                             start_indices,
                                             terminal_indices,
                                             TERMINAL_STATES,
                                             method=method,
                                             n_steps=n_step_to_use,
                                             n_simulations=n_simulations,
                                             seed=seed)

        dd_score, dd_sig = rgv.tl.simulated_visit_diff(adata, adata_perturb, TERMINAL_STATES)

        rgv.pl.simulated_visit_diff(adata,
                            adata_perturb,
                            TERMINAL_STATES,
                            total_simulations,
                            title=f"Density difference {TF}")
 
        df, palette_rel = _visits_diff_per_tf(adata, TERMINAL_STATES, dd_sig, SIGNIFICANCE_PALETTE)
        _plot_visits_dist(df, palette_rel, 0.5)
 
    # Plot density likelihood for all factors per terminal state
    else:
        _plot_visits_dist_combined(
            df_all,
            TERMINAL_STATES,
            tick_range=0.5,
            candidate_list=TF_candidate,
            sig_to_keep=["*", "**", "***"],
        )
 
    logg.info("Markov simulation complete")
 
    return res_table
