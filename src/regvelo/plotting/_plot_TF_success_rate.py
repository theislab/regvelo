import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import scanpy as sc
import scvelo as scv
import mplscience
import regvelo as rgv

def plot_TF_success_rate(
    adata,
    threshold=0.1,
    output_dir=".",
    key="markov_density_screening",
    save=False,
):
    """Rank TFs by how their knockout affects differentiation to the terminal states.

    The *success rate* measures how well cells reach the terminal states in the
    Markov density simulation: ``ctrl_rate`` is the baseline rate, and
    ``perturb_rate`` is the rate after a TF is knocked out in silico. Their
    relative change,

        ``delta_success_rate = perturb_rate / ctrl_rate - 1``,

    quantifies whether the knockout disrupts normal differentiation toward the
    terminal states. A negative value means fewer cells complete the trajectory
    after knockout (the TF is required for normal differentiation), while a
    positive value means more cells reach the terminal states (the knockout
    promotes differentiation).

    This function reads the per-TF result table from
    ``adata.uns[key]['dd_score_by_TF']`` (populated by
    :func:`rgv.tl.markov_density_screening`), computes ``delta_success_rate``,
    and draws two ranking plots: the strongest depletion hits
    (``delta_success_rate < -threshold``, differentiation impaired) and the
    strongest increased-density hits (``delta_success_rate > threshold``,
    differentiation promoted). The hit TFs shown across both plots are stored in
    ``adata.uns[key]['tf_hits']``.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData holding the screening results in ``adata.uns[key]``. Updated in
        place with ``adata.uns[key]['tf_hits']``.
    threshold : float, optional
        Magnitude cut-off on ``delta_success_rate``: a TF is a hit when
        ``delta_success_rate < -threshold`` (depletion) or
        ``delta_success_rate > threshold`` (increased density). TFs with
        ``|delta_success_rate| <= threshold`` are excluded from both plots.
        Default ``0.1``.
    output_dir : str, optional
        Directory to write the depletion/increase hit SVGs into when
        ``save=True``. Default ``"."`` (current working directory).
    key : str, optional
        Key in ``adata.uns`` under which the screening results are stored.
        Default ``"markov_density_screening"``.
    save : bool, optional
        Whether to also write the ranking figures to SVG in ``output_dir``.
        Default ``False``.

    Returns
    -------
    None
        The hit TFs are stored in ``adata.uns[key]['tf_hits']``.

    """

    res_table = adata.uns[key]["dd_score_by_TF"].copy()
    if "TF" in res_table.columns:
        res_table.index = res_table["TF"]
    else:
        res_table["TF"] = res_table.index

    # Relative change in success rate after knockout (0 = no change, <0 = depletion, >0 = increased density).
    res_table["delta_success_rate"] = res_table["perturb_rate"] / res_table["ctrl_rate"] - 1
    res_sort = res_table.sort_values(by="delta_success_rate", ascending=True)

    # Keep only the strongest depletion hits
    df_depletion = res_sort[res_sort["delta_success_rate"] < -threshold]

    with mplscience.style_context():
        sns.set_style(style="white")
        fig, ax = plt.subplots(figsize=(3, 10))
        sns.scatterplot(
            data=df_depletion, x="delta_success_rate", y="TF", palette="purple", s=200, legend=False
        )

        for _, row in df_depletion.iterrows():
            plt.hlines(
                row["TF"], xmin=0, xmax=row["delta_success_rate"],
                colors="grey", linestyles="-", alpha=0.5,
            )

        plt.xlabel("Delta success rate")
        plt.tick_params(labeltop=False, labelright=True, labelleft=False)
        plt.ylabel("")
        plt.title("Top depletion hits by success rate")

        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_color("black")
        plt.gca().spines["left"].set_visible(False)
        plt.gca().spines["bottom"].set_color("black")

        if save:
            fig.savefig(os.path.join(output_dir, "top_depletion_hits.svg"), bbox_inches="tight")
        plt.show()

    # Keep only the strongest increase rate hits
    df_increase = res_sort[res_sort["delta_success_rate"] > threshold]

    with mplscience.style_context():
        sns.set_style(style="white")
        fig, ax = plt.subplots(figsize=(3, 10))
        sns.scatterplot(
            data=df_increase, x="delta_success_rate", y="TF", palette="purple", s=200, legend=False
        )

        for _, row in df_increase.iterrows():
            plt.hlines(
                row["TF"], xmin=row["delta_success_rate"], xmax=0,
                colors="grey", linestyles="-", alpha=0.5,
            )

        plt.xlabel("Delta success rate")
        plt.tick_params(labeltop=False, labelright=True, labelleft=False)
        plt.ylabel("")
        plt.title("Top increased density hits by success rate")

        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_color("black")
        plt.gca().spines["left"].set_visible(False)
        plt.gca().spines["bottom"].set_color("black")

        if save:
            fig.savefig(os.path.join(output_dir, "top_increase_hits.svg"), bbox_inches="tight")
        plt.show()

    # Store the hit TFs shown across both ranking plots in adata.uns
    tf_hits = pd.concat([df_depletion["TF"], df_increase["TF"]]).reset_index(drop=True)
    adata.uns[key]["tf_hits"] = tf_hits
