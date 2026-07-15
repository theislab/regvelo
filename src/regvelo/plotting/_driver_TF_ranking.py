import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import scanpy as sc
import scvelo as scv
import mplscience
import regvelo as rgv

def plot_top_TF(
    markov_res_df,
    adata,
    cluster_key,
    threshold=0.1,
    output_dir=".",
):
    """Rank TFs by their knockout effect and plot the top depletion hits.

    Computes a ``delta_success_rate`` per TF as ``perturb_rate / ctrl_rate - 1``
    (the relative change in trajectory success rate after in silico knockout),
    selects TFs whose value falls below ``threshold`` (i.e. the strongest
    depletion hits), draws a plot of those hits, and shows a
    scanpy matrixplot of their expression across ``cluster_key`` groups.

    Parameters
    ----------
    markov_res_df : pandas.DataFrame or str
        Per-TF result table obtained from rgv.tl.simulated_visit_diff or a path to a saved CSV result. Must contain columns
        ``perturb_rate`` and ``ctrl_rate``. The TF identifier is taken from the
        index.
    adata : anndata.AnnData
        AnnData whose ``var_names`` include the selected TFs; used for the
        per-cluster expression matrixplot.
    cluster_key : str
        Column in ``adata.obs`` to group cells by in the matrixplot
        (e.g. ``"stage"`` or a cell-type column).
    threshold : float, optional
        Upper bound on ``delta_success_rate`` for a TF to be shown. More negative
        values are stronger depletion hits. Default ``0.1``.
    output_dir : str, optional
        Directory to save the depletion/increase hit SVGs into. Default ``"."``
        (current working directory).

    Returns
    -------
    None

    """
    
    if isinstance(markov_res_df, str):
        markov_res_df = pd.read_csv(markov_res_df, delimiter=",")

    res_table = markov_res_df.rename(columns={"Unnamed: 0": "TF"}).copy()
    if "TF" in res_table.columns:
        res_table.index = res_table["TF"]
    else:
        res_table["TF"] = res_table.index

    # Relative change in success rate after knockout (0 = no change, <0 = depletion, >0 = increased density).
    res_table["delta_success_rate"] = res_table["perturb_rate"] / res_table["ctrl_rate"] - 1
    res_sort = res_table.sort_values(by="delta_success_rate", ascending=True)

    # Keep only the strongest depletion hits
    df = res_sort[res_sort["delta_success_rate"] < threshold]

    with mplscience.style_context():
        sns.set_style(style="white")
        fig, ax = plt.subplots(figsize=(3, 10))
        sns.scatterplot(
            data=df, x="delta_success_rate", y="TF", palette="purple", s=200, legend=False
        )

        for _, row in df.iterrows():
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

    tf_hits = df["TF"]

    # Keep only the strongest increase rate hits
    df = res_sort[res_sort["delta_success_rate"] > threshold]

    with mplscience.style_context():
        sns.set_style(style="white")
        fig, ax = plt.subplots(figsize=(3, 10))
        sns.scatterplot(
            data=df, x="delta_success_rate", y="TF", palette="purple", s=200, legend=False
        )

        for _, row in df.iterrows():
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

    tf_hits = df["TF"]
