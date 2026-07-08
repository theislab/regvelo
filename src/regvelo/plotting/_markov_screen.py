import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Sequence

import mplscience
import seaborn as sns
from anndata import AnnData

import cellrank as cr
import regvelo as rgv

from ._utils import SIGNIFICANCE_PALETTE

def _visits_diff_per_tf(
    adata: AnnData,
    terminal_states: Sequence[str],
    dd_sig: np.ndarray,
    sig_palette: dict,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Collect per-terminal-state visit difference values and significance colours.

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
        print(f"{ts}: {p_value}")
        palette_rel.append(sig_palette[significance])

    return pd.DataFrame(data), palette_rel


def _plot_visits_dist(
    df: pd.DataFrame,
    palette_rel: list[str],
    tick_range: float,
) -> None:
    """
    Plot a boxplot of visit difference values per terminal state for a single TF.

    Parameters
    ----------
    df : pd.DataFrame
        Long-form DataFrame with columns 'Value' and 'Group'.
    palette_rel : list of str
        Per-group hex colours derived from significance testing.
    tick_range : float
        Half-width of the x-axis range, centred at 0.5.
    """
    with mplscience.style_context():
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(3, 3))

        sns.boxplot(
            data=df,
            y="Group",
            x="Value",
            palette=palette_rel,
            ax=ax,
            flierprops={
                "marker": ".",
                "markersize": 5,
                "markerfacecolor": "black",
                "markeredgecolor": "black",
            },
        )
        ax.set_xlabel("Density change likelihood")
        ax.set_ylabel("Terminal state")

        xmin, xmax = 0.5 - tick_range, 0.5 + tick_range
        ticks = np.arange(xmin, xmax + 1e-6, tick_range)
        if 0.5 not in ticks:
            ticks = np.sort(np.append(ticks, 0.5))

        ax.set_xlim(xmin, xmax)
        ax.set_xticks(ticks)

        for spine in ax.spines.values():
            spine.set_visible(True)

        plt.show()


def _plot_visits_dist_combined(
    df: pd.DataFrame,
    terminal_states: Sequence[str],
    candidate_list: list[str],
    tick_range: float,
    sig_to_keep: list[str],
) -> None:
    """
    Plot a combined boxplot of visit differences across all knocked-out TFs,
    one panel per terminal state, filtered to a significance threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Long-form DataFrame with columns 'Value', 'Group', 'Factor', 'significance' retrieved as output from rgv.tl.TFscreening.
    terminal_states : sequence of str
        Terminal state labels; one plot is produced per state.
    tick_range : float
        Half-width of the y-axis range, centred at 0.5.
    candidate_list : list of str
        TFs to include; others are filtered out.
    sig_to_keep : list of str
        Significance labels to display (e.g. ['*', '**', '***']).
    """
    palette = SIGNIFICANCE_PALETTE

    df = df[df["Factor"].isin(candidate_list)].copy()
    df["Value"] = df["Value"].astype(float)

    for state in terminal_states:
        mask = (df["Group"] == state) & (df["significance"].isin(sig_to_keep))
        df_subset = df[mask].sort_values(by="Value", ascending=True)

        median_val = df_subset.groupby("Factor")["Value"].median()
        order = median_val.sort_values().index

        with mplscience.style_context():
            sns.set_style("whitegrid")
            fig, ax = plt.subplots(figsize=(25, 8))

            sns.boxplot(
                data=df_subset,
                y="Value",
                x="Factor",
                hue="significance",
                order=order,
                palette=palette,
                ax=ax,
                flierprops={
                    "marker": ".",
                    "markersize": 5,
                    "markerfacecolor": "black",
                    "markeredgecolor": "black",
                },
            )

            ax.set_xlabel("Factor")
            ax.set_ylabel("Density change likelihood")
            ax.set_title(f"Terminal state: {state}")
            ax.legend(
                loc="lower right",
                bbox_to_anchor=(0.5, 0.0, 0.5, 0.5),
                frameon=True,
            )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

            ymin, ymax = 0.5 - tick_range, 0.5 + tick_range
            ticks = np.arange(ymin, ymax + 1e-6, tick_range)
            if 0.5 not in ticks:
                ticks = np.sort(np.append(ticks, 0.5))

            ax.set_ylim(ymin, ymax)
            ax.set_yticks(ticks)

            for spine in ax.spines.values():
                spine.set_visible(True)
                
            plt.show()
   plt.show()
