import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Sequence

import mplscience
import seaborn as sns
from anndata import AnnData
import regvelo as rgv

from ._utils import SIGNIFICANCE_PALETTE

def plot_visits_dist_screen(
    adata: AnnData,
    terminal_states: Sequence[str],
    candidate_list: list[str],
    tick_range: float,
    sig_to_keep: list[str] = ["n.s.", "*", "**", "***"],
    figsize: tuple[float, float] | None = None,
    key: str = "markov_density_screening",
) -> None:
    """
    Plot a combined boxplot of visit differences across all knocked-out TFs,
    one panel per terminal state, filtered to a significance threshold.

    Parameters
    ----------
    adata : AnnData
        AnnData holding the screening results in ``adata.uns[key]``. The long-form
        table ``adata.uns[key]['screen_perturbation_rate']`` (columns 'Value',
        'Group', 'Factor', 'significance') is populated by
        :func:`rgv.tl.markov_density_screening`.
    terminal_states : sequence of str
        Terminal state labels; one plot is produced per state.
    tick_range : float
        Half-width of the y-axis range, centred at 0.5.
    candidate_list : list of str
        TFs to include; others are filtered out.
    sig_to_keep : list of str
        Significance labels to display. Defaults to all levels
        (``['n.s.', '*', '**', '***']``), i.e. every factor is kept.
    figsize : tuple of float, optional
        Figure size ``(width, height)`` in inches. If ``None`` (default), the width
        adapts to the number of factors shown in each panel (one box per factor),
        clamped to a sensible range; pass a tuple to set it manually.
    key : str, optional
        Key in ``adata.uns`` under which the screening results are stored.
        Default ``"markov_density_screening"``.
    """
    palette = SIGNIFICANCE_PALETTE

    df = adata.uns[key]["screen_perturbation_rate"]
    df = df[df["Factor"].isin(candidate_list)].copy()
    df["Value"] = df["Value"].astype(float)

    for state in terminal_states:
        mask = (df["Group"] == state) & (df["significance"].isin(sig_to_keep))
        df_subset = df[mask].sort_values(by="Value", ascending=True)

        median_val = df_subset.groupby("Factor")["Value"].median()
        order = median_val.sort_values().index

        if figsize is None:
            # One box per factor: scale width with the number of factors shown.
            n_factors = len(order)
            fig_size = (float(np.clip(0.3 * n_factors, 5, 25)), 5)
        else:
            fig_size = figsize

        with mplscience.style_context():
            sns.set_style("whitegrid")
            fig, ax = plt.subplots(figsize=fig_size)

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
