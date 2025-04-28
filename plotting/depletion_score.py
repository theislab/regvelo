
import torch
import pandas as pd
from anndata import AnnData
from typing import Union, Sequence, Any, Optional
import cellrank as cr

import seaborn as sns
import matplotlib.pyplot as plt


from ..tools.abundance_test import abundance_test

def depletion_score(adata : AnnData,
                    df : pd.DataFrame,
                    color_label : str = "celltype_cluster",
                    **kwargs : Any,
                    ) -> None:
    """
    Plot depletion scores.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix of original model.
    df : pandas.DataFrame
    color_label : str
        Used for color palette
    kwargs : Any
        Optional   
        Additional keyword arguments passed to CellRank and plot functions.
    """

    fontsize = kwargs.get("fontsize", 14)
    figsize = kwargs.get("figsize", (12, 6))

    legend_loc = kwargs.get("legend_loc", "center left")
    legend_bbox = kwargs.get("legend_bbox", (1.02, 0.5))

    xlabel = kwargs.get("xlabel", "TF")
    ylabel = kwargs.get("ylabel", "Depletion score")

    plot_kwargs = {k: kwargs[k] for k in ("ax",) if k in kwargs}

    plt.figure(figsize=figsize)

    palette = dict(zip(adata.obs[color_label].cat.categories, adata.uns[f"{color_label}_colors"]))
    sns.barplot(x='TF', y='Depletion score', hue='Terminal state', data=df, palette=palette, dodge=True, **plot_kwargs)

    for i in range(len(df['TF'].unique()) - 1):
        plt.axvline(x=i + 0.5, color='gray', linestyle='--')

    plt.ylabel(ylabel, fontsize=fontsize)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.xticks(fontsize=fontsize)  
    plt.yticks(fontsize=fontsize)  

    plt.legend(
        title='Terminal state',
        bbox_to_anchor=legend_bbox, 
        loc=legend_loc,
        borderaxespad=0
    )


    plt.tight_layout()
    plt.show()
