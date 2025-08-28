import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib.lines import Line2D
import pandas as pd
from typing import Any

def regulatory_network(
    motif: pd.DataFrame, 
    figsize: tuple[int, int] = (4, 4),
    **kwargs: Any,
    ) -> None:
    r"""Visualize a gene regulatory network (GRN) from a DataFrame.

    Parameters
    ----------
    motif
        DataFrame with regulatory edges. Expected format: ``[regulator, target, sign]``,
        where ``sign`` is ``1`` for activation and ``-1`` for inhibition.
    figsize
        Figure size for the plot. Default is ``(4, 4)``.
    **kwargs
        Additional optional parameters:

        - ``inhibition_color``: color for inhibition edges.
        - ``activation_color``: color for activation edges.
        - ``self_activation_color``: color for self-activation nodes.
        - ``self_inhibition_color``: color for self-inhibition nodes.
        - ``node_color``: color for nodes.
        - ``edge_color``: color for edges.

    Returns
    -------
    Nothing, just plots the figure.
    """

    inhibition_color = kwargs.get("inhibition_color", "black")
    activation_color = kwargs.get("activation_color", "red")
    self_activation_color = kwargs.get("self_activation_color", "red")
    self_inhibition_color = kwargs.get("self_inhibition_color", "black")
    node_color = kwargs.get("node_color", "lightblue")
    edge_color = kwargs.get("edge_color", "black")
    
    # Legend
    legend_elements = [
        Line2D([0], [0], marker=">", color=inhibition_color, label="inhibition",
               markerfacecolor=inhibition_color, markersize=8, linestyle="None"),
        Line2D([0], [0], marker=">", color=activation_color, label="activation",
               markerfacecolor=activation_color, markersize=8, linestyle="None"),
    ]

    # Style
    sns.set(style="white")
    fig, ax = plt.subplots(figsize=figsize)

    # Parse motif DataFrame
    contLines = motif.values.tolist()
    genes = set(motif.iloc[:, 0]).union(set(motif.iloc[:, 1]))
    genes = list(genes)

    G = nx.MultiDiGraph()
    G.add_nodes_from(genes)

    pEdges, nEdges = [], []
    selfActGenes, selfInhGenes = set(), set()

    for regulator, target, sign in contLines:
        if sign == 1:
            pEdges.append((regulator, target))
            if regulator == target:
                selfActGenes.add(regulator)
        elif sign == -1:
            nEdges.append((regulator, target))
            if regulator == target:
                selfInhGenes.add(regulator)
        else:
            raise ValueError(f"Unsupported regulatory relationship: {sign}")

    # Add edges
    G.add_edges_from(pEdges + nEdges)

    # Layout
    pos = nx.spring_layout(G)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_color, ax=ax)
    if selfActGenes:
        nx.draw_networkx_nodes(G, pos, nodelist=list(selfActGenes), node_color=self_activation_color, ax=ax)
    if selfInhGenes:
        nx.draw_networkx_nodes(G, pos, nodelist=list(selfInhGenes), node_color=self_inhibition_color, ax=ax)

    # Labels
    nx.draw_networkx_labels(G, pos, ax=ax)

    # Edges
    nx.draw_networkx_edges(
        G, pos, edgelist=pEdges, edge_color=activation_color,
        connectionstyle="arc3,rad=0.2", arrowsize=18, ax=ax
    )
    nx.draw_networkx_edges(
        G, pos, edgelist=nEdges, edge_color=inhibition_color,
        connectionstyle="arc3,rad=0.2", arrowsize=18, ax=ax
    )

    # Legend
    ax.legend(handles=legend_elements, bbox_to_anchor=(0.4, 0))

    plt.tight_layout()
    plt.show()