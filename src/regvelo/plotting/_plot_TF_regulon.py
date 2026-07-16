import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import scanpy as sc
import scvelo as scv
import mplscience
import regvelo as rgv

def plot_TF_regulon(
    adata,
    rgv_model,
    cluster_key,
    TF,
    terminal_state_to_plot,
    coef_targets,
    coef_regulators,
    n_hits=10,
    device="cpu",
):
    """Plot regulon scores, GRN, and weight UMAPs for one TF and terminal states.

    Builds prior, inferred, and mixed (prior x inferred) GRNs, then for each of
    the four (GRN x edge-type) combinations draws a plot of the top
    ``n_hits`` regulatory edges for ``terminal_state`` and the corresponding
    regulatory-network diagram. Finally calls :func:`plot_grn_weight` for each
    top-hit set.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData used to train ``rgv_model``; must contain ``uns["skeleton"]``.
    rgv_model : str
        Path to a saved, trained RegVelo model directory (as produced by
        ``REGVELOVI.save``). Loaded via ``REGVELOVI.load(rgv_model, adata)``
        and used to infer the GRN.
    cluster_key : str
        ``adata.obs`` column passed to ``rgv.tl.inferred_grn`` (the GRN grouping).
    TF: list of str
        List of transcription factor of interest.
    terminal_state_to_plot : str
        The single terminal state to rank edges for. Must match a
        column name in ``coef_targets``/``coef_regulators``.
    coef_targets, coef_regulators : dict of str to DataFrame
        Per-TF target and regulator coefficient tables, as returned by
        :func:`regvelo.tl.compute_TF_regulon`.
    n_hits : int, optional
        Number of top edges to show per plot. Default ``10``.
    device : str, optional
        Device to use for model inference (e.g., "cuda:0" or "cpu"). Default ``"cpu"``.

    Returns
    -------
    None
    """

    def plot_regulon(TF, terminal_state_to_plot, GRN, target_type, n_hits):
        """Plot the top ``n_hits`` regulon edges for one terminal state.

        Parameters
        ----------
        TF : str
            Transcription factor whose regulon is plotted.
        terminal_state_to_plot : str
            Terminal-state column to rank edges by (descending score).
        GRN : pandas.DataFrame
            Gene-by-gene GRN (prior, inferred, or mixed) used to sign the edges in
            the network diagram.
        target_type : {"targets", "regulators"}
            Whether to use the targets-of-TF or regulators-of-TF coefficient
            table and which GRN orientation to use.
        n_hits : int
            Number of top edges to keep.

        Returns
        -------
        pandas.Series
            The top-hit gene names (targets if ``target_type == "targets"``,
            else regulators).
        """
        coef = coef_targets[TF] if target_type == "targets" else coef_regulators[TF]
        state_coef = coef.sort_values(by=terminal_state_to_plot, ascending=False)[:n_hits][terminal_state_to_plot]

        df = pd.DataFrame({"Gene": state_coef.index.tolist(), "Score": np.array(state_coef)})

        df[["source", "target"]] = df["Gene"].str.extract(r"\\text\{(\w+)\}.*?\\text\{(\w+)\}")

        df = df.sort_values(by="Score", ascending=False)

        with mplscience.style_context():
            sns.set_style(style="white")
            fig, ax = plt.subplots(figsize=(4, 6))
            sns.scatterplot(data=df, x="Score", y="Gene", palette="purple", s=200, legend=False)

            for _, row in df.iterrows():
                plt.hlines(
                    row["Gene"], xmin=0.5, xmax=row["Score"],
                    colors="grey", linestyles="-", alpha=0.5,
                )

            plt.xlabel("Depletion likelihood")
            plt.ylabel("")
            plt.title(f"{terminal_state_to_plot}")

            plt.gca().spines["top"].set_visible(False)
            plt.gca().spines["right"].set_visible(False)
            plt.gca().spines["left"].set_color("black")
            plt.gca().spines["bottom"].set_color("black")

        motif = []
        if target_type == "targets":
            top_hits = df["target"]
            for factor in top_hits[0:n_hits]:
                motif.append([TF, factor, GRN.loc[factor, TF]])
        if target_type == "regulators":
            top_hits = df["source"]
            for factor in top_hits[0:n_hits]:
                motif.append([factor, TF, GRN.loc[TF, factor]])

        motif = pd.DataFrame(motif)
        motif = motif[motif.iloc[:, 2] != 0]
        motif.columns = ["from", "to", "weight"]
        motif["weight"] = np.sign(motif["weight"])

        with mplscience.style_context():
            rgv.pl.regulatory_network(motif=motif)

        return top_hits

    vae = rgv.REGVELOVI.load(rgv_model, adata)
    
    GRN_prior = adata.uns["skeleton"].copy()
    GRN_infer = rgv.tl.inferred_grn(vae, adata, label=cluster_key, group="all", data_frame=True, device=device)
    GRN_mixed = GRN_prior * GRN_infer
    
    top_hits_targets_prior = plot_regulon(TF, terminal_state_to_plot, GRN_mixed, "targets", n_hits)

    top_hits_targets_infer = plot_regulon(TF, terminal_state_to_plot, GRN_infer, "targets", n_hits)

    top_hits_regulators_prior = plot_regulon(TF, terminal_state_to_plot, GRN_mixed, "regulators", n_hits)

    top_hits_regulators_infer = plot_regulon(TF, terminal_state_to_plot, GRN_infer, "regulators", n_hits)
    
    rgv.pl.plot_grn_weight(adata, vae, TF, top_hits_targets_infer, device=device)

    rgv.pl.plot_grn_weight(adata, vae, TF, top_hits_regulators_infer, device=device)
