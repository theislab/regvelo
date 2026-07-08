import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import scanpy as sc
import scvelo as scv
import mplscience
import regvelo as rgv

SIGNIFICANCE_PALETTE = {
    "n.s.": "#dedede",
    "*": "#90BAAD",
    "**": "#A1E5AB",
    "***": "#ADF6B1",
}

def plot_top_TF(
    markov_res_df,
    adata,
    cluster_key,
    threshold=0.1,
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

    Returns
    -------
    pandas.Index
        The TF identifiers selected as top hits (``tf_hits``).

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
        
        plt.savefig(f"top_depletion_hits.svg", format="svg", bbox_inches="tight", dpi=300)
        plt.close()

    tf_hits = df["TF"]

    sc.pl.matrixplot(adata, tf_hits, groupby=cluster_key, dendrogram=False, swap_axes=True)

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

        plt.savefig(f"top_increase_hits.svg", format="svg", bbox_inches="tight")
        plt.close()

    tf_hits = df["TF"]

    sc.pl.matrixplot(adata, tf_hits, groupby=cluster_key, dendrogram=False, swap_axes=True)


def compute_TF_regulon(
    adata,
    rgv_model,
    cluster_key,
    TF,
    TERMINAL_STATES,
    threshold=0.4,
    n_states=7,
    n_samples=50
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
    rgv_model : regvelo.REGVELOVI
        Trained RegVelo model object. Its ``module.v_encoder.fc1.weight`` provides
        the inferred GRN, and it is passed to ``rgv.tl.regulation_scanning``.
        (If you instead have a saved model directory, load it first with
        ``rgv.REGVELOVI.load(path, adata)``.)
    cluster_key : str
        ``adata.obs`` column used as the cluster label during scanning
    TF : list of str
        Candidate transcription factor to scan.
    TERMINAL_STATES : list of str
        Terminal-state labels passed to ``rgv.tl.regulation_scanning``.
    threshold : float, optional
        Absolute GRN-weight cutoff for retaining an edge as a target/regulator.
        Default ``0.4``.
    n_states : int, optional
        Number of macrostates used by ``regulation_scanning``. Default ``7``.
    n_samples : int, optional
        Number of samples drawn per scan. Default ``50``.

    Returns
    ----------
    coef_targets, coef_regulators: dict of str to DataFrame
        Per-TF target and regulator coefficient tables, keyed by TF, to be
        passed directly into :func:`plot_GRN_per_TF`.

    """
    vae = rgv.REGVELOVI.load(rgv_model, adata)

    GRN = pd.DataFrame(
        vae.module.v_encoder.fc1.weight.detach().cpu().numpy(),
        index=adata.uns["skeleton"].index.tolist(),
        columns=adata.uns["skeleton"].columns.tolist(),
    )

    import sys
    
    if not sys.warnoptions:
        import warnings
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
                                                       n_states=7, 
                                                       cluster_label=cluster_key, 
                                                       terminal_states=TERMINAL_STATES, 
                                                       TF=[TF], 
                                                       target=targets, 
                                                       effect=0,
                                                       n_samples=50)
        
            coef = pd.DataFrame(np.array(perturb_screening["coefficient"]))
            coef.index = perturb_screening["links"]
            coef.columns = list(perturb_screening["coefficient"][0].keys())
            coef_targets[TF] = coef

        # Filter to top regulators only
        regulators = GRN.loc[TF,:][GRN.loc[TF,:].abs() > threshold]
        regulators = np.array(regulators.index.tolist())[np.array(regulators) != 0]

        if regulators.size == 0:
            print("No targets found - recommended to try lower threshold")
            
        else:
            n_regulators = len(regulators)
            print(f'Found {n_regulators} regulators')
            
            perturb_screening_tf = rgv.tl.regulation_scanning(model=rgv_model, 
                                                          adata=adata, 
                                                          n_states=7, 
                                                          cluster_label=cluster_key, 
                                                          terminal_states=TERMINAL_STATES, 
                                                          TF=regulators, 
                                                          target=[TF], 
                                                          effect=0,
                                                          n_samples=50)
        
            coef = pd.DataFrame(np.array(perturb_screening_tf["coefficient"]))
            coef.index = perturb_screening_tf["links"]
            coef.columns = list(perturb_screening_tf["coefficient"][0].keys())
            coef_regulators[TF] = coef

    return coef_targets, coef_regulators


def plot_grn_weight(
    adata, 
    vae, 
    TF, 
    target_list):

    """Plot cell-resolved regulatory weights for one TF against several targets.

    For each target, extracts the cell-specific GRN weight of the ``TF -> target``
    edge from ``rgv.tl.inferred_grn(..., cell_specific_grn=True)``, stores it in
    ``adata.obs``, and draws a 3-column UMAP row: TF expression, target
    expression, and the per-cell edge weight.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData with a UMAP embedding; ``var.index`` must contain ``TF`` and every
        gene in ``target_list``.
    vae : regvelo.REGVELOVI
        Trained RegVelo model, used to infer the cell-specific GRN.
    TF : str
        Transcription factor (regulator) whose outgoing edges are plotted.
    target_list : iterable of str
        Target genes to plot, one row each.
    """
    GRN = rgv.tl.inferred_grn(vae, adata, cell_specific_grn=True)

    ncols = 3
    nrows = len(target_list)
    figsize = 6
    wspace = 0.05

    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * figsize + figsize * wspace * (ncols - 1), nrows * figsize),
    )
    plt.subplots_adjust(wspace=wspace)    

    for row, target in enumerate(target_list):
        regulon = GRN[
            :, [i == target for i in adata.var.index], [i == TF for i in adata.var.index]
        ].reshape(-1)
        adata.obs[f"{TF}_{target}_weight"] = regulon

        scv.pl.umap(adata, color=[TF], frameon=False, title=[TF], ax=axs[row, 0], show=False)
        scv.pl.umap(adata, color=[target], frameon=False, title=[target], ax=axs[row, 1], show=False)
        scv.pl.umap(adata, color=[f"{TF}_{target}_weight"], cmap="Reds", ax=axs[row, 2], show=False)


def plot_GRN_per_TF(
    adata,
    rgv_model,
    cluster_key,
    TF,
    TERMINAL_STATES,
    terminal_state_to_plot,
    coef_targets,
    coef_regulators,
    n_hits=10
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
    rgv_model : regvelo.REGVELOVI
        Trained RegVelo model (also used to infer the GRN).
    cluster_key : str
        ``adata.obs`` column passed to ``rgv.tl.inferred_grn`` (the GRN grouping).
    TF: list of str
        List of transcription factor of interest.
    TERMINAL_STATES : list of str
        All terminal-state labels (kept for context / future looping).
    terminal_state_to_plot : str
        The single terminal state to rank edges for. Must match a
        column name in ``coef_targets``/``coef_regulators``.
    coef_targets, coef_regulators : dict of str to DataFrame
        Per-TF target and regulator coefficient tables, as returned by
        :func:`compute_TF_regulon`.
    n_hits : int, optional
        Number of top edges to show per plot. Default ``10``.
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

    vae = rgv.REGVELOVI.load(rgv_model, adata, accelerator="cpu")
    
    GRN_prior = adata.uns["skeleton"].copy()
    GRN_infer = rgv.tl.inferred_grn(vae, adata, label=cluster_key, group="all", data_frame=True, device="cpu")
    GRN_mixed = GRN_prior * GRN_infer
    
    top_hits_targets_prior = plot_regulon(TF, terminal_state_to_plot, GRN_mixed, "targets", n_hits)

    top_hits_targets_infer = plot_regulon(TF, terminal_state_to_plot, GRN_infer, "targets", n_hits)

    top_hits_regulators_prior = plot_regulon(TF, terminal_state_to_plot, GRN_mixed, "regulators", n_hits)

    top_hits_regulators_infer = plot_regulon(TF, terminal_state_to_plot, GRN_infer, "regulators", n_hits)
    
    plot_grn_weight(adata, vae, TF, top_hits_targets_infer)

    plot_grn_weight(adata, vae, TF, top_hits_regulators_infer)
