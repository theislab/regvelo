import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import mplscience
import regvelo as rgv

def plot_regulon(TF, terminal_state_to_plot, GRN, target_type, n_hits, coef_df):
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
        coef_df : pandas.DataFrame
            Coefficient table containing the regulon scores for the TF,
            with gene pairs as index and terminal states as columns.

        Returns
        -------
        pandas.Series
            The top-hit gene names (targets if ``target_type == "targets"``,
            else regulators).
        """

        coef = coef_df[TF]
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
            plt.title(f"{terminal_state_to_plot}_{target_type}")

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
