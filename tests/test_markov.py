# Testing function for Markov screening and plotting
import anndata as ad
import numpy as np
import pandas as pd
import torch
import cellrank as cr
from scvi.data import synthetic_iid
import regvelo as rgv
from regvelo import REGVELOVI
from regvelo.plotting import plot_TF_regulon, plot_TF_success_rate, plot_visits_dist_screen
from regvelo.tools import compute_TF_regulon, markov_density_screening

# Common variables used in the test
cluster_key = "cell_type"
TERMINAL_STATES = ["mNC_head_mesenchymal"]
STARTING_POINTS = ["start"]

def test_markov():
    # create a small synthetic dataset
    adata = synthetic_iid()
    adata.layers["spliced"] = adata.X.copy()
    adata.layers["unspliced"] = adata.X.copy()

    # give deterministic var names that are guaranteed to exist
    adata.var_names = pd.Index([f"Gene{i}" for i in range(adata.n_vars)])
    n_gene = adata.n_vars

    # create a random GRN skeleton (DataFrame) and store in adata.uns
    grn_matrix = np.random.choice([0, 1], size=(n_gene, n_gene), p=[0.8, 0.2])
    W_df = pd.DataFrame(grn_matrix, index=adata.var_names, columns=adata.var_names)
    adata.uns["skeleton"] = W_df
    TF_list = adata.var_names.tolist()

    # prepare tensor version of W for the model
    W_tensor = torch.tensor(W_df.values.astype(np.float32))

    # Ensure minimal required obs columns exist so downstream code can run
    # create a simple clustering annotation and terminal state annotation
    adata.obs[cluster_key] = np.random.choice(["ctype1", "ctype2"], size=adata.n_obs)
    # default all to 'other' then mark a few cells as the terminal state used in the test
    adata.obs["term_states_fwd"] = "other"
    n_term = max(1, int(0.05 * adata.n_obs))
    adata.obs.loc[adata.obs.index[:n_term], "term_states_fwd"] = TERMINAL_STATES[0]

    # setup anndata for REGVELOVI and train a small model
    REGVELOVI.setup_anndata(adata, spliced_layer="spliced", unspliced_layer="unspliced")
    reg_vae = REGVELOVI(adata, W=W_tensor.T, regulators=TF_list)
    reg_vae.train()

    reg_vae.get_latent_representation()
    reg_vae.get_velocity()
    reg_vae.get_latent_time()

    # CellRank-based macrostate / fate probability estimation
    vk = cr.kernels.VelocityKernel(adata).compute_transition_matrix()
    estimator = cr.estimators.GPCCA(vk)
    estimator.compute_macrostates(n_states=10, cluster_key=cluster_key)
    # set terminal states by name (this mirrors how the test_regvelo pipeline uses terminal state names)
    estimator.set_terminal_states(TERMINAL_STATES)
    estimator.compute_fate_probabilities(tol=1e-5)

    MODEL = reg_vae
    adata_perturb_dict = {}

    # choose TF candidates from the actual var_names so knockouts exist
    TF_candidate = [TF_list[0], TF_list[1]]

    for TF in TF_candidate:
        adata_target_perturb, reg_vae_perturb = rgv.tl.in_silico_block_simulation(model=MODEL, adata=adata, TF=TF, cutoff=0)
        adata_perturb_dict[TF] = adata_target_perturb

    # Build per-terminal-state cell index map from baseline annotations
    ct_indices = {
        ct: adata.obs["term_states_fwd"][adata.obs["term_states_fwd"] == ct].index.tolist()
        for ct in TERMINAL_STATES
    }

    # compute macrostates and fate probabilities for each perturbed dataset
    for TF, adata_target_perturb in adata_perturb_dict.items():
        vkp = cr.kernels.VelocityKernel(adata_target_perturb).compute_transition_matrix()
        estimator = cr.estimators.GPCCA(vkp)
        estimator.compute_macrostates(n_states=10, cluster_key=cluster_key)
        estimator.set_terminal_states(ct_indices)
        estimator.compute_fate_probabilities()
        adata_perturb_dict[TF] = adata_target_perturb

    # run TF screening
    markov_density_screening(adata, adata_perturb_dict, TERMINAL_STATES=TERMINAL_STATES,  
                                    STARTING_POINTS=STARTING_POINTS, 
                                    tf_ko_list=TF_candidate,
                                    cluster_key=cluster_key, method="stepwise", n_step_to_use=500)

    plot_visits_dist_screen(adata, terminal_states=TERMINAL_STATES,
                               candidate_list=TF_candidate, tick_range=0.5)
    
    # plotting utilities (smoke checks)
    plot_TF_success_rate(adata, threshold=0.1)

    coef_targets, coef_regulators = compute_TF_regulon(adata, rgv_model = MODEL, cluster_key=cluster_key, 
                                                       TERMINAL_STATES=TERMINAL_STATES, TF=TF_candidate[0], 
                                                        threshold=0.9, n_states=10)
    plot_TF_regulon(adata, rgv_model=MODEL, cluster_key=cluster_key,
                TF=TF_candidate[0], 
                terminal_state_to_plot=TERMINAL_STATES[0],
                coef_targets=coef_targets, coef_regulators=coef_regulators,
                n_hits=10)
