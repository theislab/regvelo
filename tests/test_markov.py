#Testing function for Markov screening and plotting
import anndata as ad
import numpy as np
import pandas as pd
import torch
import scanpy as sc
import scvelo as scv
import cellrank as cr
import regvelo as rgv
from regvelo import REGVELOVI

from .src.tools._TFscreening import TFscreening
from .src.plotting._markov_screen import _visits_diff_per_tf, _plot_visits_dist, _plot_visits_dist_combined
from .src.plotting._driver_TF_ranking import plot_top_TF, compute_TF_regulon, plot_grn_weight, plot_GRN_per_TF

cluster_key = "cell_type"
TERMINAL_STATES = ["mNC_head_mesenchymal", "mNC_arch2", "mNC_hox34", "Pigment"]
STARTING_POINTS = ["NPB_nohox"]

def test_markov():
    adata = rgv.datasets.zebrafish_nc()
    prior_net = rgv.datasets.zebrafish_grn()
    TF_list = adata.var_names[adata.var["is_tf"]].tolist()

    sc.pp.neighbors(adata, n_neighbors=30, n_pcs=50)
    scv.pp.moments(adata)

    adata = rgv.pp.preprocess_data(adata)
    adata = rgv.pp.set_prior_grn(adata, prior_net.T)

    W = adata.uns["skeleton"].copy()
    W = torch.tensor(np.array(W))
    REGVELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")

    ## Training the model
    reg_vae = REGVELOVI(adata, W=W.T, regulators=TF_list)
    reg_vae.train()

    reg_vae.get_latent_representation()
    reg_vae.get_velocity()
    reg_vae.get_latent_time()

    vk = cr.kernels.VelocityKernel(adata).compute_transition_matrix()
    estimator = cr.estimators.GPCCA(vk)
    estimator.compute_macrostates(n_states=10, cluster_key=cluster_key)
    estimator.set_terminal_states(TERMINAL_STATES)
    estimator.compute_fate_probabilities(tol=1e-5)

    MODEL = reg_vae
    adata_perturb_dict = {}

    TF_candidate = ["nr2f5", "elf1"]

    for TF in TF_candidate:
        adata_target_perturb, reg_vae_perturb = rgv.tl.in_silico_block_simulation(model=MODEL, adata=adata, TF=TF, cutoff=0)
        adata_perturb_dict[TF] = adata_target_perturb

    # Build per-terminal-state cell index map from baseline annotations
    ct_indices = {
        ct: adata.obs["term_states_fwd"][adata.obs["term_states_fwd"] == ct].index.tolist()
        for ct in TERMINAL_STATES}

    for TF, adata_target_perturb in adata_perturb_dict.items():
        vkp = cr.kernels.VelocityKernel(adata_target_perturb).compute_transition_matrix()
        estimator = cr.estimators.GPCCA(vkp)
        estimator.compute_macrostates(n_states=10, cluster_key=cluster_key)
        estimator.set_terminal_states(ct_indices)
        estimator.compute_fate_probabilities()
        adata_perturb_dict[TF] = adata_target_perturb

    res_df = TFscreening(adata,
                         adata_perturb_dict,
                         TERMINAL_STATES,
                         STARTING_POINTS,
                         tf_ko_list=TF_candidate,
                         cluster_key=cluster_key,
                         method="stepwise",
                         n_step_to_use=500)

    # Assert that results dataframe is not empty
    assert not res_df.empty, "TFscreening results dataframe is empty"
    assert len(res_df) > 0, "No TF screening results were computed"

    plot_top_TF(res_df,
                adata,
                cluster_key=cluster_key,
                threshold=0.1)

    coef_targets, coef_regulators = compute_TF_regulon(adata,
                                                       MODEL,
                                                       cluster_key=cluster_key,
                                                       TF=TF_candidate[0],
                                                       TERMINAL_STATES=TERMINAL_STATES)

    # Assert that coefficients were computed
    assert coef_targets is not None, "coef_targets is None"
    assert coef_regulators is not None, "coef_regulators is None"

    plot_GRN_per_TF(adata,
                    MODEL,
                    cluster_key=cluster_key,
                    TF=TF_candidate[0],
                    terminal_states=TERMINAL_STATES,
                    terminal_state_to_plot=TERMINAL_STATES[0],
                    coef_targets=coef_targets,
                    coef_regulators=coef_regulators,
                    n_hits=10)
