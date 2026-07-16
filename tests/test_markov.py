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

cluster_key = "cell_type"
TERMINAL_STATES = ["mNC_head_mesenchymal", "mNC_arch2", "mNC_hox34", "Pigment"]
STARTING_POINTS = ["NPB_nohox"]

def test_markov(tmp_path):
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

    rgv.tl.set_output(adata, reg_vae, n_samples=30, batch_size=adata.n_obs)

    vk = cr.kernels.VelocityKernel(adata).compute_transition_matrix()
    estimator = cr.estimators.GPCCA(vk)
    estimator.compute_macrostates(n_states=10, cluster_key=cluster_key)
    estimator.set_terminal_states(TERMINAL_STATES)
    estimator.compute_fate_probabilities(tol=1e-5)

    MODEL = str(tmp_path / "regvelo_model")
    reg_vae.save(MODEL)
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
        estimator.compute_macrostates(n_states=7, cluster_key=cluster_key)
        estimator.set_terminal_states(ct_indices)
        estimator.compute_fate_probabilities()
        adata_perturb_dict[TF] = adata_target_perturb

    rgv.tl.markov_density_screening(adata,
                         adata_perturb_dict,
                         TERMINAL_STATES,
                         STARTING_POINTS,
                         tf_ko_list=TF_candidate,
                         cluster_key=cluster_key,
                         method="stepwise",
                         n_step_to_use=100,
                         n_simulations=200)

    res_df = adata.uns["markov_density_screening"]["dd_score_by_TF"]

    # Assert that results dataframe is not empty
    assert not res_df.empty, "markov_density_screening results dataframe is empty"
    assert len(res_df) > 0, "No TF screening results were computed"

    rgv.pl.plot_TF_success_rate(adata,
                threshold=0.1)

    coef_targets, coef_regulators = rgv.tl.compute_TF_regulon(adata,
                                                       MODEL,
                                                       cluster_key=cluster_key,
                                                       TF=TF_candidate[0],
                                                       TERMINAL_STATES=TERMINAL_STATES)

    # Assert that coefficients were computed
    assert coef_targets is not None, "coef_targets is None"
    assert coef_regulators is not None, "coef_regulators is None"

    GRN_prior = adata.uns["skeleton"].copy()
    GRN_infer = rgv.tl.inferred_grn(reg_vae, adata, label=cluster_key, group="all", data_frame=True, device="cpu")

    assert GRN_prior.index.equals(GRN_infer.index) and GRN_prior.columns.equals(GRN_infer.columns)
    GRN_mixed = GRN_prior * GRN_infer

    top_hits_targets_prior = rgv.pl.plot_regulon(TF=TF_candidate[0], terminal_state_to_plot="Pigment", GRN=GRN_mixed, target_type="targets", n_hits=10, coef_df=coef_targets)
    top_hits_targets_infer = rgv.pl.plot_regulon(TF=TF_candidate[0], terminal_state_to_plot="Pigment", GRN=GRN_infer, target_type="targets", n_hits=10, coef_df=coef_targets)

    rgv.pl.plot_grn_weight(adata, reg_vae, TF=TF_candidate[0], target_list=top_hits_targets_infer, device="cpu")

    top_hits_regulators_prior = rgv.pl.plot_regulon(TF=TF_candidate[0], terminal_state_to_plot="Pigment", GRN=GRN_mixed, target_type="regulators", n_hits=10, coef_df=coef_regulators)
    top_hits_regulators_infer = rgv.pl.plot_regulon(TF=TF_candidate[0], terminal_state_to_plot="Pigment", GRN=GRN_infer, target_type="regulators", n_hits=10, coef_df=coef_regulators)

    rgv.pl.plot_grn_weight(adata, reg_vae, TF=TF_candidate[0], target_list=top_hits_regulators_infer, device="cpu")
