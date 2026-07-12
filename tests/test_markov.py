#Testing function for Markov screening and plotting
import anndata as ad

from .src.tools._TFscreening import TFscreening

from .src.plotting._markov_screen import _visits_diff_per_tf, _plot_visits_dist, _plot_visits_dist_combined
from .src.plotting._driver_TF_ranking import plot_top_TF, compute_TF_regulon, plot_grn_weight,  plot_GRN_per_TF

def test_markov():
    adata = synthetic_iid()
    adata.layers["spliced"] = adata.X.copy()
    adata.layers["unspliced"] = adata.X.copy()
    adata.var_names = "Gene" + adata.var_names
    n_gene = len(adata.var_names)
  
    ## create W
    grn_matrix = np.random.choice([0, 1], size=(n_gene,n_gene), p=[0.8, 0.2]).T
    W = pd.DataFrame(grn_matrix, index=adata.var_names, columns=adata.var_names)
    adata.uns["skeleton"] = W
    TF_list = adata.var_names.tolist()

    ## training process
    W = adata.uns["skeleton"].copy()
    W = torch.tensor(np.array(W))
    REGVELOVI.setup_anndata(adata, spliced_layer="spliced", unspliced_layer="unspliced")

    ## Training the model
    reg_vae = REGVELOVI(adata,W=W.T,regulators = TF_list)
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

    TF_candidate = ["zic2a", "elf1"]

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
                       tf_ko_list = TF_candidate, 
                       cluster_key = 'cell_type', 
                       method = "stepwise", 
                       n_step_to_use = 500) 
    
    plot_top_TF(res_df, 
                adata, 
                cluster_key=['cell_type'], 
                threshold=0.1)

    coef_targets, coef_regulators = compute_TF_regulon(adata, 
                                                       MODEL, 
                                                       cluster_key='cell_type', 
                                                       TF=['zic2a'], 
                                                       TERMINAL_STATES=TERMINAL_STATES)

    plot_GRN_per_TF(adata, 
                    rgv_model, 
                    cluster_key=['cell_type'], 
                    TF=['zic2a'], 
                    TERMINAL_STATES, 
                    terminal_state_to_plot="mNC_head_mesenchymal", 
                    coef_targets=coef_targets, 
                    coef_regulators=coef_regulators, 
                    n_hits=10)
