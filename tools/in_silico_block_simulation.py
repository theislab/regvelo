import torch
import pandas as pd
import numpy as np

from anndata import AnnData

import os, shutil

from .._model import REGVELOVI

def in_silico_block_simulation(
        model : str,
        adata : AnnData,
        TF : str,
        effects : int = 0,
        cutoff : int = 1e-3,
        customized_GRN : torch.Tensor = None
        ) -> tuple:
    """ Perform in silico TF regulon knock-out

    Parameters
    ----------
    model
        The saved address for the RegVelo model.
    adata
        Anndata objects.
    TF
        The candidate TF, need to knockout its regulon.
    effect
        The coefficient for replacing the weights in GRN
    cutoff
        The threshold for determing which links need to be muted,
    customized_GRN
        The customized perturbed GRN
    """

    reg_vae_perturb = REGVELOVI.load(model,adata)

    perturb_GRN = reg_vae_perturb.module.v_encoder.fc1.weight.detach().clone()

    if customized_GRN is None:
        perturb_GRN[(perturb_GRN[:,[i == TF for i in adata.var.index]].abs()>cutoff).cpu().numpy().reshape(-1),[i == TF for i in adata.var.index]] = effects
        reg_vae_perturb.module.v_encoder.fc1.weight.data = perturb_GRN
    else:
        device = perturb_GRN.device
        customized_GRN =  customized_GRN.to(device)
        reg_vae_perturb.module.v_encoder.fc1.weight.data = customized_GRN
    
    adata_target_perturb = reg_vae_perturb.add_regvelo_outputs_to_adata(adata = adata)
    
    return adata_target_perturb, reg_vae_perturb