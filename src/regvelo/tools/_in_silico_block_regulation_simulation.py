import numpy as np
import torch
from anndata import AnnData

from .._model import REGVELOVI

def in_silico_block_regulation_simulation(
    model: str,
    adata: AnnData,
    regulator: str,
    target: str,
    n_samples: int = 30,
    effects: float = 0.0,
) -> AnnData:
    r"""Simulate in-silico blocking of a specific regulation between a regulator and a target gene.

    Parameters
    ----------
    model
        Path to the pretrained :class:`REGVELOVI` model.
    adata
        Annotated data matrix containing gene expression and cell-specific information.
    regulator
        Name of the regulator gene whose effect on the target gene will be blocked.
    target
        Name of the target gene affected by the regulator.
    n_samples
        Number of samples to generate during the simulation. Default is ``100``.
    effects
        Effect value to set for the (target, regulator) edge. Use ``0.0`` for a complete block.

    Returns
    -------
    AnnData
        Updated :class:`AnnData` object with perturbation results added as new outputs.
    """
    # Load model and clone GRN weights
    reg_vae_perturb = REGVELOVI.load(model, adata)
    perturb_GRN = reg_vae_perturb.module.v_encoder.fc1.weight.detach().clone()
    perturb_GRN[[i == target for i in adata.var.index], [i == regulator for i in adata.var.index]] = effects
    
    reg_vae_perturb.module.v_encoder.fc1.weight.data = perturb_GRN
    adata_target_perturb = reg_vae_perturb.add_regvelo_outputs_to_adata(adata=adata, n_samples=n_samples)

    return adata_target_perturb