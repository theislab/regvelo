import numpy as np
import pandas as pd
import torch
from regvelo import REGVELOVI
from scvi.data import synthetic_iid


def test_regvelo():
    """Smoke-test hard and soft constraints without a long CI training run."""
    rng = np.random.default_rng(0)
    adata = synthetic_iid(batch_size=16, n_genes=12, n_proteins=0)
    adata.layers["spliced"] = adata.X.copy()
    adata.layers["unspliced"] = adata.X.copy()
    adata.var_names = pd.Index([f"Gene{i}" for i in range(adata.n_vars)])

    grn_matrix = rng.choice([0, 1], size=(adata.n_vars, adata.n_vars), p=[0.8, 0.2])
    W = pd.DataFrame(grn_matrix, index=adata.var_names, columns=adata.var_names)
    adata.uns["skeleton"] = W
    W_tensor = torch.tensor(W.values.astype(np.float32))
    regulators = adata.var_names.tolist()

    REGVELOVI.setup_anndata(adata, spliced_layer="spliced", unspliced_layer="unspliced")

    for soft_constraint in (False, True):
        model = REGVELOVI(
            adata,
            W=W_tensor.T,
            regulators=regulators,
            soft_constraint=soft_constraint,
        )
        model.train(
            max_epochs=1,
            early_stopping=False,
            enable_progress_bar=False,
        )

        latent = model.get_latent_representation()
        velocity = model.get_velocity()
        latent_time = model.get_latent_time()

        assert latent.shape[0] == adata.n_obs
        assert velocity.shape == adata.shape
        assert latent_time.shape == adata.shape
        assert np.isfinite(np.asarray(velocity)).all()
        assert np.isfinite(np.asarray(latent_time)).all()
