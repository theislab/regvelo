import numpy as np
from anndata import AnnData

from .._model import REGVELOVI


# Code mostly taken from veloVI reproducibility repo
# https://yoseflab.github.io/velovi_reproducibility/estimation_comparison/simulation_w_inferred_rates.html


def set_output(
    adata: AnnData, 
    vae: REGVELOVI, 
    n_samples: int = 30, 
    batch_size: int = None
    ) -> None:
    r"""Add inference results from a trained RegVelo model to ``adata``.

    Parameters
    ----------
    adata
        Annotated data matrix.
    vae
        Trained :class:`REGVELOVI` model.
    n_samples
        Number of posterior samples to use for estimation.
    batch_size
        Minibatch size for data loading into model. If None, uses ``scvi.settings.batch_size``.

    Returns
    -------
    Nothing, just updates ``adata`` with the following fields:
    
    - ``adata.layers["velocity"]``: Estimated velocity.
    - ``adata.layers["latent_time_velovi"]``: Estimated latent time.
    - ``adata.var["fit_alpha"]``: Estimated alpha rate.
    - ``adata.var["fit_beta"]``: Estimated beta rate.
    - ``adata.var["fit_gamma"]``: Estimated gamma rate.
    - ``adata.layers["fit_t"]``: Estimated latent time scaled to the maximum value
    - ``adata.var["fit_scaling"]``: Scaling factor for the latent time.
    """
    
    latent_time = vae.get_latent_time(n_samples=n_samples, batch_size=batch_size)
    velocities = vae.get_velocity(n_samples=n_samples, batch_size=batch_size)

    t = latent_time.values
    scaling = 20 / t.max(0)

    adata.layers["velocity"] = velocities / scaling
    adata.layers["latent_time_velovi"] = latent_time

    rates = vae.get_rates()
    if "alpha" in rates:
        adata.var["fit_alpha"] = rates["alpha"] / scaling
    adata.var["fit_beta"] = rates["beta"] / scaling
    adata.var["fit_gamma"] = rates["gamma"] / scaling

    adata.layers["fit_t"] = latent_time * scaling[np.newaxis, :]
    adata.var["fit_scaling"] = 1.0