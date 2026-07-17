
import numpy as np
from anndata import AnnData

from sklearn.neighbors import NearestNeighbors

# Significance-level -> colour mapping shared by significance-annotated plots.
SIGNIFICANCE_PALETTE = {
    "n.s.": "#dedede",
    "*": "#90BAAD",
    "**": "#A1E5AB",
    "***": "#ADF6B1",
}

def delta_to_probability(delta_hits: np.ndarray, k: float=1) -> np.ndarray:
    return 1 / (1 + np.exp(-k * delta_hits))

def smooth_score(adata: AnnData, key: str = "visits_dens_diff", n_neighbors: int = 10, embedding: str = "X_pca") -> None:
    # perform neighbor smoothing
    values = adata.obs[key]
    valid_cells = values[~values.isna()].index

    coords = adata.obsm[embedding]
    valid_idx = adata.obs_names.get_indexer(valid_cells)

    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(coords)
    distances, indices = nbrs.kneighbors(coords[valid_idx])

    neighbor_indices = indices[:, 1:]

    neighbor_values = values.values[neighbor_indices]
    mean_neighbor_values = np.nanmean(neighbor_values, axis=1)
    
    adata.obs[key+"_smooth"] = np.nan
    adata.obs.loc[valid_cells, key+"_smooth"] = mean_neighbor_values
