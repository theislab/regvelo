import matplotlib.pyplot as plt
import cellrank as cr
from anndata import AnnData
from typing import Union, Sequence, Any

def fate_probabilities(adata : AnnData,
                       terminal_state : Union[str, Sequence[str]],
                       n_states : int,
                       save_kernel : bool = True,
                       **kwargs : Any
                       ) -> None:
    """
    Compute transition matrix and fate probabilities toward the terminal states and plot these for each of the 
    terminal states.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    terminal_state : str or Sequence[str]
        List of terminal states to compute probabilities for.
    n_states : int
        Number of states to compute probabilities for.
    save_kernel : bool
        Whether to write the kernel to adata. Default is True.
    kwargs : Any
        Optional   
        Additional keyword arguments passed to CellRank functions
    """

    macro_kwargs = {k: kwargs[k] for k in ("cluster_key", "method") if k in kwargs}
    compute_fate_probabilities_kwargs = {k: kwargs[k] for k in ("solver", "tol") if k in kwargs}
    plot_kwargs = {k: kwargs[k] for k in ("basis", "same_plot", "states", "title") if k in kwargs}

    vk = cr.kernels.VelocityKernel(adata).compute_transition_matrix()
    if save_kernel:
        vk.write_to_adata()

    estimator = cr.estimators.GPCCA(vk)
    estimator.compute_macrostates(n_states=n_states, **macro_kwargs)
    estimator.set_terminal_states(terminal_state)
    estimator.compute_fate_probabilities(**compute_fate_probabilities_kwargs)
    estimator.plot_fate_probabilities(**plot_kwargs)
    estimator.compute_lineage_drivers()
    plt.show()

