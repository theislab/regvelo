import numpy as np
import pandas as pd

from anndata import AnnData
from typing import Literal

def markov_density_simulation(
    adata: AnnData,
    T: np.ndarray, 
    start_indices: list[int], 
    terminal_indices: list[int], 
    terminal_states: list[str],
    n_steps: int = 100, 
    n_simulations: int = 200, 
    method: Literal["stepwise", "one-step"] = "stepwise",
    seed: int = 0,
    ) -> tuple[pd.Series, pd.Series]:

    """Simulate transitions on a velocity-derived Markov transition matrix.

    Parameters
    ----------
    adata
        Annotated data object.
    T
        Transition matrix of shape (n_cells, n_cells).
    start_indices
        Indices of starting cells.
    terminal_indices
        Indices of terminal cells.
    terminal_states
        Labels of terminal states corresponding to cells in `adata.obs["term_states_fwd"]`.
    n_steps
        Maximum number of steps per simulation.
    n_simulations
        Number of simulations per starting cell.
    method
        Simulation method {'stepwise', 'one-step'} to use:
        - 'stepwise': simulate trajectories step by step.
        - 'one-step': sample directly from T^n.
    seed
        Random seed for reproducibility.

    Returns
    -------
    tuple
        - pd.Series containing number of simulations that ended in each terminal cell.
        - pd.Series containing proportion of simulations that ended in each terminal cell.
    """
    np.random.seed(seed)
    
    T = np.asarray(T)
    start_indices = np.asarray(start_indices)
    terminal_indices = np.asarray(terminal_indices)
    terminal_set = set(terminal_indices)
    n_cells = T.shape[0]

    arrivals_array = np.zeros(n_cells, dtype=int)

    if method == "stepwise":
        row_sums = T.sum(axis=1)
        cum_T = np.cumsum(T, axis=1)

        for start in start_indices:
            for _ in range(n_simulations):
                current = start
                for _ in range(n_steps):
                    if row_sums[current] == 0:
                        break  # dead end
                    r = np.random.rand()
                    next_state = np.searchsorted(cum_T[current], r * row_sums[current])
                    current = next_state
                    if current in terminal_set:
                        arrivals_array[current] += 1
                        break

    elif method == "one-step":
        T_end = np.linalg.matrix_power(T, n_steps)
        for start in start_indices:
            x0 = np.zeros(n_cells)
            x0[start] = 1
            x_end = x0 @ T_end  # final distribution
            if x_end.sum() > 0:
                samples = np.random.choice(n_cells, size=n_simulations, p=x_end)
                for s in samples:
                    if s in terminal_set:
                        arrivals_array[s] += 1
            else:
                raise ValueError(f"Invalid probability distribution: x_end sums to 0 for start index {start}")
    else:
        raise ValueError("method must be 'stepwise' or 'one-step'")

    total_simulations = n_simulations * len(start_indices)
    visits = pd.Series({tid: arrivals_array[tid] for tid in terminal_indices}, dtype=int)
    visits_dens = pd.Series({tid: arrivals_array[tid] / total_simulations for tid in terminal_indices})

    adata.obs[f"visits_{method}"] = np.nan
    adata.obs[f"visits_{method}"].iloc[terminal_indices] = visits_dens

    dens_cum = []
    for ts in terminal_states:
        ts_cells = np.where(adata.obs["term_states_fwd"] == ts)[0]
        density = visits_dens.loc[ts_cells].sum()
        dens_cum.append(density)
    
    print("Proportion of simulations reaching a terminal cell", sum(dens_cum))

    return visits, visits_dens