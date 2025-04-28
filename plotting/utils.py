import numpy as np
import matplotlib.pyplot as plt
import scvelo as scv


def calculate_entropy(prob_matrix : np.array
                      ) -> np.array:
    """
    Calculate entropy for each row in a cell fate probability matrix.

    Parameters
    ----------
    prob_matrix : np.ndarray
        A 2D NumPy array of shape (n_cells, n_lineages) where each row represents
        a cell's fate probabilities across different lineages. Each row should sum to 1.

    Returns
    -------
    entropy : np.ndarray
        A 1D NumPy array of length n_cells containing the entropy values for each cell.
    """
    log_probs = np.zeros_like(prob_matrix)
    mask = prob_matrix != 0
    np.log2(prob_matrix, where=mask, out=log_probs)

    entropy = -np.sum(prob_matrix * log_probs, axis=1)
    return entropy

