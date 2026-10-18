import numpy as np
from anndata import AnnData
from regvelo.plotting import plot_TF_regulon, plot_TF_success_rate, plot_visits_dist_screen
from regvelo.tools import compute_TF_regulon, markov_density_screening, markov_density_simulation


def test_markov_public_api_exports():
    """Markov screening helpers should be importable from the public API."""
    for func in (
        markov_density_simulation,
        markov_density_screening,
        compute_TF_regulon,
        plot_visits_dist_screen,
        plot_TF_success_rate,
        plot_TF_regulon,
    ):
        assert callable(func)


def test_markov_density_simulation_reaches_terminal_state():
    """A deterministic two-step chain should reach the terminal cell every time."""
    adata = AnnData(np.zeros((4, 1), dtype=np.float32))
    adata.obs["term_states_fwd"] = ["other", "other", "other", "terminal"]

    transition_matrix = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    total = markov_density_simulation(
        adata,
        transition_matrix,
        start_indices=[0],
        terminal_indices=[3],
        terminal_states=["terminal"],
        n_steps=2,
        n_simulations=5,
        method="stepwise",
        seed=0,
    )

    assert total == 5
    assert adata.obs.iloc[3]["visits"] == 5
    assert adata.obs.iloc[3]["visits_dens"] == 1.0
