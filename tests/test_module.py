import torch
from regvelo._module import velocity_encoder


def test_or_activation_uses_value_equality():
    # Build the string dynamically so it is equal to, but not necessarily the
    # same object as, the literal used inside the implementation.
    activation = "".join(["O", "R"])
    assert activation == "OR"

    encoder = velocity_encoder(activate=activation, n_int=2)
    encoder.beta_mean_unconstr = torch.zeros(2)
    encoder.gamma_mean_unconstr = torch.zeros(2)

    u = torch.tensor([[0.1, 0.3]])
    s = torch.tensor([[0.2, 0.8]])

    alpha = encoder.transcription_rate(s)
    jacobian = encoder.GRN_Jacobian(s)
    du, ds = encoder(None, u, s)

    assert torch.isfinite(alpha).all()
    assert torch.isfinite(jacobian).all()
    assert torch.isfinite(du).all()
    assert torch.isfinite(ds).all()
