def normalise_quat(x):
    return x / x.square().sum(dim=-1).sqrt().unsqueeze(-1)
