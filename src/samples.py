class SizeFactor:
    def __init__(self, dataset, device=device):
        self.N = dataset["counts"].shape[0]
        self.F = dataset["counts"].shape[1]
        self.device = device

    def model(self):
        # Per Sample
        size_factor_loc = torch.zeros(self.N, device=self.device)
        size_factor_scale = torch.ones(self.N, device=self.device)
        self.size_factor = pyro.sample(
            "size_factor",
            dist.Normal(
                size_factor_loc,
                size_factor_scale
            ).to_event(1)
        )

    def guide(self):
        # Per Sample
        size_factor_loc = pyro.param(
            "size_factor_loc",
            torch.zeros(self.N, device=self.device)
        )
        size_factor_scale = pyro.param(
            "size_factor_scale",
            torch.ones(self.N, device=self.device),
            constraint=dist.constraints.positive
        )
        self.size_factor = pyro.sample(
            "size_factor",
            dist.Normal(
                size_factor_loc,
                size_factor_scale
            ).to_event(1)
        )