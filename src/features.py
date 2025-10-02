class GeneDispersion:
    def __init__(self, dataset, device=device):
        self.N = dataset["counts"].shape[0]
        self.F = dataset["counts"].shape[1]
        self.device = device

    def model(self):
        # Per Gene
        log_dispersion_loc = torch.zeros(self.F, device=self.device)
        log_dispersion_scale = torch.ones(self.F, device=self.device)
        self.log_dispersion = pyro.sample(
            "log_dispersion",
            dist.Normal(
                log_dispersion_loc,
                log_dispersion_scale
            ).to_event(1)
        )

    def guide(self):
        # Per Gene
        log_dispersion_loc = pyro.param(
            "log_dispersion_loc",
            torch.zeros(self.F, device=self.device)
        )
        log_dispersion_scale = pyro.param(
            "log_dispersion_scale",
            torch.ones(self.F, device=self.device),
            constraint=dist.constraints.positive
        )
        self.log_dispersion = pyro.sample(
            "log_dispersion",
            dist.Normal(
                log_dispersion_loc,
                log_dispersion_scale
            ).to_event(1)
        )