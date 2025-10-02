import pyro
import pyro.distributions as dist
import torch

device="cpu"

class GaussianProcess:

    def __init__(self, dataset, device=device):
        self.N = dataset["counts"].shape[0]
        self.F = dataset["counts"].shape[1]
        self.device = device

    def model(self):
        u_loc_loc = torch.zeros((self.F, self.N), device=self.device)
        u_loc_scale = torch.ones((self.F, self.N), device=self.device)
        self.u_loc = pyro.sample(
            "u_loc",
            dist.Normal(
                u_loc_loc,
                u_loc_scale,
            ).to_event(2)
        )

    def guide(self):
        u_loc_loc = pyro.param(
            "u_loc_loc",
            torch.zeros((self.F, self.N), device=self.device)
        )
        u_loc_scale = pyro.param(
            "u_loc_scale",
            torch.ones((self.F, self.N), device=self.device),
            constraint=dist.constraints.positive
        )
        self.u_loc = pyro.sample(
            "u_loc",
            dist.Normal(
                u_loc_loc,
                u_loc_scale,
            ).to_event(2)
        )