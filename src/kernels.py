import pyro
import pyro.distributions as dist
import torch

device="cpu"

class BaseKernel:
    def __init__(self, dataset, device="cuda"):
        self.N = dataset["counts"].shape[0]
        self.F = dataset["counts"].shape[1]
        self.device = device

    @staticmethod
    def condition_covariance_eigclip(covariance, epsilon=1e-6):
        eigvals, eigvecs = torch.linalg.eigh(covariance)
        eigvals_clipped = torch.clamp(eigvals, min=epsilon)
        eigvals_clipped_diag = torch.diag_embed(eigvals_clipped)
        covariance_conditioned = eigvecs @ eigvals_clipped_diag @ eigvecs.transpose(-2, -1)
        return covariance_conditioned

    # FIXME: Really should be GP that calls this. 
    def draw_realization(self, gp, X_train, n_intervals=200):
        X_test = torch.linspace(
            X_train.min(),
            X_train.max(),
            n_intervals
        ).to(self.device).float()

        K_xx = self.compute_covariance(
            X_train,
            X_train
        )
        K_xs = self.compute_covariance(
            X_train,
            X_test
        )
        K_ss = self.compute_covariance(
            X_test,
            X_test
        )

        K_xx_inv = torch.linalg.inv(
            self.condition_covariance_eigclip(K_xx)
        )

        K_sx = K_xs.transpose(-2, -1)
        posterior_covariance = K_ss - K_sx @ K_xx_inv @ K_xs

        # Posterior
        posterior_mean = (K_sx @ K_xx_inv @ gp.u_loc.unsqueeze(-1)).squeeze(-1)
        posterior_covariance = self.condition_covariance_eigclip(posterior_covariance)

        return posterior_mean, posterior_covariance

class RBFKernel(BaseKernel):
    def __init__(self, dataset, device="cuda"):
        super().__init__(dataset=dataset, device=device)

   
    def model(self):
        # Length Scale
        lengthscale_loc = torch.ones(self.F, device=self.device) * 1.5
        lengthscale_scale = torch.ones(self.F, device=self.device) * .5
        self.lengthscale = pyro.sample(
            "rbf_lengthscale",
            dist.LogNormal(
                lengthscale_loc,
                lengthscale_scale
            ).to_event(1)
        )

        # Variance
        sigma_scale = torch.ones(self.F, device=self.device)
        self.sigma = pyro.sample(
            "rbf_sigma",
            dist.HalfNormal(
                sigma_scale
            ).to_event(1)
        )

    def guide(self):
        # Length Scale
        lengthscale_loc = pyro.param(
            "rbf_lengthscale_loc",
            torch.ones(self.F, device=self.device) * 1.5,
            constraint=dist.constraints.positive
        )
        lengthscale_scale = pyro.param(
            "rbf_lengthscale_scale",
            torch.ones(self.F, device=self.device) * .5,
            constraint=dist.constraints.positive
        )
        self.lengthscale = pyro.sample(
            "rbf_lengthscale",
            dist.LogNormal(
                lengthscale_loc,
                lengthscale_scale
            ).to_event(1)
        )

        # Variance
        sigma_scale = pyro.param(
            "rbf_sigma_scale",
            torch.ones(self.F, device=self.device),
            constraint=dist.constraints.positive
        )
        self.sigma = pyro.sample(
            "rbf_sigma",
            dist.HalfNormal(sigma_scale).to_event(1)
        )

    def compute_covariance(self, X1, X2):
        """
        Compute RBF covariance between two input sets.
        """
        diff = X1[:, None, :] - X2[None, :, :]  # (N, M, D)
        dists_sq = (diff ** 2).sum(-1)  # (N, M)

        # Assume self.lengthscale and self.sigma are already set and shape (D,)
        ls = self.lengthscale.mean()  # simplify: scalar
        sigma = self.sigma.mean()     # simplify: scalar

        cov = sigma**2 * torch.exp(-dists_sq / (2 * ls**2))
        return cov

    def compute_covariance(self, X1, X2):
        distance_squared_term = torch.cdist(
            X1[:, None], 
            X2[:, None]
        ).pow(2)
        length_scale_term = (2 * self.lengthscale[:, None, None]**2)
        variance_term = self.sigma[:, None, None]**2
        raw_covariance = variance_term * torch.exp(
            -distance_squared_term[None, :, :] / length_scale_term
        )
        return raw_covariance

    def compute_posterior_covariance(kernel, X_train, X_test, noise_variance):
        """
        Compute the posterior covariance of a GP using the RBF kernel.
        
        Args:
            kernel: An RBFKernel object with learned parameters.
            X_train: (N, D) torch.Tensor
            X_test: (M, D) torch.Tensor
            noise_variance: scalar or (N,) torch.Tensor
        
        Returns:
            posterior_covariance: (M, M) torch.Tensor
        """
        # Compute kernel matrices
        K_xx = kernel.compute_covariance(X_train, X_train)  # (N, N)
        K_xs = kernel.compute_covariance(X_train, X_test)   # (N, M)
        K_ss = kernel.compute_covariance(X_test, X_test)    # (M, M)

        # Add noise to K_xx
        K_xx_noisy = K_xx + noise_variance * torch.eye(K_xx.shape[0], device=K_xx.device)

        # Inverse
        K_xx_inv = torch.linalg.inv(K_xx_noisy)

        # Posterior covariance
        posterior_cov = K_ss - K_xs.T @ K_xx_inv @ K_xs

        return posterior_cov

    def forward(self, dataset):
        # Raw Kernel 
        covariance = self.compute_covariance(
            dataset["age_normalized"], 
            dataset["age_normalized"]
        )
        return covariance

class PeriodicKernel(BaseKernel):
    def __init__(self, dataset, device="cuda"):
        super().__init__(dataset=dataset, device=device)

    def model(self):
        lengthscale_loc = torch.ones(self.F, device=self.device) * 1.5
        lengthscale_scale = torch.ones(self.F, device=self.device) * .5
        self.lengthscale = pyro.sample(
            "periodic_lengthscale",
            dist.LogNormal(
                lengthscale_loc, 
                lengthscale_scale
            ).to_event(1)
        )

        phase_shift_loc = torch.zeros(self.F, device=self.device)
        phase_shift_kappa = torch.ones(self.F, device=self.device)
        self.phase_shift = pyro.sample(
            "periodic_phase_shift",
            dist.VonMises(
                phase_shift_loc,
                phase_shift_kappa
            ).to_event(1)
        )

        period_loc = torch.tensor(1., device=self.device)
        period_scale = torch.tensor(1., device=self.device)
        self.period = pyro.sample(
            "periodic_period",
            dist.LogNormal(
                period_loc,
                period_scale
            )
        )

        sigma_scale = torch.ones(self.F, device=self.device)
        self.sigma = pyro.sample(
            "periodic_sigma",
            dist.HalfNormal(
                sigma_scale       
            ).to_event(1)
        )

    def guide(self):
        lengthscale_loc = pyro.param(
            "periodic_lengthscale_loc",
            torch.ones(self.F, device=self.device) * 1.5,
            constraint=dist.constraints.positive
        )
        lengthscale_scale = pyro.param(
            "periodic_lengthscale_scale",
            torch.ones(self.F, device=self.device) * .5,
            constraint=dist.constraints.positive
        )
        self.lengthscale = pyro.sample(
            "periodic_lengthscale",
            dist.LogNormal(
                lengthscale_loc,
                lengthscale_scale
            ).to_event(1)
        )

        phase_shift_loc = pyro.param(
            "periodic_phase_shift_loc",
            torch.zeros(self.F, device=self.device),
            constraint=dist.constraints.interval(-torch.pi, torch.pi)
        )
        phase_shift_kappa = pyro.param(
            "periodic_phase_shift_kappa",
            torch.ones(self.F, device=self.device),
            constraint=dist.constraints.positive
        )
        self.phase_shift = pyro.sample(
            "periodic_phase_shift",
            dist.VonMises(
                phase_shift_loc,
                phase_shift_kappa
            ).to_event(1)
        )

        period_loc = pyro.param(
            "periodic_period_loc",
            torch.tensor(1.0, device=self.device), 
            constraint=dist.constraints.positive
        )
        period_scale = pyro.param(
            "periodic_period_scale",
            torch.tensor(0.5, device=self.device),
            constraint=dist.constraints.positive
        )
        self.period = pyro.sample(
            "periodic_period",
            dist.LogNormal(
                period_loc,
                period_scale
            )
        )

        sigma_scale = pyro.param(
            "periodic_sigma_scale",
            torch.ones(self.F, device=self.device),
            constraint=dist.constraints.positive
        )
        self.sigma = pyro.sample(
            "periodic_sigma",
            dist.HalfNormal(
                sigma_scale
            ).to_event(1)
        )

    def compute_covariance(self, X1, X2):
        """
        Compute the periodic kernel covariance between two sets of inputs: X1 and X2
        Returns: (F, N1, N2)
        """
        x1 = X1[:, None]  # (N1, 1)
        x2 = X2[None, :]  # (1, N2)
        diff = (x1 - x2).abs()[None, :, :]  # (1, N1, N2)
        arg = torch.pi * (diff + self.phase_shift[:, None, None]) / self.period
        raw_cov = self.sigma[:, None, None]**2 * torch.exp(-2 * (torch.sin(arg)**2) / self.lengthscale[:, None, None]**2)
        return raw_cov

    def compute_posterior_covariance(self, X_train, X_test, noise_variance):
        """
        Compute the posterior covariance matrix for GP with a periodic kernel.
        X_train, X_test: (N,) and (M,) 1D tensors
        noise_variance: scalar or (N,) tensor
        Returns: (F, M, M)
        """
        K_xx = self.compute_covariance(X_train, X_train)  # (F, N, N)
        K_xs = self.compute_covariance(X_train, X_test)   # (F, N, M)
        K_ss = self.compute_covariance(X_test, X_test)    # (F, M, M)

        eye = torch.eye(K_xx.size(-1), device=self.device).expand(K_xx.shape)
        K_xx_noisy = K_xx + noise_variance * eye  # Broadcast noise across features

        # Inversion per feature
        K_xx_inv = torch.linalg.inv(K_xx_noisy)  # (F, N, N)

        # Posterior covariance: K_ss - K_sx^T K_xx^-1 K_sx
        K_sx = K_xs.transpose(-2, -1)  # (F, M, N)
        posterior_cov = K_ss - K_sx @ K_xx_inv @ K_xs  # (F, M, M)
        return posterior_cov

    def forward(self, dataset):
        return self.compute_covariance(dataset["age_normalized"], dataset["age_normalized"])