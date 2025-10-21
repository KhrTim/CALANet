"""
Generalized Time Warping Invariant Dictionary Learning (GTWIDL)
Implementation based on the paper:
"Generalized Time Warping Invariant Dictionary Learning for Time Series Classification and Clustering"
https://arxiv.org/abs/2306.17690
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List, Callable
from scipy import interpolate


class BasisFunctions:
    """Continuous monotonic basis functions for time warping"""

    @staticmethod
    def polynomial(t: torch.Tensor, degree: int = 3) -> torch.Tensor:
        """Polynomial basis functions: t^1, t^2, ..., t^degree"""
        basis = []
        for d in range(1, degree + 1):
            basis.append(t ** d)
        return torch.stack(basis, dim=-1)

    @staticmethod
    def exponential(t: torch.Tensor, n_basis: int = 5) -> torch.Tensor:
        """Exponential basis functions: exp(k*t) for different k values"""
        k_values = torch.linspace(0.1, 2.0, n_basis, device=t.device)
        basis = []
        for k in k_values:
            basis.append(torch.exp(k * t))
        return torch.stack(basis, dim=-1)

    @staticmethod
    def logarithmic(t: torch.Tensor, n_basis: int = 5) -> torch.Tensor:
        """Logarithmic basis functions: log(1 + k*t)"""
        k_values = torch.linspace(0.5, 5.0, n_basis, device=t.device)
        basis = []
        for k in k_values:
            basis.append(torch.log(1 + k * t))
        return torch.stack(basis, dim=-1)

    @staticmethod
    def hyperbolic_tangent(t: torch.Tensor, n_basis: int = 5) -> torch.Tensor:
        """Hyperbolic tangent basis functions: tanh(k*(t-c))"""
        k_values = torch.linspace(1.0, 5.0, n_basis, device=t.device)
        c_values = torch.linspace(0.2, 0.8, n_basis, device=t.device)
        basis = []
        for k, c in zip(k_values, c_values):
            basis.append(torch.tanh(k * (t - c)))
        return torch.stack(basis, dim=-1)

    @staticmethod
    def ispline(t: torch.Tensor, n_basis: int = 5, order: int = 3) -> torch.Tensor:
        """I-spline basis functions (integrated B-splines, monotonic)"""
        # Convert to numpy for scipy
        t_np = t.cpu().numpy()

        # Create knots
        n_knots = n_basis - order + 1
        knots = np.linspace(0, 1, n_knots + 2)[1:-1]

        # Add boundary knots
        knots = np.concatenate([
            np.zeros(order),
            knots,
            np.ones(order)
        ])

        basis_list = []
        for i in range(n_basis):
            # Create B-spline
            bspline = interpolate.BSpline.basis_element(knots[i:i+order+2], extrapolate=False)

            # Integrate to get I-spline (monotonic)
            ispline_vals = []
            for t_val in t_np.flatten():
                if t_val <= 0:
                    ispline_vals.append(0.0)
                elif t_val >= 1:
                    ispline_vals.append(1.0)
                else:
                    # Numerical integration
                    from scipy.integrate import quad
                    val, _ = quad(lambda x: bspline(x) if bspline(x) is not None else 0,
                                  0, t_val, limit=50)
                    ispline_vals.append(val)

            basis_list.append(torch.tensor(ispline_vals, dtype=t.dtype, device=t.device).reshape(t.shape))

        return torch.stack(basis_list, dim=-1)


class GeneralizedTimeWarping:
    """Generalized time warping operator using continuous basis functions"""

    def __init__(self,
                 length: int,
                 n_basis: int = 5,
                 basis_type: str = 'polynomial',
                 device: str = 'cpu'):
        """
        Args:
            length: Length of the time series
            n_basis: Number of basis functions
            basis_type: Type of basis functions ('polynomial', 'exponential', 'logarithmic', 'tanh', 'ispline')
            device: Device to run on
        """
        self.length = length
        self.n_basis = n_basis
        self.basis_type = basis_type
        self.device = device

        # Normalized time points [0, 1]
        self.t = torch.linspace(0, 1, length, device=device)

        # Generate basis matrix Q
        self.Q = self._generate_basis_matrix()

    def _generate_basis_matrix(self) -> torch.Tensor:
        """Generate the basis matrix Q of monotonic functions"""
        if self.basis_type == 'polynomial':
            basis = BasisFunctions.polynomial(self.t, degree=self.n_basis)
        elif self.basis_type == 'exponential':
            basis = BasisFunctions.exponential(self.t, n_basis=self.n_basis)
        elif self.basis_type == 'logarithmic':
            basis = BasisFunctions.logarithmic(self.t, n_basis=self.n_basis)
        elif self.basis_type == 'tanh':
            basis = BasisFunctions.hyperbolic_tangent(self.t, n_basis=self.n_basis)
        elif self.basis_type == 'ispline':
            basis = BasisFunctions.ispline(self.t, n_basis=self.n_basis)
        else:
            raise ValueError(f"Unknown basis type: {self.basis_type}")

        # Add linear component for baseline
        linear = self.t.unsqueeze(-1)
        Q = torch.cat([linear, basis], dim=-1)

        return Q

    def generate_warping_path(self, beta: torch.Tensor) -> torch.Tensor:
        """
        Generate warping path from basis coefficients

        Args:
            beta: Basis coefficients (n_basis + 1,) - non-negative

        Returns:
            p: Warping path (length,) - monotonically increasing
        """
        # Ensure non-negativity
        beta = torch.clamp(beta, min=0)

        # Generate path: p = Q @ beta
        p = torch.matmul(self.Q, beta)

        # Ensure monotonicity and boundary constraints
        p = torch.cumsum(torch.nn.functional.relu(torch.diff(p, prepend=torch.tensor([0.0], device=self.device))), dim=0)

        # Normalize to [0, length-1]
        p = p / (p[-1] + 1e-8) * (self.length - 1)

        return p

    def warp_time_series(self, x: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        Warp a time series using the warping path defined by beta

        Args:
            x: Time series (length,) or (length, n_dims)
            beta: Basis coefficients

        Returns:
            x_warped: Warped time series
        """
        p = self.generate_warping_path(beta)

        # Interpolate using the warping path
        if x.ndim == 1:
            x_warped = self._interpolate_1d(x, p)
        else:
            x_warped = torch.stack([self._interpolate_1d(x[:, i], p) for i in range(x.shape[1])], dim=1)

        return x_warped

    def _interpolate_1d(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Linear interpolation for 1D signal"""
        length = x.shape[0]

        # Clamp indices to valid range
        indices = torch.clamp(indices, 0, length - 1)

        # Get integer and fractional parts
        indices_floor = torch.floor(indices).long()
        indices_ceil = torch.ceil(indices).long()

        # Handle boundary
        indices_ceil = torch.clamp(indices_ceil, 0, length - 1)

        # Linear interpolation
        frac = indices - indices_floor.float()
        x_warped = x[indices_floor] * (1 - frac) + x[indices_ceil] * frac

        return x_warped


class GTWIDL:
    """Generalized Time Warping Invariant Dictionary Learning"""

    def __init__(self,
                 n_atoms: int,
                 atom_length: int,
                 n_basis: int = 5,
                 basis_type: str = 'polynomial',
                 lambda_sparse: float = 0.1,
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 device: str = 'cpu',
                 verbose: bool = True):
        """
        Args:
            n_atoms: Number of dictionary atoms
            atom_length: Length of each dictionary atom
            n_basis: Number of basis functions for time warping
            basis_type: Type of basis functions
            lambda_sparse: Sparsity regularization parameter
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            device: Device to run on
            verbose: Print training progress
        """
        self.n_atoms = n_atoms
        self.atom_length = atom_length
        self.n_basis = n_basis
        self.basis_type = basis_type
        self.lambda_sparse = lambda_sparse
        self.max_iter = max_iter
        self.tol = tol
        self.device = device
        self.verbose = verbose

        # Initialize dictionary randomly
        self.dictionary = torch.randn(atom_length, n_atoms, device=device)
        self.dictionary = self.dictionary / torch.norm(self.dictionary, dim=0, keepdim=True)

        # Time warping operator
        self.time_warping = GeneralizedTimeWarping(atom_length, n_basis, basis_type, device)

    def fit(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Learn dictionary from time series data

        Args:
            X: Time series data (n_samples, length) or (n_samples, length, n_dims)

        Returns:
            dictionary: Learned dictionary atoms
            alphas: Sparse coefficients for each sample
            betas: Warping path coefficients for each sample
        """
        n_samples = X.shape[0]
        series_length = X.shape[1]

        # Handle multi-dimensional time series
        if X.ndim == 3:
            n_dims = X.shape[2]
        else:
            n_dims = 1
            X = X.unsqueeze(-1)

        # Re-initialize dictionary with proper dimensions for multivariate data
        if n_dims > 1:
            self.dictionary = torch.randn(self.atom_length, self.n_atoms, n_dims, device=self.device)
            self.dictionary = self.dictionary / torch.norm(self.dictionary.reshape(self.atom_length, self.n_atoms, -1), dim=0, keepdim=True)
        else:
            # Keep 2D for univariate
            if self.dictionary.ndim == 3:
                self.dictionary = self.dictionary.squeeze(-1)

        # Initialize sparse coefficients and warping coefficients
        alphas = torch.rand(n_samples, self.n_atoms, device=self.device) * 0.1
        betas = [torch.ones(self.n_basis + 1, device=self.device) * 0.1 for _ in range(n_samples)]

        prev_loss = float('inf')

        for iteration in range(self.max_iter):
            # Step 1: Update sparse coefficients (alpha)
            alphas = self._update_sparse_coefficients(X, betas)

            # Step 2: Update warping coefficients (beta)
            betas = self._update_warping_coefficients(X, alphas, betas)

            # Step 3: Update dictionary atoms
            self.dictionary = self._update_dictionary(X, alphas, betas)

            # Compute loss
            loss = self._compute_loss(X, alphas, betas)

            if self.verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}/{self.max_iter}, Loss: {loss:.6f}")

            # Check convergence
            if abs(prev_loss - loss) < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {iteration}")
                break

            prev_loss = loss

        return self.dictionary, alphas, betas

    def _update_sparse_coefficients(self, X: torch.Tensor, betas: List[torch.Tensor]) -> torch.Tensor:
        """Update sparse coefficients using FISTA or coordinate descent"""
        n_samples = X.shape[0]
        alphas = torch.zeros(n_samples, self.n_atoms, device=self.device)

        for i in range(n_samples):
            x_i = X[i]
            beta_i = betas[i]

            # Warp dictionary atoms
            warped_dict = []
            for k in range(self.n_atoms):
                # Handle both 2D (univariate) and 3D (multivariate) dictionaries
                if self.dictionary.ndim == 3:
                    atom = self.dictionary[:, k, :]  # (length, n_dims)
                else:
                    atom = self.dictionary[:, k]  # (length,)
                warped_atom = self.time_warping.warp_time_series(atom, beta_i)
                warped_dict.append(warped_atom)
            warped_dict = torch.stack(warped_dict, dim=1)  # (length, n_atoms, n_dims) or (length, n_atoms)

            # Solve sparse coding: min ||x - D*alpha||^2 + lambda*||alpha||_1
            alpha_i = self._soft_threshold_lasso(x_i, warped_dict, self.lambda_sparse)
            alphas[i] = alpha_i

        return alphas

    def _soft_threshold_lasso(self, x: torch.Tensor, D: torch.Tensor, lambda_val: float, max_iter: int = 50) -> torch.Tensor:
        """Solve LASSO using iterative soft thresholding"""
        n_atoms = D.shape[1]
        alpha = torch.zeros(n_atoms, device=self.device)

        # Flatten for easier computation
        x_flat = x.flatten()
        D_flat = D.reshape(-1, n_atoms)

        # Compute Lipschitz constant
        L = torch.linalg.norm(D_flat.T @ D_flat, ord=2)
        step_size = 1.0 / (L + 1e-8)

        for _ in range(max_iter):
            # Gradient step
            residual = x_flat - D_flat @ alpha
            grad = -D_flat.T @ residual
            alpha = alpha - step_size * grad

            # Soft thresholding
            alpha = torch.sign(alpha) * torch.clamp(torch.abs(alpha) - step_size * lambda_val, min=0)

        return alpha

    def _update_warping_coefficients(self, X: torch.Tensor, alphas: torch.Tensor,
                                    betas: List[torch.Tensor]) -> List[torch.Tensor]:
        """Update warping path coefficients"""
        n_samples = X.shape[0]
        new_betas = []

        for i in range(n_samples):
            x_i = X[i]
            alpha_i = alphas[i]
            beta_i = betas[i]

            # Optimize beta using gradient descent
            beta_i_new = self._optimize_beta(x_i, alpha_i, beta_i)
            new_betas.append(beta_i_new)

        return new_betas

    def _optimize_beta(self, x: torch.Tensor, alpha: torch.Tensor, beta_init: torch.Tensor,
                       max_iter: int = 20, lr: float = 0.01) -> torch.Tensor:
        """Optimize warping coefficients for a single sample"""
        beta = beta_init.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([beta], lr=lr)

        for _ in range(max_iter):
            optimizer.zero_grad()

            # Reconstruct signal
            reconstruction = torch.zeros_like(x)
            has_atoms = False
            for k in range(self.n_atoms):
                if alpha[k] > 1e-6:
                    has_atoms = True
                    # Handle both 2D (univariate) and 3D (multivariate) dictionaries
                    if self.dictionary.ndim == 3:
                        atom = self.dictionary[:, k, :]  # (length, n_dims)
                    else:
                        atom = self.dictionary[:, k]  # (length,)
                    warped_atom = self.time_warping.warp_time_series(atom, torch.clamp(beta, min=0))
                    reconstruction = reconstruction + alpha[k] * warped_atom

            # Skip if no atoms contribute (no gradient path)
            if not has_atoms:
                continue

            # Compute loss
            loss = torch.mean((x - reconstruction) ** 2)

            # Check if loss has gradient
            if not loss.requires_grad:
                # No gradient path, skip optimization
                break

            loss.backward()
            optimizer.step()

            # Project to non-negative
            with torch.no_grad():
                beta.clamp_(min=0)

        return beta.detach()

    def _update_dictionary(self, X: torch.Tensor, alphas: torch.Tensor,
                          betas: List[torch.Tensor]) -> torch.Tensor:
        """Update dictionary atoms"""
        new_dict = torch.zeros_like(self.dictionary)

        for k in range(self.n_atoms):
            numerator = torch.zeros(self.atom_length, X.shape[-1] if X.ndim == 3 else 1, device=self.device)
            denominator = 0.0

            for i in range(X.shape[0]):
                if alphas[i, k] > 1e-6:
                    x_i = X[i]

                    # Warp the residual back
                    residual = x_i.clone()
                    for j in range(self.n_atoms):
                        if j != k and alphas[i, j] > 1e-6:
                            # Handle both 2D (univariate) and 3D (multivariate) dictionaries
                            if self.dictionary.ndim == 3:
                                atom_j = self.dictionary[:, j, :]  # (length, n_dims)
                            else:
                                atom_j = self.dictionary[:, j]  # (length,)
                            warped_atom_j = self.time_warping.warp_time_series(atom_j, betas[i])
                            residual = residual - alphas[i, j] * warped_atom_j

                    numerator = numerator + alphas[i, k] * residual
                    denominator += alphas[i, k] ** 2

            if denominator > 1e-8:
                if self.dictionary.ndim == 3:
                    new_dict[:, k, :] = (numerator / denominator)
                    # Normalize
                    new_dict[:, k, :] = new_dict[:, k, :] / (torch.norm(new_dict[:, k, :]) + 1e-8)
                else:
                    new_dict[:, k] = (numerator / denominator).squeeze()
                    # Normalize
                    new_dict[:, k] = new_dict[:, k] / (torch.norm(new_dict[:, k]) + 1e-8)
            else:
                if self.dictionary.ndim == 3:
                    new_dict[:, k, :] = self.dictionary[:, k, :]
                else:
                    new_dict[:, k] = self.dictionary[:, k]

        return new_dict

    def _compute_loss(self, X: torch.Tensor, alphas: torch.Tensor, betas: List[torch.Tensor]) -> float:
        """Compute reconstruction loss + sparsity penalty"""
        total_loss = 0.0

        for i in range(X.shape[0]):
            x_i = X[i]

            # Reconstruct
            reconstruction = torch.zeros_like(x_i)
            for k in range(self.n_atoms):
                if alphas[i, k] > 1e-6:
                    # Handle both 2D (univariate) and 3D (multivariate) dictionaries
                    if self.dictionary.ndim == 3:
                        atom = self.dictionary[:, k, :]  # (length, n_dims)
                    else:
                        atom = self.dictionary[:, k]  # (length,)
                    warped_atom = self.time_warping.warp_time_series(atom, betas[i])
                    reconstruction = reconstruction + alphas[i, k] * warped_atom

            # Reconstruction error
            recon_error = torch.mean((x_i - reconstruction) ** 2)

            # Sparsity penalty
            sparsity = self.lambda_sparse * torch.sum(torch.abs(alphas[i]))

            total_loss += recon_error + sparsity

        return total_loss.item() / X.shape[0]

    def transform(self, X: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Transform time series using learned dictionary

        Args:
            X: Time series data (n_samples, length) or (n_samples, length, n_dims)

        Returns:
            alphas: Sparse coefficients
            betas: Warping coefficients
        """
        if X.ndim == 2:
            X = X.unsqueeze(-1)

        n_samples = X.shape[0]

        # Initialize
        alphas = torch.zeros(n_samples, self.n_atoms, device=self.device)
        betas = [torch.ones(self.n_basis + 1, device=self.device) * 0.1 for _ in range(n_samples)]

        # Optimize for fixed dictionary
        for iteration in range(min(self.max_iter, 50)):
            alphas = self._update_sparse_coefficients(X, betas)
            betas = self._update_warping_coefficients(X, alphas, betas)

        return alphas, betas