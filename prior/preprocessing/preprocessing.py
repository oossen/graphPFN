from typing import Optional, Tuple
import torch
from torch import Tensor


class Preprocessor:
    """
    A module for preprocessing a tabular dataset. This goes in two steps:
    
    *Fitting* takes in tensors
    X (shape [batch_size, n_samples, n_features]) and Y (shape [batch_size, n_samples])
    and fits the preprocessing pipeline to them.
    
    *Processing* takes in tensors of the same shape and applies the previously-fitted pipeline.
    """

    def __init__(
        self,
        negative_one_one_scaling: bool = True,
        standardize: bool = False,
        yeo_johnson: bool = False,
        remove_outliers: bool = True,
        outlier_quantile: float = 0.95,
        eps: float = 1e-8,
        y_clip_quantile: Optional[float] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Parameters
        ----
        negative_one_one_scaling : bool
            Whether to scale features and targets to [-1, 1].
        standardize : bool
            Whether to standardize features (zero mean, unit variance).
        yeo_johnson : bool
            Whether to apply Yeo-Johnson transform to features before standardization.
        remove_outliers : bool
            Whether to winsorize features based on quantiles.
        outlier_quantile : float
            Upper quantile (q). We winsorize using (1-q, q). Example: 0.95 → clamp to [p5, p95].
        eps : float
            Small numerical constant for divisions / logs.
        y_clip_quantile : float (optional)
            Optional winsorization for Y.
        device/dtype: Optional overrides for output tensors.
        """
        assert 0 < outlier_quantile <= 1.0, "outlier_quantile must be in (0, 1]."

        self.negative_one_one_scaling = negative_one_one_scaling
        self.standardize = standardize
        self.yeo_johnson = yeo_johnson
        self.remove_outliers = remove_outliers
        self.outlier_quantile = outlier_quantile
        self.eps = eps
        self.y_clip_quantile = y_clip_quantile

        self.device = device
        self.dtype = dtype

    # ------------------------ public API ------------------------

    def fit(self,
        X: Tensor,
        Y: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        self._validate_inputs(X, Y)
        
        # feature processing
        if self.remove_outliers:
            q = float(self.outlier_quantile)
            if not (0.5 < q <= 1.0):
                raise ValueError("outlier_quantile should be in (0.5, 1.0].")
            lower_q = 1.0 - q
            # torch.quantile supports q as tensor
            qs = torch.tensor([lower_q, q], device=X.device, dtype=X.dtype)
            # Compute per (B, F)
            # shape: [B, 2, F]
            Q = torch.quantile(X.transpose(1, 2), qs, dim=-1, keepdim=False).transpose(0, 1)
            self.lo_x = Q[:, 0, :].unsqueeze(1)  # [B,1,F]
            self.hi_x = Q[:, 1, :].unsqueeze(1)  # [B,1,F]
            X = X.clamp(min=self.lo_x, max=self.hi_x)
        
        if self.yeo_johnson:
            self.lambdas = self._fit_yeo_johnson_lambdas(X)
            X = self._apply_yeo_johnson(X, self.lambdas)
        
        if self.standardize:
            self.mean_x = X.mean(dim=1, keepdim=True) # [B,1,F]
            self.std_x = X.std(dim=1, keepdim=True).clamp_min(self.eps)
            X = (X - self.mean_x) / self.std_x
            
        if self.negative_one_one_scaling:
            # Min-max per (B,F) using train, then scale both sets
            self.min_x = X.amin(dim=1, keepdim=True)
            self.max_x = X.amax(dim=1, keepdim=True)
            self.rng_x = (self.max_x - self.min_x).clamp_min(self.eps)
            X = 2.0 * (X - self.min_x) / self.rng_x - 1.0
        
        # target processing
        if self.y_clip_quantile is not None:
            q = float(self.y_clip_quantile)
            if not (0.5 < q <= 1.0):
                raise ValueError("y_clip_quantile should be in (0.5, 1.0].")
            qs = torch.tensor([1.0 - q, q], device=Y.device, dtype=Y.dtype)
            Q = torch.quantile(Y, qs, dim=1, keepdim=True)  # [B,2,1] effectively
            self.lo_y = Q[:, 0:1, :]
            self.hi_y = Q[:, 1:2, :]
            Y = Y.clamp(min=self.lo_y.squeeze(-1), max=self.hi_y.squeeze(-1))

        if self.negative_one_one_scaling:
            self.min_y = Y.amin(dim=1, keepdim=True)
            self.max_y = Y.amax(dim=1, keepdim=True)
            self.rng_y = (self.max_y - self.min_y).clamp_min(self.eps)
            Y = 2.0 * (Y - self.min_y) / self.rng_y - 1.0
        
        return X, Y
    
    def process(
        self,
        X: Tensor,  # [B, N, F]
        Y: Tensor,  # [B, N]
    ) -> Tuple[Tensor, Tensor]:
        self._validate_inputs(X, Y)

        # feature processing
        if self.remove_outliers:
            X = X.clamp(min=self.lo_x, max=self.hi_x)
        
        if self.yeo_johnson:
            X = self._apply_yeo_johnson(X, self.lambdas)
        
        if self.standardize:
            X = (X - self.mean_x) / self.std_x
            
        if self.negative_one_one_scaling:
            X = 2.0 * (X - self.min_x) / self.rng_x - 1.0
        
        # target processing
        if self.y_clip_quantile is not None:
            Y = Y.clamp(min=self.lo_y.squeeze(-1), max=self.hi_y.squeeze(-1))

        if self.negative_one_one_scaling:
            Y = 2.0 * (Y - self.min_y) / self.rng_y - 1.0

        # Cast/device if requested
        if self.dtype is not None:
            X = X.to(self.dtype)
            Y = Y.to(self.dtype)
        if self.device is not None:
            X = X.to(self.device)
            Y = Y.to(self.device)

        return X, Y

    # ------------------------ helpers: validation ------------------------

    def _validate_inputs(self, X: Tensor, Y: Tensor) -> None:
        if X.dim() != 3:
            raise ValueError(f"X must have shape [B, N, F], got {tuple(X.shape)}.")
        if Y.dim() != 2:
            raise ValueError(f"Y must have shape [B, N], got {tuple(Y.shape)}.")
        if X.shape[0] != Y.shape[0] or X.shape[1] != Y.shape[1]:
            raise ValueError(f"Batch size / sample count mismatch between X{tuple(X.shape)} and Y{tuple(Y.shape)}.")

    # ------------------------ helpers: Yeo-Johnson ------------------------

    @staticmethod
    def _yeo_johnson_transform(x: Tensor, lam: Tensor, eps: float) -> Tensor:
        """
        Compute Yeo-Johnsom transform of x and lam.
        The two inputs must be broadcastable.
        
        For example, in applying the fitted transform, we have x of shape [B, N, F] and lam of shape [B, 1, F].
        During fitting (when several lambdas are evaluated), we have x of shape [B, N, 1, F] and lam of shape [1, 1, E, 1].
        (E is the number of values for lambda being evaluated.)
        """
        # piecewise
        pos = x >= 0
        lam_near0 = torch.abs(lam) < 1e-6

        # For x >= 0:
        # lam != 0: ((x + 1)^lam - 1) / lam
        # lam == 0: log(x + 1)
        out_pos = torch.where(
            lam_near0, torch.log1p(x.clamp_min(0.0) + 0.0), ((x + 1.0).clamp_min(eps) ** lam - 1.0) / (lam + 0.0)
        )

        # For x < 0:
        # lam != 2: - ((-x + 1)^(2 - lam) - 1) / (2 - lam)
        # lam == 2: -log(-x + 1)
        two_minus_lam = 2.0 - lam
        near2 = torch.abs(two_minus_lam) < 1e-6
        xm = (-x).clamp_min(0.0) + 1.0
        out_neg = torch.where(
            near2, -torch.log(xm.clamp_min(eps)), -((xm.clamp_min(eps) ** two_minus_lam - 1.0) / two_minus_lam)
        )

        return torch.where(pos, out_pos, out_neg)

    def _fit_yeo_johnson_lambdas(self, X: Tensor) -> Tensor:
        """
        Fit per-batch, per-feature lambda by grid-search MLE under normality:
        maximizes Gaussian log-likelihood of transformed data (up to constants)
        using the Jacobian term of YJ (sum log|dT/dx|). This is a pragmatic, stable approach.

        Returns:
            lambdas: [B, F]
        """
        # Lambda grid
        grid = torch.linspace(-2.0, 2.0, steps=41, device=X.device, dtype=X.dtype)  # 0.1 step
        # Prepare broadcast shapes
        x = X.unsqueeze(2)  # [B, N, 1, F]
        lam = grid.view(1, 1, -1, 1)  # [1,1,L,1] -> broadcast to [B,N,L,F]

        # Transform
        xt = self._yeo_johnson_transform(x, lam, self.eps)  # [B, N, L, F]

        # Gaussian MLE log-likelihood (per (B,L,F)): -N/2 * log(var) + Jacobian term
        # Compute mean/var across N
        mean = xt.mean(dim=1, keepdim=True)
        """Changed keepdim to False"""
        var = xt.var(dim=1, unbiased=False, keepdim=False).clamp_min(self.eps)
        ll_gauss = -0.5 * (xt - mean).pow(2).sum(dim=1) / var  # [B, L, F] up to constants
        ll_gauss += -0.5 * torch.log(var.squeeze(1)) * xt.shape[1]        # add -N/2 log(var)

        # Add Jacobian log|dT/dx|
        # For Yeo-Johnson, the derivative:
        # x>=0: (x+1)^(lam-1)
        # x<0:  (1-x)^(1-lam)
        """In this block, changed Xtr for xt"""
        pos = X >= 0
        jac_pos = ((xt+ 1.0).clamp_min(self.eps)) ** (lam - 1.0)  # [B,N,L,F]
        jac_neg = ((1.0 - xt).clamp_min(self.eps)) ** (1.0 - lam)
        log_jac = torch.where(pos.unsqueeze(2), torch.log(jac_pos.clamp_min(self.eps)), torch.log(jac_neg.clamp_min(self.eps)))
        ll = ll_gauss + log_jac.sum(dim=1)  # [B, L, F]

        # Choose best lambda per (B,F)
        idx = torch.argmax(ll, dim=1)  # [B, F] index into grid
        lambdas = grid[idx]            # [B, F]
        return lambdas

    def _apply_yeo_johnson(self, X: Tensor, lambdas: Tensor) -> Tensor:
        # reshape lambdas to [B,1,F] for broadcasting across samples
        lam = lambdas.unsqueeze(1)
        return self._yeo_johnson_transform(X, lam, self.eps)