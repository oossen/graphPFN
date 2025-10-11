from typing import Tuple
import torch
from torch import Tensor


class Preprocessor:
    """
    A module for preprocessing a tabular dataset. This goes in two steps:
    
    *Fitting* takes in tensors
    X (shape [batch_size, n_samples, n_features]) and Y (shape [batch_size, n_samples])
    and fits the preprocessing pipeline to them.
    
    *Processing* takes in tensors of the same shape and applies the previously-fitted pipeline.
    
    Additionally, there is a *unprocessing* operation that reverts the preprocessor's transformation as much as possible.
    """

    def __init__(
        self,
        negative_one_one_scaling: bool = True,
        standardize: bool = False,
        remove_outliers: bool = True,
        outlier_quantile: float = 0.95,
        eps: float = 1e-8,
    ):
        """
        Parameters
        ----
        negative_one_one_scaling : bool
            Whether to scale features and targets to [-1, 1].
        standardize : bool
            Whether to standardize features (zero mean, unit variance).
        remove_outliers : bool
            Whether to winsorize features based on quantiles.
        outlier_quantile : float
            Upper quantile (q). We winsorize using (1-q, q). Example: 0.95 → clamp to [p5, p95].
        eps : float
            Small numerical constant for divisions / logs.
        """
        assert 0 < outlier_quantile <= 1.0, "outlier_quantile must be in (0, 1]."

        self.negative_one_one_scaling = negative_one_one_scaling
        self.standardize = standardize
        self.remove_outliers = remove_outliers
        self.outlier_quantile = outlier_quantile
        self.eps = eps

    # ------------------------ public API ------------------------

    def fit(self,
        X: Tensor,
        Y: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        self._validate_inputs(X, Y)
        
        # feature processing
        if self.remove_outliers:
            upper_q = float(self.outlier_quantile)
            if not (0.5 < upper_q <= 1.0):
                raise ValueError("outlier_quantile should be in (0.5, 1.0].")
            lower_q = 1.0 - upper_q
            self.lo_x = torch.quantile(X, lower_q, dim=1, keepdim=True)
            self.hi_x = torch.quantile(X, upper_q, dim=1, keepdim=True)
            X = X.clamp(min=self.lo_x, max=self.hi_x)
        
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
        if self.remove_outliers:
            upper_q = float(self.outlier_quantile)
            if not (0.5 < upper_q <= 1.0):
                raise ValueError("y_clip_quantile should be in (0.5, 1.0].")
            lower_q = 1.0 - upper_q
            self.lo_y = torch.quantile(Y, lower_q, dim=1, keepdim=True)
            self.hi_y = torch.quantile(Y, upper_q, dim=1, keepdim=True)
            Y = Y.clamp(min=self.lo_y, max=self.hi_y)
            
        if self.standardize:
            self.mean_y = Y.mean(dim=1, keepdim=True) # [B,1]
            self.std_y = Y.std(dim=1, keepdim=True).clamp_min(self.eps)
            Y = (Y - self.mean_y) / self.std_y

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
        
        if self.standardize:
            X = (X - self.mean_x) / self.std_x
            
        if self.negative_one_one_scaling:
            X = 2.0 * (X - self.min_x) / self.rng_x - 1.0
        
        # target processing
        if self.remove_outliers:
            Y = Y.clamp(min=self.lo_y, max=self.hi_y)
            
        if self.standardize:
            Y = (Y - self.mean_y) / self.std_y

        if self.negative_one_one_scaling:
            Y = 2.0 * (Y - self.min_y) / self.rng_y - 1.0

        return X, Y
    
    def process_x(self, X: Tensor) -> Tensor:
        if self.remove_outliers:
            X = X.clamp(min=self.lo_x, max=self.hi_x)
        
        if self.standardize:
            X = (X - self.mean_x) / self.std_x
            
        if self.negative_one_one_scaling:
            X = 2.0 * (X - self.min_x) / self.rng_x - 1.0
            
        return X
    
    def unprocess_y(self, Y: Tensor) -> Tensor:
        """Undo standardization and scaling."""
        if self.negative_one_one_scaling:
            Y = (Y + 1.0) * self.rng_y / 2.0 + self.min_y
        
        if self.standardize:
            Y = self.std_y * Y + self.mean_y
        
        return Y        

    def _validate_inputs(self, X: Tensor, Y: Tensor) -> None:
        if X.dim() != 3:
            raise ValueError(f"X must have shape [B, N, F], got {tuple(X.shape)}.")
        if Y.dim() != 2:
            raise ValueError(f"Y must have shape [B, N], got {tuple(Y.shape)}.")
        if X.shape[0] != Y.shape[0] or X.shape[1] != Y.shape[1]:
            raise ValueError(f"Batch size / sample count mismatch between X{tuple(X.shape)} and Y{tuple(Y.shape)}.")
