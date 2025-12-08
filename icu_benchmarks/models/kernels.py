import gin
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel


@gin.configurable
class GaussianRBFKernel(BaseEstimator, TransformerMixin):
    """RBF feature map that can be plugged into sklearn pipelines.

    The transformer replaces the original feature vector with its similarity to a
    set of reference points (``centers``) using the standard Gaussian / RBF
    kernel.  When combined with linear models this enables simple kernel methods
    without having to re-implement each estimator.
    """

    def __init__(
        self,
        gamma: float | None = None,
        num_centers: int | None = None,
        center_selection: str = "all",
        random_state: int | None = None,
    ):
        """
        Args:
            gamma: Width parameter of the RBF. If ``None`` defaults to
                ``1 / n_features`` (matching sklearn's default).
            num_centers: Optional cap on how many reference points to keep. When
                ``None`` all training samples are used.
            center_selection: Strategy for choosing centers. Currently supports
                ``"all"`` (use every point) and ``"random"`` (uniform sampling).
            random_state: Seed used when ``center_selection == "random"``.
        """
        self.gamma = gamma
        self.num_centers = num_centers
        self.center_selection = center_selection
        self.random_state = random_state
        self.centers_ = None
        self._gamma = gamma
        self.n_features_in_ = None

    def fit(self, X, y=None):
        """Select reference centers and prepare the kernel width."""
        X = self._validate_input(X)
        total = X.shape[0]

        if self.num_centers is None or self.center_selection == "all" or self.num_centers >= total:
            centers = X
        elif self.center_selection == "random":
            rng = np.random.default_rng(self.random_state)
            indices = rng.choice(total, size=self.num_centers, replace=False)
            centers = X[indices]
        else:
            raise ValueError(f"Unsupported center selection strategy: {self.center_selection}")

        self.centers_ = np.array(centers, copy=True)
        self.n_features_in_ = X.shape[1]
        self._gamma = self.gamma if self.gamma is not None else 1.0 / self.n_features_in_
        return self

    def transform(self, X):
        """Project data onto the RBF feature space."""
        if self.centers_ is None:
            raise RuntimeError("GaussianRBFKernel must be fitted before calling transform.")
        X = self._validate_input(X)
        return rbf_kernel(X, self.centers_, gamma=self._gamma)

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)

    def _validate_input(self, X):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D input.")
        return X
