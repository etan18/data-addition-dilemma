import numpy as np
from sklearn.kernel_approximation import RBFSampler
from tqdm import tqdm


def binary_search(f, low=0, high=1, tol=1e-3):
    """
    Find a zero of the function f using binary search.
    
    Parameters:
    f (function): The function to find the zero of.
    tol (float): The tolerance for stopping the search. Stops when |f(x)| < tol

    Returns:
    float: The point where the function is zero or (1 + sign(f(0)))/2 if signs do not differ.
    """
    f0, f1 = np.sign(f(0)), np.sign(f(1))
    if (f0 == f1) and f1 != 0:
        return (1 + f0) / 2

    while True:
        mid = (low + high) / 2
        f_mid = f(mid)
        if abs(f_mid) <= tol:
            return mid
        sign_f_mid = np.sign(f_mid)

        if sign_f_mid == f0:
            low = mid
        else:
            high = mid



class K29:
    """
    K29 algorithm implementation with Random Fourier Features and categorical features.
    Optimized version with improved runtime performance.
    """
    def __init__(
        self,
        n_rff_features=100,
        gamma=1.0,
        random_state=None,
        test_hospital_id=None,
    ):
        """
        Parameters:
        n_rff_features (int): Number of Random Fourier Features PER SCALE
        gamma (float or list[float]): kept for backward-compat; list of RBF gammas for multiscale kernel
        """
        self.n_rff_features = n_rff_features
        self.random_state = random_state

        # multiscale: if None, fall back to single gamma
        if isinstance(gamma, list):
            if len(gamma) == 1:
                self.gamma = gamma[0]
            else:
                self.gammas = gamma
        else:
            self.gamma = gamma

        # residual reweighting
        self.pos_weight = 10.0
        self.neg_weight = 1.0

        # History storage
        self.history_z = []
        self.history_g = []
        self.history_p = []
        self.history_y = []
        self.history_rff_z = []  # now stores MULTISCALE rff vector

        # RFF transformer(s)
        self.rffs = None  # list[RBFSampler]
        self.fitted_ = False

    def _split_features(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            z = X[:-1]
            g = X[-1]
            g = g.item() if isinstance(g, np.ndarray) else g
        else:
            z = X[:, :-1]
            g = X[:, -1]
        return z, g

    # NEW helper: multiscale transform
    def _rff_transform(self, z_1d):
        """
        Returns concatenated RFF features across all scales.
        z_1d: shape (d-1,)
        returns: shape (len(gammas)*n_rff_features,)
        """
        z_1d = np.asarray(z_1d)
        feats = [rff.transform(z_1d.reshape(1, -1))[0] for rff in self.rffs]
        return np.concatenate(feats, axis=0)

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        assert X.shape[0] == len(y), "X and y must have the same number of samples"
        n = X.shape[0]

        if self.random_state is None:
            permutation = np.random.permutation(n)
        else:
            rng = np.random.RandomState(self.random_state)
            permutation = rng.permutation(n)

        X = X[permutation]
        y = y[permutation]

        # If pos_weight not provided, set it to inverse prevalence (common default)
        if self.pos_weight is None:
            pos_rate = float(np.mean(y))
            # avoid divide-by-zero
            self.pos_weight = (1.0 / max(pos_rate, 1e-8))

        # Reset history
        self.history_z = []
        self.history_g = []
        self.history_p = []
        self.history_y = []
        self.history_rff_z = []

        for i in tqdm(range(n)):
            z_i, g_i = self._split_features(X[i])

            if i == 0:
                p_i = 0.5

                # Initialize multiscale RFFs on first point
                self.rffs = []
                # make deterministic-but-different seeds per gamma
                base_seed = self.random_state if self.random_state is not None else None
                for j, gm in enumerate(self.gammas):
                    rs = None if base_seed is None else int(base_seed + 10007 * j)
                    rff = RBFSampler(
                        n_components=self.n_rff_features,
                        gamma=float(gm),
                        random_state=rs
                    )
                    rff.fit(z_i.reshape(1, -1))
                    self.rffs.append(rff)
            else:
                p_i = self._predict_single(X[i])

            rff_z_i = self._rff_transform(z_i)

            self.history_z.append(z_i)
            self.history_g.append(g_i)
            self.history_p.append(p_i)
            self.history_y.append(y[i])
            self.history_rff_z.append(rff_z_i)

        self.fitted_ = True
        return self

    def _predict_single(self, X):
        z, g = self._split_features(X)

        rff_z = self._rff_transform(z)  # multiscale
        history_len = len(self.history_z)

        if history_len > 0:
            history_rff = np.array(self.history_rff_z)              # (t, Dms)
            history_p = np.array(self.history_p, dtype=np.float64)  # (t,)
            history_y = np.array(self.history_y, dtype=np.float64)  # (t,)

            # base residuals
            residuals = history_y - history_p  # (t,)

            # NEW: reweight residuals by label
            # (y==1)*pos_weight + (y==0)*neg_weight
            weights = np.where(history_y > 0.5, float(self.pos_weight), float(self.neg_weight))
            residuals = residuals * weights

            categorical_factors = np.empty(history_len, dtype=np.float64)
            for idx in range(history_len):
                categorical_factors[idx] = 2.0 if self.history_g[idx] == g else 1.0

            rff_dot_history = history_rff @ rff_z  # (t,)
        else:
            history_p = residuals = rff_dot_history = categorical_factors = None

        rff_self = float(np.dot(rff_z, rff_z))

        def potential(p):
            s = 0.0
            if history_len > 0:
                dot_products = rff_dot_history + p * history_p + 1.0
                k_vals = dot_products * categorical_factors
                s += float(np.dot(k_vals, residuals))

            dot_self = rff_self + p * p + 1.0
            s += float(dot_self * (1.0 - 2.0 * p))
            return s

        pot_1 = potential(1.0)
        if pot_1 >= 0:
            return 1.0

        pot_0 = potential(0.0)
        if pot_0 <= 0:
            return 0.0

        return binary_search(potential)

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 2:
            return np.array([self._predict_single(row) for row in X], dtype=float)
        return float(self._predict_single(X))
