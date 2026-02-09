import numpy as np
from sklearn.kernel_approximation import RBFSampler
from tqdm import tqdm


def binary_search(f, low=0, high=1, tol=1e-3, max_iter=100):
    """
    Find a zero of the function f using binary search.
    
    Parameters:
    f (function): The function to find the zero of.
    tol (float): The tolerance for stopping the search. Stops when |f(x)| < tol
    max_iter (int): Maximum number of iterations to prevent infinite loops

    Returns:
    float: The point where the function is zero or (1 + sign(f(0)))/2 if signs do not differ.
    """
    f0, f1 = np.sign(f(0)), np.sign(f(1))
    if (f0 == f1) and f1 != 0:
        return (1 + f0) / 2

    for _ in range(max_iter):
        mid = (low + high) / 2
        f_mid = f(mid)
        if abs(f_mid) <= tol:
            return mid
        sign_f_mid = np.sign(f_mid)

        if sign_f_mid == f0:
            low = mid
        else:
            high = mid
    
    # Return midpoint if max_iter reached
    return (low + high) / 2


class K29:
    """
    K29 algorithm implementation with Random Fourier Features and categorical features.
    Optimized version with improved runtime performance.
    """
    def __init__(self, n_rff_features=100, gamma=1.0, random_state=None, test_hospital_id=None, pos_weight=1.0, neg_weight=1.0):
        """
        Parameters:
        n_rff_features (int): Number of Random Fourier Features to use
        gamma (float): RBF kernel parameter for RFF
        random_state (int): Random seed for reproducibility
        test_hospital_id: Optional categorical id to process first during fit
        """
        self.n_rff_features = n_rff_features
        if isinstance(gamma, (list, tuple, np.ndarray)):
            gammas = list(gamma)
            if len(gammas) == 0:
                raise ValueError("gamma list must be non-empty")
            self.gammas = [float(g) for g in gammas]
        else:
            self.gammas = [float(gamma)]

        self.random_state = random_state
        self.test_hospital_id = test_hospital_id

        self.theta = None
        self.theta_by_g = None

        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        
        # RFF transformer
        self.rffs = None
        self.fitted_ = False
        
    def _split_features(self, X):
        """
        Split X into continuous features z and categorical feature g.
        X has d features: first d-1 are continuous (z), last is categorical (g).
        
        Parameters:
        X (array-like): Features of shape (n_samples, d) or (d,)
        
        Returns:
        z (array): Continuous features of shape (n_samples, d-1) or (d-1,)
        g (array): Categorical features of shape (n_samples,) or scalar
        """
        X = np.asarray(X)
        if X.ndim == 1:
            z = X[:-1]
            g = X[-1]
            g = g.item() if isinstance(g, np.ndarray) else g
        else:
            z = X[:, :-1]
            g = X[:, -1]
        return z, g

    def _concat_p(self, z, p):
        # z: (d-1,)  p: scalar
        z = np.asarray(z, dtype=np.float64)
        return np.concatenate([z, np.asarray([p], dtype=np.float64)], axis=0)

    def _initialize_rffs(self, z_aug_1d):
        """Initialize RBFSamplers (single or multiscale) deterministically."""
        if self.rffs is not None:
            return

        self.rffs = []
        base_seed = self.random_state

        for j, gm in enumerate(self.gammas):
            # deterministic-but-different seeds across scales
            rs = None if base_seed is None else int(base_seed + 10007 * j)
            rff = RBFSampler(
                n_components=self.n_rff_features,
                gamma=float(gm),
                random_state=rs,
            )
            rff.fit(z_aug_1d.reshape(1, -1))
            self.rffs.append(rff)

    def _rff_transform(self, z_aug_1d):
        """Concatenate RFF features across all scales."""
        feats = [rff.transform(z_aug_1d.reshape(1, -1))[0] for rff in self.rffs]
        return np.concatenate(feats, axis=0)

    # def fit(self, X, y, log_curve=False, log_every=1, metrics_logger=None):
    #     X = np.asarray(X)
    #     y = np.asarray(y, dtype=np.float64)
    #     assert X.shape[0] == len(y), "X and y must have the same number of samples"
    #     n = X.shape[0]

    #     rng = np.random if self.random_state is None else np.random.RandomState(self.random_state)

    #     if self.test_hospital_id is None:
    #         permutation = rng.permutation(n)
    #     else:
    #         _, g_all = self._split_features(X)
    #         mask = g_all == self.test_hospital_id
    #         test_indices = np.nonzero(mask)[0]
    #         other_indices = np.nonzero(~mask)[0]
    #         rng.shuffle(other_indices)
    #         permutation = np.concatenate([test_indices, other_indices])

    #     X = X[permutation]
    #     y = y[permutation]

    #     # auto pos_weight if requested
    #     if not self.pos_weight:
    #         pos_rate = float(np.mean(y))
    #         self.pos_weight = 1.0 / max(pos_rate, 1e-8)
    #     else:
    #         # ensure it's set
    #         if self.pos_weight is None:
    #             self.pos_weight = 1.0

    #     self.theta = None
    #     self.theta_by_g = {}
    #     self.rffs = None

    #     log_every = int(log_every) if log_every is not None else 1
    #     if log_every < 1:
    #         log_every = 1
    #     log_curve = bool(log_curve) and metrics_logger is not None
    #     correct = 0
    #     eps = 1e-12

    #     for i in tqdm(range(n)):
    #         z_i, g_i = self._split_features(X[i])

    #         if i == 0:
    #             p_i = 0.5
    #         else:
    #             p_i = self._predict_single(z_i, g_i)

    #         z_aug_i = self._concat_p(z_i, p_i)

    #         if i == 0 and self.rffs is None:
    #             self._initialize_rffs(z_aug_i)

    #         rff_i = self._rff_transform(z_aug_i)

    #         # theta dimension = (#rff_total + bias)
    #         if self.theta is None:
    #             d = rff_i.shape[0] + 1
    #             self.theta = np.zeros(d, dtype=np.float64)

    #         # weighted residual
    #         yi = float(y[i])
    #         w = self.pos_weight if yi > 0.5 else self.neg_weight
    #         resid = (yi - float(p_i)) * float(w)

    #         # global update
    #         self.theta[:-1] += rff_i * resid
    #         self.theta[-1]  += 1.0 * resid

    #         # per-g update
    #         tg = self.theta_by_g.get(g_i)
    #         if tg is None:
    #             tg = np.zeros_like(self.theta)
    #             self.theta_by_g[g_i] = tg

    #         tg[:-1] += rff_i * resid
    #         tg[-1]  += 1.0 * resid

    #         if log_curve:
    #             yi = float(y[i])
    #             y_hat = 1.0 if p_i >= 0.5 else 0.0
    #             if y_hat == yi:
    #                 correct += 1
    #             if i % log_every == 0:
    #                 p_clip = float(np.clip(p_i, eps, 1.0 - eps))
    #                 logloss = -(yi * np.log(p_clip) + (1.0 - yi) * np.log(1.0 - p_clip))
    #                 metrics_logger.log_metrics(
    #                     {
    #                         "curve/p": float(p_i),
    #                         "curve/y": yi,
    #                         "curve/logloss": float(logloss),
    #                         "curve/accuracy": float(y_hat == yi),
    #                         "curve/cum_accuracy": float(correct / (i + 1)),
    #                     },
    #                     step=i,
    #                 )

    #     self.fitted_ = True
    #     return self

    def fit(
        self,
        X,
        y,
        log_curve=False,
        log_every=1,
        metrics_logger=None,
        *,
        lr0: float = 0.4,          # base LR after warmup (safe with normalization)
        lr_min: float = 1e-4,      # floor to keep late points contributing
        warmup: int = 200,         # steps
        power: float = 0.5,        # 0.5 => inverse-sqrt decay
        l2_decay: float = 0.0,     # try 1e-6 to 1e-5 if theta drifts
        normalize: bool = True,    # NLMS-style normalization
    ):
        X = np.asarray(X)
        y = np.asarray(y, dtype=np.float64)
        assert X.shape[0] == len(y), "X and y must have the same number of samples"
        n = X.shape[0]

        rng = np.random if self.random_state is None else np.random.RandomState(self.random_state)

        if self.test_hospital_id is None:
            permutation = rng.permutation(n)
        else:
            _, g_all = self._split_features(X)
            mask = g_all == self.test_hospital_id
            test_indices = np.nonzero(mask)[0]
            other_indices = np.nonzero(~mask)[0]
            rng.shuffle(other_indices)
            permutation = np.concatenate([test_indices, other_indices])

        X = X[permutation]
        y = y[permutation]

        # auto pos_weight if requested
        if not self.pos_weight:
            pos_rate = float(np.mean(y))
            self.pos_weight = 1.0 / max(pos_rate, 1e-8)
        else:
            if self.pos_weight is None:
                self.pos_weight = 1.0

        self.theta = None
        self.theta_by_g = {}
        self.rffs = None

        log_every = int(log_every) if log_every is not None else 1
        if log_every < 1:
            log_every = 1
        log_curve = bool(log_curve) and metrics_logger is not None
        correct = 0
        eps = 1e-12

        def lr_at(step: int) -> float:
            t = step + 1
            if warmup > 0 and t <= warmup:
                lr = lr0 * (t / warmup)
            else:
                lr = lr0 / (t ** power)
            return max(lr, lr_min)

        for i in tqdm(range(n)):
            z_i, g_i = self._split_features(X[i])

            if i == 0:
                p_i = 0.5
            else:
                p_i = self._predict_single(z_i, g_i)

            z_aug_i = self._concat_p(z_i, p_i)

            if i == 0 and self.rffs is None:
                self._initialize_rffs(z_aug_i)

            rff_i = self._rff_transform(z_aug_i)

            # theta dimension = (#rff_total + bias)
            if self.theta is None:
                d = rff_i.shape[0] + 1
                self.theta = np.zeros(d, dtype=np.float64)

            # weighted residual
            yi = float(y[i])
            w = self.pos_weight if yi > 0.5 else self.neg_weight
            resid = (yi - float(p_i)) * float(w)

            lr = lr_at(i)

            # NLMS normalization: scale by feature energy (including bias as +1)
            if normalize:
                phi2 = float(np.dot(rff_i, rff_i) + 1.0)
                step_size = lr / phi2
            else:
                step_size = lr

            # optional very small weight decay (keeps norms bounded)
            if l2_decay > 0.0:
                self.theta *= (1.0 - l2_decay)

            # global update with scheduled step size
            self.theta[:-1] += step_size * rff_i * resid
            self.theta[-1]  += step_size * 1.0  * resid

            # per-g update
            tg = self.theta_by_g.get(g_i)
            if tg is None:
                tg = np.zeros_like(self.theta)
                self.theta_by_g[g_i] = tg

            if l2_decay > 0.0:
                tg *= (1.0 - l2_decay)

            tg[:-1] += step_size * rff_i * resid
            tg[-1]  += step_size * 1.0  * resid

            if log_curve:
                y_hat = 1.0 if p_i >= 0.5 else 0.0
                if y_hat == yi:
                    correct += 1
                if i % log_every == 0:
                    p_clip = float(np.clip(p_i, eps, 1.0 - eps))
                    logloss = -(yi * np.log(p_clip) + (1.0 - yi) * np.log(1.0 - p_clip))
                    theta_norm = float(np.linalg.norm(self.theta))
                    metrics_logger.log_metrics(
                        {
                            "curve/p": float(p_i),
                            "curve/y": yi,
                            "curve/logloss": float(logloss),
                            "curve/accuracy": float(y_hat == yi),
                            "curve/cum_accuracy": float(correct / (i + 1)),
                            "curve/lr": float(lr),
                            "curve/step_size": float(step_size),
                            "curve/theta_norm": theta_norm,
                        },
                        step=i,
                    )

        self.fitted_ = True
        return self


    def _predict_single(self, z, g):
        if self.rffs is None:
            # not fitted yet; but allow prediction during fit after first point
            z_aug = self._concat_p(z, 0.5)
            self._initialize_rffs(z_aug)

        # Ensure theta exists (defensive)
        if self.theta is None:
            rff_tmp = self._rff_transform(self._concat_p(z, 0.5))
            d = rff_tmp.shape[0] + 1
            self.theta = np.zeros(d, dtype=np.float64)
            self.theta_by_g = {}

        t = self.theta
        tg = self.theta_by_g.get(g)
        if tg is None:
            tg = np.zeros_like(t)

        # Combine once (saves a tiny bit)
        w_rff = t[:-1] + tg[:-1]
        b = float(t[-1] + tg[-1])

        def potential(p):
            z_aug = self._concat_p(z, p)
            rff_z = self._rff_transform(z_aug)

            # history term
            s = float(np.dot(rff_z, w_rff)) + b

            # self term (using rff([z,p]) since p is inside the features)
            rff_self = float(np.dot(rff_z, rff_z))
            s += (rff_self + 1.0) * (1.0 - 2.0 * p)

            return s

        if potential(1.0) >= 0:
            return 1.0
        if potential(0.0) <= 0:
            return 0.0
        return binary_search(potential)
    
    def predict(self, X):
        """
        Make predictions on new data points.
        
        Parameters:
        X (array-like): Data to predict on, shape (n_samples, d) or (d,)
        
        Returns:
        array or float: Predicted probabilities
        """
        if self.rffs is None:
            raise ValueError("Model must be fitted before making predictions.")
        
        X = np.asarray(X)
        single_sample = X.ndim == 1
        
        if single_sample:
            z, g = self._split_features(X)
            return self._predict_single(z, g)
        
        # Multiple samples
        predictions = []
        for x in X:
            z, g = self._split_features(x)
            predictions.append(self._predict_single(z, g))
        
        return np.array(predictions)
