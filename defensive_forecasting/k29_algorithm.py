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
    def __init__(self, n_rff_features=100, gamma=1.0, random_state=None, test_hospital_id=None):
        """
        Parameters:
        n_rff_features (int): Number of Random Fourier Features to use
        gamma (float): RBF kernel parameter for RFF
        random_state (int): Random seed for reproducibility
        test_hospital_id: Optional categorical id to process first during fit
        """
        self.n_rff_features = n_rff_features
        self.gamma = gamma
        self.random_state = random_state
        self.test_hospital_id = test_hospital_id

        self.theta = None
        self.theta_by_g = None
        
        # RFF transformer
        self.rff = None
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
        else:
            z = X[:, :-1]
            g = X[:, -1]
        return z, g
    
    def fit(self, X, y):
        """
        Fit the model to the data using online learning.
        
        Parameters:
        X (array-like): n by d matrix where first d-1 columns are continuous, last is categorical
        y (array-like): array of binary labels in [0,1]
        
        Returns:
        self: Fitted model
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        assert X.shape[0] == len(y), "X and y must have the same number of samples"
        
        n = X.shape[0]

        rng = np.random if self.random_state is None else np.random.RandomState(self.random_state)

        if self.test_hospital_id is None:
            permutation = rng.permutation(n)
        else:
            # Process the held-out hospital first, then a shuffled remainder
            _, g_all = self._split_features(X)
            mask = g_all == self.test_hospital_id
            test_indices = np.nonzero(mask)[0]
            other_indices = np.nonzero(~mask)[0]
            rng.shuffle(other_indices)
            permutation = np.concatenate([test_indices, other_indices])

        X = X[permutation]
        y = y[permutation]
    
        self.theta = None
        self.theta_by_g = {}
        
        # Process data sequentially
        for i in tqdm(range(n)):
            z_i, g_i = self._split_features(X[i])
            
            if i == 0:
                # First point: use p = 0.5
                p_i = 0.5
                # Initialize RFF on first continuous features
                # sklearn's RBFSampler uses gamma parameter
                self.rff = RBFSampler(
                    n_components=self.n_rff_features,
                    gamma=self.gamma,
                    random_state=self.random_state
                )
                self.rff.fit(z_i.reshape(1, -1))
            else:
                # Make prediction using current history
                p_i = self._predict_single(z_i, g_i)

            rff_z_i = self.rff.transform(z_i.reshape(1, -1))[0]

            # Update theta caches
            # Phi(z_i, p_i) = (rff_z_i, p_i, 1)
            if self.theta is None:
                d = rff_z_i.shape[0] + 2
                self.theta = np.zeros(d, dtype=np.float64)
                if self.theta_by_g is None:
                    self.theta_by_g = {}

            gi_val = g_i.item() if isinstance(g_i, np.ndarray) else g_i
            resid = float(y[i]) - float(p_i)

            # Global Theta update
            self.theta[:-2] += rff_z_i * resid
            self.theta[-2]  += float(p_i) * resid
            self.theta[-1]  += 1.0 * resid

            # Per-g Theta update
            theta_g = self.theta_by_g.get(gi_val)
            if theta_g is None:
                theta_g = np.zeros_like(self.theta)
                self.theta_by_g[gi_val] = theta_g

            theta_g[:-2] += rff_z_i * resid
            theta_g[-2]  += float(p_i) * resid
            theta_g[-1]  += 1.0 * resid
        
        self.fitted_ = True
        return self
    
    def _predict_single(self, z, g):
        g_val = g.item() if isinstance(g, np.ndarray) else g
        rff_z = self.rff.transform(z.reshape(1, -1))[0]
        rff_self = float(np.dot(rff_z, rff_z))

        # Grab caches (should exist after fit; keep fallback if you want)
        if self.theta is None:
            d = rff_z.shape[0] + 2
            self.theta = np.zeros(d, dtype=np.float64)
            if self.theta_by_g is None:
                self.theta_by_g = {}

        t = self.theta
        tg = self.theta_by_g.get(g_val)
        if tg is None:
            tg = np.zeros_like(t)

        # Precompute history-term coefficients:
        # history(p) = A + B*p + C
        A = float(np.dot(rff_z, t[:-2]) + np.dot(rff_z, tg[:-2]))
        B = float(t[-2] + tg[-2])
        C = float(t[-1] + tg[-1])

        def potential(p):
            # History term
            s = A + B * p + C
            # Self term: (||rff||^2 + p^2 + 1) * (1 - 2p)
            s += (rff_self + p * p + 1.0) * (1.0 - 2.0 * p)
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
        if self.rff is None:
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
