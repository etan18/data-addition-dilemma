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
    def __init__(self, n_rff_features=100, gamma=1.0, random_state=None):
        """
        Parameters:
        n_rff_features (int): Number of Random Fourier Features to use
        gamma (float): RBF kernel parameter for RFF
        random_state (int): Random seed for reproducibility
        """
        self.n_rff_features = n_rff_features
        self.gamma = gamma
        self.random_state = random_state
        
        # History storage - use lists for append efficiency
        self.history_z = []  # Continuous features (d-1 dimensions)
        self.history_g = []  # Categorical features (last dimension)
        self.history_p = []  # Predictions
        self.history_y = []  # Labels
        self.history_rff_z = []  # Cached RFF features for continuous part
        
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
    
    def _phi(self, z, p):
        """
        Feature mapping: Phi(z, p) = (RFF(z), p, 1)
        
        Parameters:
        z (array): Continuous features of shape (d-1,) or (n_samples, d-1)
        p (float or array): Probability predictions
        
        Returns:
        array: Phi features of shape (n_rff_features + 2,) or (n_samples, n_rff_features + 2)
        """
        z = np.asarray(z)
        if z.ndim == 1:
            # Single sample
            rff_z = self.rff.transform(z.reshape(1, -1))[0]
            return np.concatenate([rff_z, [p, 1.0]])
        else:
            # Multiple samples
            rff_z = self.rff.transform(z)
            p_array = np.asarray(p)
            if p_array.ndim == 0:
                p_array = np.full(len(z), p)
            bias = np.ones(len(z))
            return np.column_stack([rff_z, p_array, bias])
    
    def _kernel(self, z1, g1, p1, z2, g2, p2):
        """
        Kernel function: k((z,g,p), (z',g',p')) = <Phi(z,p), Phi(z',p')> * (1{g=g'} + 1)
        
        Parameters:
        z1, z2: Continuous features
        g1, g2: Categorical features
        p1, p2: Probability predictions
        
        Returns:
        float: Kernel value
        """
        phi1 = self._phi(z1, p1)
        phi2 = self._phi(z2, p2)
        
        # Dot product of Phi features
        dot_product = np.dot(phi1, phi2)
        
        # Categorical factor: 2 if g1 == g2, else 1
        # Handle numpy array comparisons
        g1_val = g1.item() if isinstance(g1, np.ndarray) else g1
        g2_val = g2.item() if isinstance(g2, np.ndarray) else g2
        categorical_factor = 2.0 if g1_val == g2_val else 1.0
        
        return dot_product * categorical_factor
    
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

        if self.random_state is None:
            permutation = np.random.permutation(n)
        else:
            rng = np.random.RandomState(self.random_state)
            permutation = rng.permutation(n)
            
        X = X[permutation]
        y = y[permutation]
        
        # Initialize history
        self.history_z = []
        self.history_g = []
        self.history_p = []
        self.history_y = []
        self.history_rff_z = []
        
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
            
            # Store in history
            self.history_z.append(z_i)
            self.history_g.append(g_i)
            self.history_p.append(p_i)
            self.history_y.append(y[i])
            self.history_rff_z.append(rff_z_i)
        
        self.fitted_ = True
        return self
    
    def _predict_single(self, z, g):
        """
        Make prediction on a single data point (optimized version).
        
        Parameters:
        z: Continuous features
        g: Categorical feature
        
        Returns:
        float: Predicted probability
        """
        g_val = g.item() if isinstance(g, np.ndarray) else g
        rff_z = self.rff.transform(z.reshape(1, -1))[0]
        history_len = len(self.history_z)
        
        # Precompute history arrays once
        if history_len > 0:
            # Convert to numpy arrays once
            history_rff = np.array(self.history_rff_z)
            history_p = np.array(self.history_p, dtype=np.float64)
            history_y = np.array(self.history_y, dtype=np.float64)
            residuals = history_y - history_p
            
            # Precompute categorical factors
            categorical_factors = np.empty(history_len, dtype=np.float64)
            for idx in range(history_len):
                categorical_factors[idx] = 2.0 if self.history_g[idx] == g_val else 1.0
            
            # Precompute RFF dot products (doesn't depend on p)
            rff_dot_history = history_rff @ rff_z  # Shape: (history_len,)
        else:
            history_p = history_y = residuals = rff_dot_history = categorical_factors = None
        
        # Precompute self RFF dot product
        rff_self = np.dot(rff_z, rff_z)
        
        # Compute potential function S_t(p)
        def potential(p):
            """
            S_t(p) = sum_{i=1}^{t-1} k((z_t, p), (z_i, p_i)) * (y_i - p_i)
                     + (1/2) * k((z_t, p), (z_t, p)) * (1 - 2p)
            """
            s = 0.0
            if history_len > 0:
                # Vectorized computation: dot_products = rff_dot + p * p_history + bias_term
                dot_products = rff_dot_history + p * history_p + 1.0
                k_vals = dot_products * categorical_factors
                s += np.dot(k_vals, residuals)
            
            # Self-kernel term; categorical factor is always 2 when comparing with itself
            dot_self = rff_self + p * p + 1.0
            s += dot_self * (1 - 2 * p)
            
            return s
        
        # Early exit checks to avoid binary search when possible
        pot_1 = potential(1.0)
        if pot_1 >= 0:
            return 1.0
        
        pot_0 = potential(0.0)
        if pot_0 <= 0:
            return 0.0
        
        # Find p such that S_t(p) = 0
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
