import numpy as np
from sklearn.kernel_approximation import RBFSampler
from tqdm import tqdm
from catboost import CatBoostClassifier, Pool
from collections import deque

class K29:
    """
    Partially-online RFF + CatBoost.
    """
    def __init__(
        self,
        n_rff_features=100,
        gamma=1.0,
        random_state=None,
        update_every=250,
        trees_per_update=100,
        max_trees=1000,
        window_size=7000,          # rolling window size for rebuilds
        use_gpu=False,
        gpu_devices="4:5",
        catboost_params=None,
        float_dtype=np.float32,    # store cached RFFs in float32 by default
    ):
        self.n_rff_features = int(n_rff_features)
        self.gamma = float(gamma)
        self.random_state = random_state

        self.update_every = int(update_every)
        self.trees_per_update = int(trees_per_update)
        self.max_trees = int(max_trees)
        self.window_size = int(window_size)

        if self.update_every <= 0:
            raise ValueError("update_every must be > 0")
        if self.trees_per_update <= 0:
            raise ValueError("trees_per_update must be > 0")
        if self.max_trees <= 0:
            raise ValueError("max_trees must be > 0")
        if self.window_size <= 0:
            raise ValueError("window_size must be > 0")

        self.float_dtype = float_dtype

        self.rff = RBFSampler(
            n_components=self.n_rff_features,
            gamma=self.gamma,
            random_state=self.random_state,
        )
        self._rff_ready = False

        # CatBoost base params
        params = dict(
            loss_function="Logloss",
            eval_metric="Logloss",
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=self.random_state,
            verbose=False,
            allow_writing_files=False,
        )
        if use_gpu:
            params.update(dict(task_type="GPU", devices=gpu_devices))
        if catboost_params:
            params.update(catboost_params)
        self._cb_base_params = params

        self._cat_feature_idx = [self.n_rff_features]  # last column is categorical g

        self.model = None
        self.fitted_ = False

        # Update buffer (recent labeled points since last update)
        self._buf_rff = []
        self._buf_g = []
        self._buf_y = []
        self._buf_pos = 0
        self._buf_neg = 0

        # Rolling window cache (stores cached RFF so rebuilds don't recompute)
        self._win_rff = deque(maxlen=self.window_size)
        self._win_g = deque(maxlen=self.window_size)
        self._win_y = deque(maxlen=self.window_size)

    def featurize(self, x):
        """
        Returns cached features for single datapoint:
          rff_z: (D,) float_dtype
          g: int32
        """
        x = np.asarray(x)
        assert x.ndim == 1, "Expected a single sample x with shape (d,)"

        z = x[:-1]
        g = x[-1]
        if isinstance(g, np.ndarray):
            g = g.item()
        g = np.int32(g)

        if not self._rff_ready:
            self.rff.fit(np.asarray(z).reshape(1, -1))
            self._rff_ready = True

        rff_z = self.rff.transform(np.asarray(z).reshape(1, -1))[0].astype(self.float_dtype, copy=False)
        return rff_z, g

    def build_pool(self, rff_list, g_list, y_list):
        """
        Build CatBoost Pool. We only construct an object matrix at TRAIN time.
        """
        R = np.vstack(rff_list)  # (m, D)
        g = np.asarray(g_list, dtype=np.int32)
        y = np.asarray(y_list, dtype=np.int32)

        X = np.empty((R.shape[0], R.shape[1] + 1), dtype=object)
        X[:, :-1] = R
        X[:, -1] = g

        return Pool(X, y, cat_features=self._cat_feature_idx)

    def rebuild_model(self):
        """
        Rebuild a fixed-size (max_trees) model on the rolling window.
        """
        if len(self._win_y) < 2:
            return
        y_arr = np.asarray(self._win_y, dtype=np.int8)
        if y_arr.min() == y_arr.max(): # one one label present, skip
            return

        pool = self.build_pool(list(self._win_rff), list(self._win_g), list(self._win_y))
        m = CatBoostClassifier(**self._cb_base_params, iterations=self.max_trees)
        m.fit(pool)
        self.model = m
        self.fitted_ = True

    def update(self, force=False):
        """
        Every update_every points, train on the buffered chunk.

        - If model size + trees_per_update <= max_trees: extend via init_model on the chunk.
        - Else: rebuild from scratch with max_trees on the rolling window.
        """
        if (not force) and (len(self._buf_y) < self.update_every):
            return
        if len(self._buf_y) == 0:
            return

        # Need both classes in chunk to do an update; otherwise keep buffering.
        if not (self._buf_pos > 0 and self._buf_neg > 0):
            return

        # Push chunk into rolling window
        for rff_z, g, y in zip(self._buf_rff, self._buf_g, self._buf_y):
            self._win_rff.append(rff_z)
            self._win_g.append(g)
            self._win_y.append(y)

        chunk_pool = self.build_pool(self._buf_rff, self._buf_g, self._buf_y)

        if self.model is None:
            iters = min(self.trees_per_update, self.max_trees)
            self.model = CatBoostClassifier(**self._cb_base_params, iterations=iters)
            self.model.fit(chunk_pool)
            self.fitted_ = True
        else:
            current = 0 if self.model is None else int(getattr(self.model, "tree_count_", 0))
            proposed = current + self.trees_per_update
            if proposed <= self.max_trees:
                new_model = CatBoostClassifier(**self._cb_base_params, iterations=self.trees_per_update)
                new_model.fit(chunk_pool, init_model=self.model)
                self.model = new_model
                self.fitted_ = True
            else:
                self.rebuild_model()

        # clear buffer
        self._buf_rff.clear()
        self._buf_g.clear()
        self._buf_y.clear()
        self._buf_pos = 0
        self._buf_neg = 0

    def step(self, x, y=None):
        """
        One streaming step:
          - compute/cache RFF
          - predict p
          - if y provided: buffer label and possibly update
        """
        rff_z, g = self.featurize(x)

        if not self.fitted_:
            p = 0.5
        else:
            Xrow = np.empty((1, self.n_rff_features + 1), dtype=object)
            Xrow[0, :-1] = rff_z
            Xrow[0, -1] = g
            p = float(self.model.predict_proba(Xrow)[:, 1][0])

        if y is not None:
            y_i = np.int8(int(y))
            self._buf_rff.append(rff_z)
            self._buf_g.append(g)
            self._buf_y.append(y_i)
            if y_i == 1:
                self._buf_pos += 1
            else:
                self._buf_neg += 1

            self.update(force=False)

        return p

    def fit(self, X, y, return_preds=False, finalize=True, show_progress=True):
        X = np.asarray(X)
        y = np.asarray(y)

        preds = [] if return_preds else None
        for i in tqdm(range(len(y))):
            p = self.step(X[i], y=y[i])
            if return_preds:
                preds.append(p)

        if finalize:
            self.update(force=True)

        if return_preds:
            return self, np.asarray(preds, dtype=float)
        return self

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            return self.step(X, y=None)
        out = np.empty((X.shape[0],), dtype=float)
        for i in range(X.shape[0]):
            out[i] = self.step(X[i], y=None)
        return out