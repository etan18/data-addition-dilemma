import gin
import lightgbm as lgbm
import numpy as np
import wandb
from pathlib import Path
from typing import Optional
from catboost import CatBoostClassifier as CatBoostModel, Pool
from sklearn import linear_model
from sklearn import ensemble
from sklearn import neural_network
from sklearn import svm
from icu_benchmarks.models.wrappers import MLWrapper
from icu_benchmarks.contants import RunMode
from icu_benchmarks.models.utils import export_catboost_feature_artifacts
from wandb.integration.lightgbm import wandb_callback
from defensive_forecasting.k29_algorithm import K29


class LGBMWrapper(MLWrapper):
    def fit_model(self, train_data, train_labels, val_data, val_labels):
        """Fitting function for LGBM models."""
        self.model.set_params(random_state=np.random.get_state()[1][0])
        callbacks = [lgbm.early_stopping(self.hparams.patience, verbose=True), lgbm.log_evaluation(period=-1)]

        if wandb.run is not None:
            callbacks.append(wandb_callback())

        self.model = self.model.fit(
            train_data,
            train_labels,
            eval_set=(val_data, val_labels),
            verbose=True,
            callbacks=callbacks,
        )
        val_loss = list(self.model.best_score_["valid_0"].values())[0]
        return val_loss


@gin.configurable
class LGBMClassifier(LGBMWrapper):
    _supported_run_modes = [RunMode.classification]

    def __init__(self, *args, **kwargs):
        self.model = self.set_model_args(lgbm.LGBMClassifier, *args, **kwargs)
        super().__init__(*args, **kwargs)

    def predict(self, features):
        """Predicts labels for the given features."""
        return self.model.predict_proba(features)


@gin.configurable
class LGBMRegressor(LGBMWrapper):
    _supported_run_modes = [RunMode.regression]

    def __init__(self, *args, **kwargs):
        self.model = self.set_model_args(lgbm.LGBMRegressor, *args, **kwargs)
        super().__init__(*args, **kwargs)


@gin.configurable
class CatBoostClassifier(MLWrapper):
    _supported_run_modes = [RunMode.classification]

    def __init__(
        self,
        *args,
        log_feature_artifacts: bool = True,
        feature_artifact_subdir: str = "feature_analysis",
        save_catboost_model_json: bool = False,
        **kwargs,
    ):
        self.log_feature_artifacts = log_feature_artifacts
        self.feature_artifact_subdir = feature_artifact_subdir
        self.save_catboost_model_json = save_catboost_model_json
        self.model = self.set_model_args(CatBoostModel, *args, **kwargs)
        super().__init__(*args, **kwargs)

    def fit_model(self, train_data, train_labels, val_data, val_labels):
        """Train CatBoost with early stopping and return validation loss."""
        train_pool = Pool(train_data, label=train_labels)
        val_pool = Pool(val_data, label=val_labels)

        # Keep runs reproducible but still leverage gin-configured params.
        self.model.set_params(random_seed=int(np.random.get_state()[1][0]))

        self.model.fit(
            train_pool,
            eval_set=val_pool,
            verbose=True,
            use_best_model=True,
            early_stopping_rounds=self.hparams.patience,
        )

        val_pred = self.model.predict_proba(val_data)
        return self.loss(val_labels, val_pred)

    def predict(self, features):
        return self.model.predict_proba(features)

    def save_model(self, save_path, file_name, file_extension=".joblib"):
        super().save_model(save_path, file_name, file_extension)
        if self.log_feature_artifacts:
            artifact_dir = Path(save_path) / self.feature_artifact_subdir
            export_catboost_feature_artifacts(
                self.model,
                getattr(self, "trained_columns", None),
                artifact_dir,
                save_model_json=self.save_catboost_model_json,
            )


@gin.configurable
class K29Classifier(MLWrapper):
    _supported_run_modes = [RunMode.classification]

    def __init__(
        self,
        *args,
        n_rff_features: int = 100,
        gamma: float = 1.0,
        random_state: int = None,
        categorical_index: Optional[int] = -1,
        test_hospital_id: Optional = None,
        **kwargs,
    ):
        self.categorical_index = categorical_index
        self.n_rff_features = int(n_rff_features)
        if self.n_rff_features < 1:
            raise ValueError("n_rff_features must be at least 1.")
        self.model = K29(
            n_rff_features=self.n_rff_features,
            gamma=gamma,
            random_state=random_state,
            # test_hospital_id=test_hospital_id,
        )
        super().__init__(*args, **kwargs)

    def _prepare_features(self, features):
        """Move the configured categorical column to the last position for K29."""
        if self.categorical_index is None:
            return np.asarray(features)

        arr = np.asarray(features)
        cat_idx = self.categorical_index if self.categorical_index >= 0 else arr.shape[-1] + self.categorical_index
        if cat_idx < 0 or cat_idx >= arr.shape[-1]:
            raise ValueError(f"categorical_index {self.categorical_index} out of bounds for input with shape {arr.shape}.")

        if arr.ndim == 1:
            if cat_idx == arr.shape[0] - 1:
                return arr
            categorical = np.array([arr[cat_idx]])
            continuous = np.delete(arr, cat_idx, axis=0)
            return np.concatenate([continuous, categorical], axis=0)

        if cat_idx == arr.shape[1] - 1:
            return arr

        categorical = arr[:, cat_idx][:, None]
        continuous = np.delete(arr, cat_idx, axis=1)
        return np.concatenate([continuous, categorical], axis=1)

    def fit_model(self, train_data, train_labels, val_data, val_labels):
        prepared_train = self._prepare_features(train_data)
        self.model.fit(prepared_train, train_labels)
        val_pred = self.predict(val_data)
        return self.loss(val_labels, val_pred)

    def predict(self, features):
        prepared_features = self._prepare_features(features)
        positive_prob = np.asarray(self.model.predict(prepared_features), dtype=float)
        if positive_prob.ndim == 0:
            positive_prob = np.array([positive_prob])
        negative_prob = 1.0 - positive_prob
        stacked = np.vstack([negative_prob, positive_prob]).T
        return stacked


# Scikit-learn models
@gin.configurable
class LogisticRegression(MLWrapper):
    __supported_run_modes = [RunMode.classification]

    def __init__(self, *args, **kwargs):
        self.model = self.set_model_args(linear_model.LogisticRegression, *args, **kwargs)
        super().__init__(*args, **kwargs)


@gin.configurable()
class LinearRegression(MLWrapper):
    _supported_run_modes = [RunMode.regression]

    def __init__(self, *args, **kwargs):
        self.model = self.set_model_args(linear_model.LinearRegression, *args, **kwargs)
        super().__init__(*args, **kwargs)


@gin.configurable()
class ElasticNet(MLWrapper):
    _supported_run_modes = [RunMode.regression]

    def __init__(self, *args, **kwargs):
        self.model = self.set_model_args(linear_model.ElasticNet, *args, **kwargs)
        super().__init__(*args, **kwargs)


@gin.configurable
class RFClassifier(MLWrapper):
    _supported_run_modes = [RunMode.classification]

    def __init__(self, *args, **kwargs):
        self.model = self.set_model_args(ensemble.RandomForestClassifier, *args, **kwargs)
        super().__init__(*args, **kwargs)


@gin.configurable
class SVMClassifier(MLWrapper):
    _supported_run_modes = [RunMode.classification]

    def __init__(self, *args, **kwargs):
        self.model = self.model_args(svm.SVC, *args, **kwargs)
        super().__init__(*args, **kwargs)


@gin.configurable
class SVMRegressor(MLWrapper):
    _supported_run_modes = [RunMode.regression]

    def __init__(self, *args, **kwargs):
        self.model = self.model_args(svm.SVR, *args, **kwargs)
        super().__init__(*args, **kwargs)


@gin.configurable
class PerceptronClassifier(MLWrapper):
    _supported_run_modes = [RunMode.classification]

    def __init__(self, *args, **kwargs):
        self.model = self.set_model_args(neural_network.MLPClassifier, *args, **kwargs)
        super().__init__(*args, **kwargs)


@gin.configurable
class MLPClassifier(MLWrapper):
    _supported_run_modes = [RunMode.classification]

    def __init__(self, *args, **kwargs):
        self.model = self.set_model_args(neural_network.MLPClassifier, *args, **kwargs)
        super().__init__(*args, **kwargs)


class MLPRegressor(MLWrapper):
    _supported_run_modes = [RunMode.regression]

    def __init__(self, *args, **kwargs):
        self.model = self.set_model_args(neural_network.MLPRegressor, *args, **kwargs)
        super().__init__(*args, **kwargs)
