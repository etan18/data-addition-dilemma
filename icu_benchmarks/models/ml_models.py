import gin
import logging
import lightgbm as lgbm
import numpy as np
import wandb
from pathlib import Path
from typing import Optional
from catboost import CatBoostClassifier as CatBoostModel, Pool
from joblib import dump
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
        params = self.model.get_params()
        if not params.get("eval_metric"):
            # Ensure Accuracy is tracked alongside the loss for curve logging.
            self.model.set_params(eval_metric="Accuracy")

        self.model.fit(
            train_pool,
            eval_set=val_pool,
            verbose=True,
            use_best_model=True,
            early_stopping_rounds=self.hparams.patience,
        )

        metrics_logger = getattr(self, "metrics_logger", None)
        if metrics_logger is not None:
            evals_result = self.model.get_evals_result() or {}
            split_map = {"learn": "train", "validation_0": "val"}
            for split_key, metrics in evals_result.items():
                split_name = split_map.get(split_key, split_key)
                for metric_name, values in metrics.items():
                    metric_key = f"curve/{split_name}_{metric_name}".lower()
                    for step_idx, value in enumerate(values):
                        metrics_logger.log_metrics({metric_key: float(value)}, step=step_idx)

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
        pos_weight: float = 16.0,
        neg_weight: float = 1.0,
        categorical_index: Optional[int] = -1,
        test_hospital_id: Optional = None,
        log_curve: bool = False,
        log_every: int = 1,
        wandb_log_hparams: bool = True,
        wandb_metadata: Optional[dict] = None,
        **kwargs,
    ):
        self.categorical_index = categorical_index
        self.n_rff_features = int(n_rff_features)
        if self.n_rff_features < 1:
            raise ValueError("n_rff_features must be at least 1.")
        self.log_curve = bool(log_curve)
        self.log_every = int(log_every)
        self.wandb_log_hparams = bool(wandb_log_hparams)
        self.wandb_metadata = wandb_metadata
        self.model = K29(
            n_rff_features=self.n_rff_features,
            gamma=gamma,
            random_state=random_state,
            test_hospital_id=test_hospital_id,
            pos_weight=pos_weight,
            neg_weight=neg_weight,
        )
        super().__init__(*args, **kwargs)

    def _log_wandb_config(self):
        if not self.wandb_log_hparams or wandb.run is None:
            return
        config = {
            "model": "K29",
            "n_rff_features": self.n_rff_features,
            "gamma": self.model.gammas,
            "random_state": self.model.random_state,
            "test_hospital_id": self.model.test_hospital_id,
            "pos_weight": self.model.pos_weight,
            "neg_weight": self.model.neg_weight,
            "categorical_index": self.categorical_index,
            "log_curve": self.log_curve,
            "log_every": self.log_every,
        }
        if isinstance(self.wandb_metadata, dict):
            config["metadata"] = self.wandb_metadata
        try:
            wandb.config.update(config, allow_val_change=True)
        except Exception:
            logging.warning("wandb config update failed for K29.", exc_info=True)

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
        self._log_wandb_config()
        metrics_logger = getattr(self, "metrics_logger", None)
        if self.log_curve and metrics_logger is not None:
            self.model.fit(
                prepared_train,
                train_labels,
                log_curve=True,
                log_every=self.log_every,
                metrics_logger=metrics_logger,
            )
        else:
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

    def save_model(self, save_path, file_name, file_extension=".joblib"):
        """Persist the full wrapper so eval-only can restore expected methods."""
        path = save_path / (file_name + file_extension)
        try:
            dump(self, path)
            logging.info(f"Model saved to {str(path.resolve())}.")
        except Exception as e:
            logging.error(f"Cannot save model to path {str(path.resolve())}: {e}.")


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
