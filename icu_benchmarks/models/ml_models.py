import gin
import lightgbm as lgbm
import numpy as np
import wandb
from catboost import CatBoostClassifier as CatBoostModel, Pool
from sklearn import linear_model
from sklearn import ensemble
from sklearn import neural_network
from sklearn import svm
from icu_benchmarks.models.wrappers import MLWrapper
from icu_benchmarks.contants import RunMode
from wandb.lightgbm import wandb_callback


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

    def __init__(self, *args, **kwargs):
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
