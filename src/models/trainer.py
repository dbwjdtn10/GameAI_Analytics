"""모델 학습 모듈."""

import optuna
from lightgbm import LGBMClassifier
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.config import RANDOM_STATE

optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_baseline_model() -> LogisticRegression:
    """Baseline: Logistic Regression."""
    return LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )


def get_xgboost_model(**kwargs) -> XGBClassifier:
    """XGBoost 모델."""
    defaults = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE,
        "eval_metric": "logloss",
        "verbosity": 0,
    }
    defaults.update(kwargs)
    return XGBClassifier(**defaults)


def get_lgbm_model(**kwargs) -> LGBMClassifier:
    """LightGBM 모델."""
    defaults = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE,
        "verbosity": -1,
    }
    defaults.update(kwargs)
    return LGBMClassifier(**defaults)


def get_ensemble_voting(xgb_params: dict = None, lgbm_params: dict = None) -> VotingClassifier:
    """Soft Voting 앙상블."""
    xgb = get_xgboost_model(**(xgb_params or {}))
    lgbm = get_lgbm_model(**(lgbm_params or {}))
    lr = get_baseline_model()

    return VotingClassifier(
        estimators=[("xgb", xgb), ("lgbm", lgbm), ("lr", lr)],
        voting="soft",
    )


def get_ensemble_stacking(xgb_params: dict = None, lgbm_params: dict = None) -> StackingClassifier:
    """Stacking 앙상블."""
    xgb = get_xgboost_model(**(xgb_params or {}))
    lgbm = get_lgbm_model(**(lgbm_params or {}))

    return StackingClassifier(
        estimators=[("xgb", xgb), ("lgbm", lgbm)],
        final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        cv=5,
        passthrough=False,
    )


def tune_xgboost(X_train, y_train, X_val, y_val, n_trials: int = 50) -> dict:
    """Optuna로 XGBoost 하이퍼파라미터 튜닝."""
    from sklearn.metrics import roc_auc_score

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
        }
        model = get_xgboost_model(**params)
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_pred_proba)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    return study.best_params


def tune_lgbm(X_train, y_train, X_val, y_val, n_trials: int = 50) -> dict:
    """Optuna로 LightGBM 하이퍼파라미터 튜닝."""
    from sklearn.metrics import roc_auc_score

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
        }
        model = get_lgbm_model(**params)
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_pred_proba)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    return study.best_params
