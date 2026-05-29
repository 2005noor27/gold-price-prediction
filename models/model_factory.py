"""
models/model_factory.py
=======================
Factory functions for creating ML model instances.
All imports are deferred so missing optional packages
(xgboost, lightgbm) fail gracefully.
"""

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


def make_model(name: str):
    """
    Return an unfitted sklearn-compatible estimator for the given name.

    Supported names
    ---------------
    "Random Forest"       RandomForestRegressor
    "Linear Regression"   LinearRegression
    "XGBoost"             XGBRegressor  (requires xgboost)
    "LightGBM"            LGBMRegressor (requires lightgbm)
    "Neural Network (MLP)" MLPRegressor (sklearn)
    """
    if name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1
        )

    if name == "Linear Regression":
        return LinearRegression()

    if name == "XGBoost":
        try:
            from xgboost import XGBRegressor
            return XGBRegressor(
                n_estimators=300, learning_rate=0.05,
                max_depth=6, random_state=42, verbosity=0,
            )
        except ImportError:
            raise ImportError("XGBoost is not installed. Run: pip install xgboost")

    if name == "LightGBM":
        try:
            from lightgbm import LGBMRegressor
            return LGBMRegressor(
                n_estimators=500, learning_rate=0.03,
                num_leaves=63, min_child_samples=20,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbose=-1,
            )
        except ImportError:
            raise ImportError("LightGBM is not installed. Run: pip install lightgbm")

    if name == "Neural Network (MLP)":
        from sklearn.neural_network import MLPRegressor
        return MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu", solver="adam",
            max_iter=500, random_state=42,
            early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=20,
        )

    raise ValueError(f"Unknown model name: '{name}'. "
                     f"Choose from: Random Forest, Linear Regression, "
                     f"XGBoost, LightGBM, Neural Network (MLP)")


TREE_MODELS = {"Random Forest", "XGBoost", "LightGBM"}
NEEDS_SCALING = {"Neural Network (MLP)"}
