"""Small, reusable building blocks for leakage-safe regression modeling."""

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "hc_mortgage_mean"
NUMERIC_FEATURES = [
    "pop",
    "family_mean",
    "second_mortgage",
    "home_equity",
    "debt",
    "hs_degree",
    "age_median",
    "pct_own",
    "married",
    "separated",
    "divorced",
]
CATEGORICAL_FEATURES = ["STATEID", "type"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_datasets(data_dir: str | Path = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the repository's original train and test CSV files."""
    data_dir = Path(data_dir)
    return pd.read_csv(data_dir / "train.csv"), pd.read_csv(data_dir / "test.csv")


def add_age_median(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the mean of male and female median ages.

    This reproduces the historical notebook's definition. Pandas skips neither
    input here: if one sex-specific median is missing, the result stays missing
    for the pipeline's numeric imputer to handle.
    """
    required = {"male_age_median", "female_age_median"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Cannot derive age_median; missing columns: {sorted(missing)}")
    result = data.copy()
    result["age_median"] = (
        result["male_age_median"] + result["female_age_median"]
    ) / 2
    return result


def _select_model_features(data: pd.DataFrame) -> pd.DataFrame:
    """Engineer and select predictors without consulting the target column."""
    engineered = add_age_median(data)
    missing_features = set(MODEL_FEATURES).difference(engineered.columns)
    if missing_features:
        raise ValueError(f"Missing model features: {sorted(missing_features)}")
    features = engineered[MODEL_FEATURES].copy()
    # IDs represent labels, not quantities; strings make that intent explicit.
    features["STATEID"] = features["STATEID"].map(
        lambda value: str(value) if pd.notna(value) else np.nan
    )
    return features


def audit_split_overlap(
    train_data: pd.DataFrame, holdout_data: pd.DataFrame
) -> dict[str, int | None]:
    """Audit UID and exact predictor overlap without changing either split.

    Feature hashes avoid an expensive Cartesian comparison. The target is not
    read or included, so the result is strictly a dataset-integrity diagnostic
    and cannot expose holdout outcomes during model selection.
    """
    if "UID" in train_data.columns and "UID" in holdout_data.columns:
        train_uids = set(train_data["UID"].dropna().unique())
        holdout_uids = set(holdout_data["UID"].dropna().unique())
        overlapping_uids: int | None = len(train_uids.intersection(holdout_uids))
    else:
        overlapping_uids = None

    indicators = holdout_overlap_indicators(train_data, holdout_data)
    return {
        "overlapping_uid_values": overlapping_uids,
        "holdout_rows_matching_train_features": int(
            indicators["matching_features"].sum()
        ),
    }


def holdout_overlap_indicators(
    train_data: pd.DataFrame, holdout_data: pd.DataFrame
) -> pd.DataFrame:
    """Flag holdout rows with a shared UID and/or matching predictor vector.

    The returned frame preserves the holdout index. It intentionally ignores
    target values and does not mutate or filter either input data frame.
    """
    shared_uid = pd.Series(False, index=holdout_data.index, dtype=bool)
    if "UID" in train_data.columns and "UID" in holdout_data.columns:
        train_uids = set(train_data["UID"].dropna().unique())
        shared_uid = holdout_data["UID"].notna() & holdout_data["UID"].isin(
            train_uids
        )

    train_features = _select_model_features(train_data)
    holdout_features = _select_model_features(holdout_data)
    train_hashes = pd.util.hash_pandas_object(train_features, index=False)
    holdout_hashes = pd.util.hash_pandas_object(holdout_features, index=False)
    matching_features = holdout_hashes.isin(set(train_hashes))
    matching_features.index = holdout_data.index

    indicators = pd.DataFrame(
        {
            "shared_uid": shared_uid,
            "matching_features": matching_features,
        },
        index=holdout_data.index,
    )
    indicators["any_overlap"] = indicators.any(axis="columns")
    return indicators


def prepare_supervised_data(
    data: pd.DataFrame, *, drop_duplicates: bool = True
) -> tuple[pd.DataFrame, pd.Series, dict[str, int]]:
    """Engineer features, remove unavailable targets, and return X, y, counts."""
    if TARGET not in data.columns:
        raise ValueError(f"Required target column {TARGET!r} is unavailable")
    engineered = add_age_median(data)

    missing_target_rows = int(engineered[TARGET].isna().sum())
    supervised = engineered.loc[engineered[TARGET].notna()].copy()
    duplicate_rows = int(supervised.duplicated().sum())
    if drop_duplicates:
        supervised = supervised.drop_duplicates()

    features = _select_model_features(supervised)
    target = supervised[TARGET].astype(float)
    audit = {
        "input_rows": len(data),
        "missing_target_rows_removed": missing_target_rows,
        "duplicate_rows_removed": duplicate_rows if drop_duplicates else 0,
        "modeling_rows": len(supervised),
    }
    return features, target, audit


def build_preprocessor() -> ColumnTransformer:
    """Build separate imputation/scaling and imputation/encoding branches."""
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES),
         ("categorical", categorical, CATEGORICAL_FEATURES)]
    )


def candidate_models(random_state: int = 42) -> Mapping[str, object]:
    """Return the intentionally small comparison set used by the notebook."""
    return {
        "DummyRegressor": DummyRegressor(strategy="mean"),
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def build_pipeline(model: object) -> Pipeline:
    """Keep preprocessing and estimation together to prevent leakage."""
    return Pipeline([("preprocessor", build_preprocessor()), ("model", model)])


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate the three regression metrics used throughout the project."""
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def compare_models(
    features: pd.DataFrame, target: pd.Series, random_state: int = 42
) -> pd.DataFrame:
    """Compare candidates using five training-only, shuffled folds."""
    folds = KFold(n_splits=5, shuffle=True, random_state=random_state)
    scoring = {
        "r2": "r2",
        "mse": "neg_mean_squared_error",
        "mae": "neg_mean_absolute_error",
    }
    rows = []
    for name, model in candidate_models(random_state).items():
        scores = cross_validate(
            build_pipeline(model), features, target, cv=folds, scoring=scoring
        )
        fold_rmse = np.sqrt(-scores["test_mse"])
        rows.append(
            {
                "model": name,
                "r2_mean": scores["test_r2"].mean(),
                "r2_std": scores["test_r2"].std(),
                "rmse_mean": fold_rmse.mean(),
                "rmse_std": fold_rmse.std(),
                "mae_mean": (-scores["test_mae"]).mean(),
                "mae_std": (-scores["test_mae"]).std(),
            }
        )
    return pd.DataFrame(rows).sort_values("rmse_mean").reset_index(drop=True)


def state_diagnostics(
    state_ids: pd.Series,
    y_true: pd.Series,
    y_pred: np.ndarray,
    minimum_observations: int = 30,
) -> pd.DataFrame:
    """Summarize holdout errors for sufficiently represented states only."""
    if minimum_observations < 2:
        raise ValueError("minimum_observations must be at least 2")
    frame = pd.DataFrame(
        {
            "STATEID": state_ids.to_numpy(),
            "actual": np.asarray(y_true),
            "predicted": np.asarray(y_pred),
        }
    )
    rows = []
    for state_id, group in frame.groupby("STATEID", dropna=False):
        if len(group) < minimum_observations:
            continue
        metrics = regression_metrics(group["actual"], group["predicted"])
        rows.append({"STATEID": state_id, "observations": len(group), **metrics})
    columns = ["STATEID", "observations", "r2", "rmse", "mae"]
    return pd.DataFrame(rows, columns=columns).sort_values(
        "rmse", ignore_index=True
    )
