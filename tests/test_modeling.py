import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from src.modeling import (
    MODEL_FEATURES,
    TARGET,
    add_age_median,
    audit_split_overlap,
    build_pipeline,
    holdout_overlap_indicators,
    prepare_supervised_data,
    regression_metrics,
    state_diagnostics,
)


def sample_frame() -> pd.DataFrame:
    rows = []
    for index in range(8):
        rows.append(
            {
                "male_age_median": 30 + index,
                "female_age_median": 40 + index,
                "pop": 100 + index,
                "family_mean": 50_000 + index * 100,
                "second_mortgage": 1.0 + index,
                "home_equity": 2.0 + index,
                "debt": 3.0 + index,
                "hs_degree": 70.0 + index,
                "pct_own": 60.0 + index,
                "married": 50.0 + index,
                "separated": 4.0 + index,
                "divorced": 8.0 + index,
                "STATEID": 1 if index < 4 else 2,
                "type": "City" if index % 2 else "Town",
                TARGET: 900.0 + index * 10,
            }
        )
    return pd.DataFrame(rows)


def test_age_median_is_arithmetic_mean():
    result = add_age_median(sample_frame())
    assert result.loc[0, "age_median"] == 35.0


def test_preparation_removes_missing_target_and_returns_required_features():
    frame = sample_frame()
    frame.loc[0, TARGET] = np.nan
    features, target, audit = prepare_supervised_data(frame)
    assert list(features.columns) == MODEL_FEATURES
    assert len(target) == 7
    assert audit["missing_target_rows_removed"] == 1


def test_split_overlap_counts_uids_and_matching_predictor_rows_without_target():
    train = sample_frame().iloc[:3].drop(columns=TARGET)
    holdout = sample_frame().iloc[2:5].drop(columns=TARGET).copy()
    train["UID"] = [10, 11, 12]
    holdout["UID"] = [12, 13, 14]

    overlap = audit_split_overlap(train, holdout)

    assert overlap == {
        "overlapping_uid_values": 1,
        "holdout_rows_matching_train_features": 1,
    }


def test_overlap_indicators_flag_each_reason_and_count_union_once():
    train = sample_frame().iloc[:3].copy()
    holdout = sample_frame().iloc[2:6].copy()
    train["UID"] = [10, 11, 12]
    # First row matches both, second only shares UID, third only matches features,
    # and fourth matches neither.
    holdout["UID"] = [12, 11, 14, 15]
    holdout.loc[holdout.index[1], "pop"] = 999
    raw_predictors = [column for column in train.columns if column not in {"UID", TARGET}]
    holdout.loc[holdout.index[2], raw_predictors] = train.loc[
        train.index[0], raw_predictors
    ].to_numpy()

    indicators = holdout_overlap_indicators(train, holdout)

    assert indicators["shared_uid"].tolist() == [True, True, False, False]
    assert indicators["matching_features"].tolist() == [True, False, True, False]
    assert indicators["any_overlap"].tolist() == [True, True, True, False]
    assert indicators["any_overlap"].sum() == 3


def test_overlap_indicators_ignore_target_values():
    train = sample_frame().iloc[:3].copy()
    holdout = sample_frame().iloc[2:5].copy()
    train["UID"] = [10, 11, 12]
    holdout["UID"] = [12, 13, 14]
    before = holdout_overlap_indicators(train, holdout)

    train[TARGET] = -1_000_000
    holdout[TARGET] = 1_000_000
    after = holdout_overlap_indicators(train, holdout)

    pd.testing.assert_frame_equal(before, after)


def test_pipeline_predicts_with_unknown_categories():
    features, target, _ = prepare_supervised_data(sample_frame())
    pipeline = build_pipeline(Ridge())
    pipeline.fit(features, target)
    unseen = features.iloc[[0]].copy()
    unseen["STATEID"] = "99"
    unseen["type"] = "Village"
    assert pipeline.predict(unseen).shape == (1,)


def test_regression_metrics_have_expected_values():
    metrics = regression_metrics(
        pd.Series([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])
    )
    assert metrics == {"r2": 1.0, "rmse": 0.0, "mae": 0.0}


def test_state_diagnostics_applies_minimum_observation_threshold():
    states = pd.Series([1, 1, 1, 2, 2])
    actual = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    diagnostics = state_diagnostics(states, actual, actual.to_numpy(), 3)
    assert diagnostics["STATEID"].tolist() == [1]
    assert diagnostics.loc[0, "observations"] == 3


def test_state_diagnostics_rejects_tiny_threshold():
    with pytest.raises(ValueError, match="at least 2"):
        state_diagnostics(pd.Series([1]), pd.Series([1.0]), np.array([1.0]), 1)
