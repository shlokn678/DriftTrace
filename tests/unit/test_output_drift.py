"""Unit tests for prediction/output drift, especially categorical (classifier) output.

Covers the fix for the 'prediction (output)' card showing INSUFFICIENT DATA:
- a classifier's output baseline is CATEGORICAL (class-frequency PSI, no numeric KS),
- the live window actually collects the output value from the event (not ev.features),
- missing reference predictions -> no output node (output monitoring unavailable),
- too few window samples -> INSUFFICIENT_DATA (no fabricated metric).
"""

from __future__ import annotations

import pytest

from drifttrace.bundle.reference import (
    OUTPUT_NODE,
    FeatureProfile,
    ReferenceProfile,
    baseline_from_profile,
    output_label,
)
from drifttrace.drift.config import DriftConfig, DriftThresholds
from drifttrace.drift.engine import detect_drift
from drifttrace.drift.verdict import Verdict


def _config(min_samples: int = 30) -> DriftConfig:
    return DriftConfig(thresholds=DriftThresholds(), min_samples=min_samples, psi_bins=10)


def _profile() -> ReferenceProfile:
    # One numeric feature; the output node is added separately via output_values.
    fp = FeatureProfile(
        name="x", kind="continuous", count=100, values=[float(i) for i in range(100)]
    )
    return ReferenceProfile(feature_names=["x"], features={"x": fp}, n_rows=100)


# --------------------------------------------------------------------------------------
# output_label canonicalization
# --------------------------------------------------------------------------------------
@pytest.mark.unit
def test_output_label_canonicalizes_numeric_class_ids() -> None:
    # 0.0 and 0 must map to the same class label so reference and window agree.
    assert output_label(0.0) == "0"
    assert output_label(0) == "0"
    assert output_label(1.0) == "1"
    assert output_label("malignant") == "malignant"
    assert output_label(2.5) == "2.5"  # non-integer kept as-is


# --------------------------------------------------------------------------------------
# categorical output baseline
# --------------------------------------------------------------------------------------
@pytest.mark.unit
def test_classifier_output_baseline_is_categorical() -> None:
    # Reference predictions are class ids 0/1 (as floats, like the standardized event).
    outputs = [0.0] * 60 + [1.0] * 40
    baseline = baseline_from_profile(
        _profile(), model_version="m", output_values=outputs, output_kind="categorical"
    )
    node = baseline.nodes[OUTPUT_NODE]
    assert node.kind == "categorical"
    assert node.count == 100
    # Frequencies, keyed by canonical class label.
    assert node.categories["0"] == pytest.approx(0.6)
    assert node.categories["1"] == pytest.approx(0.4)


@pytest.mark.unit
def test_regressor_output_baseline_is_continuous() -> None:
    outputs = [float(i) for i in range(100)]
    baseline = baseline_from_profile(
        _profile(), model_version="m", output_values=outputs, output_kind="continuous"
    )
    assert baseline.nodes[OUTPUT_NODE].kind == "continuous"
    assert baseline.nodes[OUTPUT_NODE].values == outputs


# --------------------------------------------------------------------------------------
# categorical output drift via class-frequency PSI (no KS for class labels)
# --------------------------------------------------------------------------------------
@pytest.mark.unit
def test_categorical_output_stable_vs_drift() -> None:
    outputs = [0.0] * 60 + [1.0] * 40  # reference: 60% class 0, 40% class 1
    baseline = baseline_from_profile(
        _profile(), model_version="m", output_values=outputs, output_kind="categorical"
    )
    cfg = _config(min_samples=30)

    # Stable: window matches the reference class mix.
    stable_window = ["0"] * 60 + ["1"] * 40
    rep = detect_drift({}, {OUTPUT_NODE: stable_window}, baseline, cfg, window_id="w0")
    node = rep.nodes[OUTPUT_NODE]
    assert node.ks is None  # KS is NOT applied to categorical class labels
    assert node.psi is not None
    assert node.verdict == str(Verdict.STABLE)

    # Drift: the class mix flips hard (mostly class 1 now).
    drift_window = ["0"] * 5 + ["1"] * 95
    rep2 = detect_drift({}, {OUTPUT_NODE: drift_window}, baseline, cfg, window_id="w1")
    node2 = rep2.nodes[OUTPUT_NODE]
    assert node2.ks is None
    assert node2.psi is not None and node2.psi["psi"] is not None
    assert node2.verdict == str(Verdict.DRIFT)


@pytest.mark.unit
def test_categorical_output_zero_frequency_class_is_safe() -> None:
    # Reference has a class the window never produces; PSI must stay finite (eps-smoothed),
    # not infinite/NaN, and still flag the large shift.
    outputs = [0.0] * 50 + [1.0] * 50
    baseline = baseline_from_profile(
        _profile(), model_version="m", output_values=outputs, output_kind="categorical"
    )
    window = ["0"] * 100  # class 1 absent entirely
    rep = detect_drift({}, {OUTPUT_NODE: window}, baseline, _config(), window_id="w0")
    psi = rep.nodes[OUTPUT_NODE].psi
    assert psi is not None and psi["psi"] is not None
    import math

    assert math.isfinite(psi["psi"])
    assert rep.nodes[OUTPUT_NODE].verdict == str(Verdict.DRIFT)


@pytest.mark.unit
def test_categorical_output_insufficient_samples() -> None:
    outputs = [0.0] * 60 + [1.0] * 40
    baseline = baseline_from_profile(
        _profile(), model_version="m", output_values=outputs, output_kind="categorical"
    )
    # Fewer than min_samples window observations -> INSUFFICIENT_DATA, no fabricated PSI.
    rep = detect_drift(
        {}, {OUTPUT_NODE: ["0", "1", "0"]}, baseline, _config(min_samples=30), window_id="w0"
    )
    node = rep.nodes[OUTPUT_NODE]
    assert node.verdict == str(Verdict.INSUFFICIENT_DATA)
    assert node.psi is not None and node.psi["psi"] is None  # no value fabricated


# --------------------------------------------------------------------------------------
# missing reference predictions -> no output node at all
# --------------------------------------------------------------------------------------
@pytest.mark.unit
def test_no_reference_outputs_means_no_output_node() -> None:
    baseline = baseline_from_profile(_profile(), model_version="m", output_values=None)
    assert OUTPUT_NODE not in baseline.nodes  # output monitoring unavailable, not faked
