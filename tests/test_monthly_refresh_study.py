import json

import numpy as np
import pandas as pd
import pytest

from research import monthly_refresh_study as study
from test_premium_meta_models import repeated_data


def prepared():
    frame = study.path_study.enrich(repeated_data())
    for i, name in enumerate(study.path_study.PATH_FEATURES):
        frame[name] = np.sin(np.arange(len(frame))*.01+i)
    return frame


def test_monthly_schedule_is_fixed_six_month_calibration_and_excludes_reserved_days():
    folds = study.monthly_folds()
    assert len(folds)==10
    assert study.longer_fold(folds[0])==("2026-01", "2025-07-01", "2025-12-31", "2026-01-08", "2026-01-31")
    assert study.longer_fold(folds[-1])==("2026-10", "2026-04-01", "2026-09-30", "2026-10-01", "2026-10-13")
    frame = prepared()
    last = frame.iloc[-1:].copy()
    last.index = [999999]
    last["trading_day"] = pd.Timestamp("2026-10-14").date()
    assert 999999 not in study.evaluation_rows(pd.concat([frame, last])).index


def test_current_month_outcomes_and_future_rows_cannot_change_earlier_month_model_scores():
    frame = prepared()
    fold = study.monthly_folds()[3]  # April2026; fit beforeOct2025, calOct-Mar.
    val, scores, detail = study.path_study.block(frame, fold, study.PROFILE)
    modified = frame.copy()
    current = modified.trading_day>=pd.Timestamp("2026-04-01").date()
    future = modified.trading_day>=pd.Timestamp("2026-05-01").date()
    modified.loc[current, "label"] = 1-modified.loc[current, "label"]
    modified.loc[current, ["net_pnl", "stress_net_pnl"]] = -99999.
    modified.loc[future, study.path_study.ALIGNED_FEATURES+study.path_study.PATH_FEATURES] = 999.
    changed_val, changed_scores, changed_detail = study.path_study.block(modified, fold, study.PROFILE)
    assert val.index.equals(changed_val.index)
    assert np.allclose(scores, changed_scores, equal_nan=True)
    assert detail==changed_detail
    fit, cal, _ = study.raw_parts(frame, study.longer_fold(fold))
    assert str(fit.trading_day.max())<"2025-09-30"
    assert str(cal.trading_day.max())<"2026-03-31"


def test_refresh_reindexes_scores_to_common_cohort_and_retains_disabled_months(monkeypatch):
    frame = prepared()
    evaluation = study.evaluation_rows(frame)
    def block(frame, fold, profile):
        val = study.evaluation_rows(study.raw_parts(frame, study.longer_fold(fold))[2])
        value = np.nan if fold[0]=="2026-02" else .6
        return val.iloc[::-1], np.repeat(value, len(val)), {}
    monkeypatch.setattr(study.path_study, "block", block)
    scores, diagnostics = study.refresh_scores(frame, evaluation, study.PROFILE)
    assert len(scores)==len(evaluation) and len(diagnostics)==10
    february = pd.to_datetime(evaluation.timestamp).dt.month==2
    assert np.isnan(scores[february]).all()
    assert np.allclose(scores[~february], .6)
    for item in diagnostics:
        assert item["raw_fit_last_day"]<item["fold"][1]
        assert item["raw_cal_last_day"]<item["fold"][3]


def test_continuous_ledger_is_not_reset_at_month_boundary_and_unknown_halts():
    frame = prepared().iloc[:3].copy()
    frame["timestamp"] = pd.to_datetime(["2026-01-30T04:00Z", "2026-02-02T04:00Z", "2026-02-03T04:00Z"])
    frame["trading_day"] = frame.timestamp.dt.date
    frame["net_pnl"] = [-198000., 200., 200.]
    first = study.replay(frame, np.repeat(.6, 3))
    assert first.index.tolist()==[frame.index[0]]  # Remaining equity cannot buy next lot.
    frame["net_pnl"] = 200.
    frame.loc[frame.index[1], ["label_status", "label", "net_pnl"]] = ["CENSORED", None, None]
    selected = study.replay(frame, np.repeat(.6, 3))
    assert selected.index.tolist()==frame.index[:2].tolist()


def test_reference_policy_cannot_be_changed_even_with_updated_file_hashes(tmp_path, monkeypatch):
    from dataclasses import asdict
    from pathlib import Path
    monkeypatch.setattr(study, "REFERENCE", tmp_path)
    specification = {"code_hash":study.fingerprint(Path("research/option_path_models.py")), "dependency_code_hashes":{}}
    policy = {"profile":asdict(study.PROFILE), "scope":"CE", "cutoff":.45}
    (tmp_path/"specification.json").write_text(json.dumps(specification))
    (tmp_path/"policy.json").write_text(json.dumps(policy))
    def report():
        (tmp_path/"report.json").write_text(json.dumps({
            "specification_hash":study.fingerprint(tmp_path/"specification.json"),
            "policy_hash":study.fingerprint(tmp_path/"policy.json")}))
    report()
    study.verify_reference()
    policy["cutoff"] = .5
    (tmp_path/"policy.json").write_text(json.dumps(policy))
    report()
    with pytest.raises(ValueError, match="predetermined"):
        study.verify_reference()


def test_data_reuses_verified_causal_cache_and_bounds_new_reads_before_reserved_dates(tmp_path, monkeypatch):
    reference, labels_dir = tmp_path/"reference", tmp_path/"labels"
    reference.mkdir()
    labels_dir.mkdir()
    monkeypatch.setattr(study, "REFERENCE", reference)
    monkeypatch.setattr(study.path_study, "LABEL_SOURCE", labels_dir)
    frame = prepared()
    original = frame.copy()
    dev = frame.loc[frame.trading_day<=pd.Timestamp("2025-12-31").date()]
    label_path = labels_dir/f"{study.path_study.RULE.key}-labels.pkl"
    dev.to_pickle(label_path)
    feature_path = reference/"development-features.pkl"
    dev[["timestamp", "instrument_key"]+study.path_study.PATH_FEATURES].to_pickle(feature_path)
    (reference/"feature-audit.json").write_text(json.dumps({"snapshot_hash":study.fingerprint(feature_path)}))
    monkeypatch.setattr(study.path_study, "load_source", lambda *args:(frame.copy(), {"dataset_hash":"fixed"}))
    def load_paths(later):
        assert str(later.trading_day.min())>="2026-01-01"
        assert str(later.trading_day.max())<="2026-10-13"
        return {}
    monkeypatch.setattr(study.path_study, "load_paths", load_paths)
    monkeypatch.setattr(study.path_study, "relabel", lambda f, *args:f.copy())
    prepared_frame, _ = study.load_data({"label_cache_hash":study.fingerprint(label_path)})
    assert str(prepared_frame.trading_day.max())=="2026-10-13"
    pd.testing.assert_frame_equal(frame, original)
    (reference/"feature-audit.json").write_text(json.dumps({"snapshot_hash":"tampered"}))
    with pytest.raises(ValueError, match="snapshot changed"):
        study.load_data({"label_cache_hash":study.fingerprint(label_path)})


@pytest.mark.parametrize("profitable", [False, True])
def test_run_preregisters_before_reading_history_and_never_promotes(tmp_path, monkeypatch, profitable):
    reference, source, output = tmp_path/"reference", tmp_path/"source", tmp_path/"run"
    reference.mkdir()
    source.mkdir()
    for name in ("specification.json", "policy.json", "report.json", "candidate.joblib", "feature-audit.json", "development-features.pkl"):
        (reference/name).write_text("{}")
    (source/"report.json").write_text("{}")
    frame = prepared()
    if not profitable:
        frame["net_pnl"], frame["stress_net_pnl"], frame["label"] = -100., -110., 0
    evaluation = study.evaluation_rows(frame)
    frozen = study.summarize(study.replay(evaluation, np.repeat(.6, len(evaluation))), study.PROFILE.capital, 5)
    monkeypatch.setattr(study, "REFERENCE", reference)
    monkeypatch.setattr(study.path_study, "SOURCE", source)
    monkeypatch.setattr(study, "verify_reference", lambda:({"audit":{"metrics":frozen}}, {"label_cache_hash":"fixed"}))
    def load_data(spec):
        frozen_spec = json.loads((output/"specification.json").read_text())
        assert frozen_spec["cutoff"]==.45 and len(frozen_spec["arms"])==3
        assert frozen_spec["activation_eligible"] is False
        return frame.copy(), {"dataset_hash":"fixed"}
    monkeypatch.setattr(study, "load_data", load_data)
    monkeypatch.setattr(study, "frozen_scores", lambda val:np.repeat(.6, len(val)))
    monkeypatch.setattr(study, "refresh_scores", lambda frame, val, profile:(np.repeat(.6, len(val)), []))
    monkeypatch.setattr(study, "bootstrap_comparison", lambda *args:{"mean_net_per_trade_95pct":[1., 2.],
        "paired_net_improvement_per_day_95pct":[1., 2.]})
    report = study.run(output)
    assert not report["activation_eligible"] and not report["app_model_selection_changed"]
    assert report["previously_inspected_history"] and report["frozen_comparator_reproduced_exactly"]
    assert (not report["target_rejections"])==profitable
    assert not report["status"].startswith("HISTORICAL_TARGET_MET")
    with pytest.raises(ValueError, match="overwrite"):
        study.run(output)
