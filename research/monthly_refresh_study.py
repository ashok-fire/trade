"""Fixed monthly-refresh mechanism diagnostic on already-exposed history only."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from app.aligned_training import bootstrap_comparison
from app.specialist_research import quality_logit
from research import option_path_models as path_study
from research.decision_day_specialists import raw_parts
from research.pooled_day_models import longer_fold
from research.profit_objective import fingerprint, write_json
from research.walk_forward_tuning import risk_eligible, summarize

REFERENCE = Path("models/research/option-path-models-20261025")
PROFILE = path_study.Profile(200000, 252, True)
CUTOFF = .45
START, END = "2026-01-08", "2026-10-13"
ARMS = ("frozen50", "monthly50", "monthly44")


def monthly_folds(start=START, end=END):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    return [(str(month), "unused", "unused", max(start, month.start_time).date().isoformat(),
             min(end, month.end_time).date().isoformat())
            for month in pd.period_range(start, end, freq="M")]


def verify_reference():
    report = json.loads((REFERENCE/"report.json").read_text())
    spec = json.loads((REFERENCE/"specification.json").read_text())
    for filename, key in (("specification.json", "specification_hash"), ("policy.json", "policy_hash")):
        if fingerprint(REFERENCE/filename)!=report[key]:
            raise ValueError("Frozen reference provenance changed")
    if fingerprint(Path("research/option_path_models.py"))!=spec["code_hash"]:
        raise ValueError("Frozen option-path implementation changed")
    for filename, expected in spec["dependency_code_hashes"].items():
        if fingerprint(Path(filename))!=expected:
            raise ValueError("Frozen reference dependency changed")
    policy = json.loads((REFERENCE/"policy.json").read_text())
    if policy["profile"]!=asdict(PROFILE) or policy["scope"]!="CE" or policy["cutoff"]!=CUTOFF:
        raise ValueError("Reference is not the predetermined research policy")
    return report, spec


def load_data(reference_spec):
    source, source_report = path_study.load_source(path_study.SOURCE)
    label_path = path_study.LABEL_SOURCE/f"{path_study.RULE.key}-labels.pkl"
    if fingerprint(label_path)!=reference_spec["label_cache_hash"]:
        raise ValueError("Native-exit development labels changed")
    labels = pd.read_pickle(label_path)
    dev = source.loc[labels.index].copy()
    if dev.trading_day.max()>pd.Timestamp("2025-12-31").date() or not dev[["timestamp", "instrument_key"]].equals(labels[["timestamp", "instrument_key"]]):
        raise ValueError("Development labels are not aligned")
    audit = json.loads((REFERENCE/"feature-audit.json").read_text())
    feature_path = REFERENCE/"development-features.pkl"
    if fingerprint(feature_path)!=audit["snapshot_hash"]:
        raise ValueError("Frozen development feature snapshot changed")
    features = pd.read_pickle(feature_path)
    if not dev.index.equals(features.index) or not dev[["timestamp", "instrument_key"]].equals(features[["timestamp", "instrument_key"]]):
        raise ValueError("Development option features are not aligned")
    dev[path_study.OUTCOMES] = labels[path_study.OUTCOMES]
    dev[path_study.PATH_FEATURES] = features[path_study.PATH_FEATURES]
    # Jan1-7, if present, may become known training evidence in later months;
    # they are not evaluated. Never read the reserved October14-23 window.
    later = source.loc[(source.trading_day>=pd.Timestamp("2026-01-01").date()) &
                       (source.trading_day<=pd.Timestamp(END).date())].copy()
    paths = path_study.load_paths(later)
    later = path_study.relabel(later, paths, path_study.RULE)
    later[path_study.PATH_FEATURES] = path_study.option_features(later, paths)
    return path_study.enrich(pd.concat([dev, later]).sort_values("timestamp", kind="stable")), source_report


def evaluation_rows(frame):
    prepared = path_study.prepare(frame, PROFILE)
    mask = ((prepared.trading_day>=pd.Timestamp(START).date()) &
            (prepared.trading_day<=pd.Timestamp(END).date()) & (prepared.direction=="CE"))
    mask &= risk_eligible(prepared, PROFILE.capital, 3, 5)
    result = prepared.loc[mask].sort_values("timestamp", kind="stable")
    if not result.index.is_unique:
        raise ValueError("Opportunity index must be unique")
    return result


def frozen_scores(evaluation):
    # Trusted local research artifact only, fingerprinted before loading.
    bundle = joblib.load(REFERENCE/"candidate.joblib")
    detail = bundle["metadata"]
    if detail["profile"]!=asdict(PROFILE) or not detail["calibration"]["CE"]["enabled"]:
        raise ValueError("Frozen reference schema or calibration changed")
    with threadpool_limits(limits=1):
        raw = bundle["estimator"].predict_proba(evaluation[detail["model_features"]])[:, 1]
        return bundle["calibrators"]["CE"].predict_proba(quality_logit(raw))[:, 1]


def refresh_scores(frame, evaluation, profile):
    parts, diagnostics = [], []
    for fold in monthly_folds():
        val, scores, detail = path_study.block(frame, fold, profile)
        val, scores = path_study.scoped(val, scores, "CE")
        if not val.empty:
            parts.append(pd.Series(scores, index=val.index, dtype=float))
        fit, cal, _ = raw_parts(frame, longer_fold(fold))
        diagnostics.append({"month":fold[0], "fold":longer_fold(fold),
            "raw_fit_last_day":str(fit.trading_day.max()) if len(fit) else None,
            "raw_cal_last_day":str(cal.trading_day.max()) if len(cal) else None, **detail})
        print(f"Monthly {'50' if profile.option_path else '44'} inputs: {fold[0]} complete", flush=True)
    combined = pd.concat(parts) if parts else pd.Series(dtype=float)
    if not combined.index.is_unique or set(combined.index)!=set(evaluation.index):
        raise ValueError("Monthly refresh changed evaluation opportunities")
    return combined.reindex(evaluation.index).to_numpy(), diagnostics


def replay(frame, scores):
    # One call for ALL months: never reset equity or skip an unknown at refresh.
    return path_study.select(frame, scores, CUTOFF, PROFILE.capital)


def quarter_metrics(selected):
    result = []
    stamp = pd.to_datetime(selected.timestamp, utc=True).dt.tz_convert("Asia/Kolkata")
    quarter = stamp.dt.year.astype(str)+"-Q"+stamp.dt.quarter.astype(str)
    for name in ("2026-Q1", "2026-Q2", "2026-Q3", "2026-Q4"):
        result.append({"quarter":name, **summarize(selected.loc[quarter==name], PROFILE.capital, 5)})
    return result


def run(output):
    output = Path(output)
    if output.exists():
        raise ValueError("Refusing to overwrite frozen monthly-refresh research")
    output.mkdir(parents=True)
    reference_report, reference_spec = verify_reference()
    specification = {"schema":"monthly-refresh-diagnostic-v1", "code_hash":fingerprint(Path(__file__)),
        "reference_hashes":{name:fingerprint(REFERENCE/name) for name in ("specification.json", "policy.json", "report.json", "candidate.joblib", "feature-audit.json", "development-features.pkl")},
        "dependency_code_hashes":{str(p):fingerprint(p) for p in (Path("research/option_path_models.py"), Path("app/aligned_training.py"))},
        "source_report_hash":fingerprint(path_study.SOURCE/"report.json"),
        "label_cache_hash":reference_spec["label_cache_hash"],
        "hypothesis":"Scheduled causal monthly model/calibration refresh may reduce the observed temporal degradation of a year-frozen estimator",
        "arms":ARMS, "primary_contrast":"monthly50 minus frozen50", "secondary_contrast":"monthly50 minus monthly44",
        "profile":asdict(PROFILE), "scope":"CE", "cutoff":CUTOFF, "exits":vars(path_study.RULE),
        "folds":[longer_fold(f) for f in monthly_folds()],
        "refresh":"before first observed evaluated session of each calendar month; preceding6 complete months calibration; expanding earlier fit; purge final observed day in both; same sample/positive-slope gates and effective-day weights as frozen study",
        "model":"unchanged fixed HGB100iterations/7leaves/lr.05/minleaf20/L2=10/seed42; full inputs50 or paired44; half-life252 days; no learner/cutoff/exit/capital/window optimisation",
        "ledger":"continuous200000, one historical lot, 3%/5%, cash plus reserve fees, first passing/day, no monthly reset; first selected unknown outcome halts entire ledger",
        "economic_gates":"100completed selected trades/coverage99%/PF1.3/positive base and double-slip net/DD<=20000/no daily overshoot/>=60% positive active months/>=3of4 stress-positive quarters with10trades/positive lower95 mean expectancy and paired improvement vs both controls",
        "win_gate":"strictly>70% realised net wins; economic partial track separately>=60%; no final success even if passed because history is already exposed",
        "uncertainty":"10000 five-observed-day circular day-block resamples, seed42; descriptive only",
        "reserved_dates_excluded":["2026-10-14", "2026-10-23"], "previously_inspected_history":True,
        "activation_eligible":False, "runtime_compatible":False, "app_or_registry_changes":False,
        "stop_rule":"one fixed three-arm diagnostic; if unsupported, wait for meaningful fresh quote-based Shadow outcomes; no grid expansion"}
    write_json(output/"specification.json", specification)
    spec_hash = fingerprint(output/"specification.json")
    frame, source_report = load_data(reference_spec)
    evaluation = evaluation_rows(frame)
    frame.to_pickle(output/"prepared-history.pkl")
    write_json(output/"dataset.json", {"source_hash":source_report["dataset_hash"], "prepared_hash":fingerprint(output/"prepared-history.pkl"),
        "history_rows":len(frame), "history_days":frame.trading_day.nunique(), "evaluation_rows":len(evaluation),
        "evaluation_days":evaluation.trading_day.nunique(), "first_day":str(evaluation.trading_day.min()), "last_day":str(evaluation.trading_day.max())})
    scores = {"frozen50":frozen_scores(evaluation)}
    selections = {"frozen50":replay(evaluation, scores["frozen50"])}
    frozen = summarize(selections["frozen50"], PROFILE.capital, 5)
    if frozen!=reference_report["audit"]["metrics"]:
        raise ValueError("Frozen comparator did not reproduce previous audit exactly")
    diagnostics = {}
    for arm, profile in (("monthly50", PROFILE), ("monthly44", path_study.Profile(200000, 252, False))):
        scores[arm], diagnostics[arm] = refresh_scores(frame, evaluation, profile)
        selections[arm] = replay(evaluation, scores[arm])
        write_json(output/f"{arm}-diagnostics.json", diagnostics[arm])
    metrics = {arm:summarize(selections[arm], PROFILE.capital, 5) for arm in ARMS}
    quarters = {arm:quarter_metrics(selections[arm]) for arm in ARMS}
    comparisons = {arm:bootstrap_comparison(evaluation, selections["monthly50"], selections[arm]) for arm in ("frozen50", "monthly44")}
    m = metrics["monthly50"]
    rejections = path_study.rejection_reasons(m, PROFILE.capital)
    if sum(q["trades"]>=10 and q["stress_net_pnl"]>0 for q in quarters["monthly50"])<3:
        rejections.append("three_stable_quarters")
    for arm, uncertainty in comparisons.items():
        if uncertainty.get("paired_net_improvement_per_day_95pct", [-1])[0]<=0:
            rejections.append(f"positive_lower95_improvement_vs_{arm}")
    if comparisons["frozen50"].get("mean_net_per_trade_95pct", [-1])[0]<=0:
        rejections.append("positive_lower95_mean_expectancy")
    target_rejections = list(rejections)
    if (m["win_rate_pct"] or 0)<=70:
        target_rejections.append("strictly_above_70pct_net_wins")
    status = ("HISTORICAL_DIAGNOSTIC_TARGET_ONLY_FRESH_EVIDENCE_REQUIRED" if not target_rejections else
              "HISTORICAL_ECONOMIC_IMPROVEMENT_NOT_70_PERCENT" if not rejections else "REJECTED_MONTHLY_REFRESH_DIAGNOSTIC")
    for arm, selected in selections.items():
        selected.to_json(output/f"{arm}-trades.json", orient="records", date_format="iso", indent=2)
    prediction_frame = evaluation[["timestamp", "instrument_key", "trading_day", "label_status"]].copy()
    for arm in ARMS:
        prediction_frame[arm] = scores[arm]
    prediction_frame.to_pickle(output/"predictions.pkl")
    if fingerprint(output/"specification.json")!=spec_hash:
        raise ValueError("Frozen monthly-refresh specification changed")
    for name, expected in specification["reference_hashes"].items():
        if fingerprint(REFERENCE/name)!=expected:
            raise ValueError("Frozen reference changed during study")
    report = {"status":status, "specification_hash":spec_hash, "source_hash":source_report["dataset_hash"],
        "predictions_hash":fingerprint(output/"predictions.pkl"), "metrics":metrics, "quarters":quarters,
        "comparisons":comparisons, "economic_rejections":rejections, "target_rejections":target_rejections,
        "frozen_comparator_reproduced_exactly":True, "activation_eligible":False, "app_model_selection_changed":False,
        "previously_inspected_history":True, "next_step":"Wait for meaningful fresh quote-evidenced completed Shadow observations; no deterministic retry or grid expansion"}
    write_json(output/"report.json", report)
    print({"status":status, "metrics":metrics, "rejections":target_rejections}, flush=True)
    return report


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
