"""Bounded causal same-contract option-price-path experiment; research only."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from app.aligned_training import bootstrap_comparison, labelled
from app.specialist_research import quality_logit
from app.training_contract import ALIGNED_FEATURES
from research.decision_day_specialists import CUTOFFS, make_estimator, raw_parts
from research.effective_day_models import EXTRA_FEATURES, LABEL_SOURCE, RULE, SOURCE, effective_weights, enrich
from research.exit_strategy_study import load_paths, relabel, utc
from research.pooled_day_models import longer_fold
from research.precision_tuning import load_source
from research.profit_objective import fingerprint, write_json
from research.walk_forward_tuning import FOLDS, FINAL_FOLD, risk_eligible, select_capital, summarize

PATH_FEATURES = ["option_return_1bar", "option_return_3bars", "option_log_vol_3bars",
                 "option_range_fraction", "option_body_fraction", "option_close_location"]
OUTCOMES = ["label_status", "label", "net_pnl", "stress_net_pnl", "net_return", "exit_reason", "exit_timestamp"]


@dataclass(frozen=True)
class Profile:
    capital: int
    half_life: int | None
    option_path: bool

    @property
    def key(self):
        return f"capital{self.capital}-h{self.half_life or 'all'}-{'path50' if self.option_path else 'control44'}"


PROFILES = tuple(Profile(*values) for values in product((100000, 200000), (None, 252), (False, True)))


def valid_bar(bar):
    if bar is None:
        return False
    values = np.array([bar.open, bar.high, bar.low, bar.close], dtype=float)
    return bool(np.isfinite(values).all() and min(values)>0
                and bar.low<=min(bar.open, bar.close) and bar.high>=max(bar.open, bar.close))


def option_features(frame, paths):
    """Exact same-key/day lookups through signal-bar close, never a future fill."""
    records = []
    for row in frame.itertuples():
        stamp = utc(row.timestamp)
        if stamp.tz_convert("Asia/Kolkata").date()!=row.trading_day:
            raise ValueError("Signal timestamp and trading day disagree")
        path = paths.get((row.instrument_key, row.trading_day), {})
        current = path.get(stamp)
        result = dict.fromkeys(PATH_FEATURES, np.nan)
        if valid_bar(current):
            if abs(float(current.close)-float(row.decision_premium))>.005:
                raise ValueError("Cached signal premium disagrees with option feature candle")
            opening, high, low, close = map(float, (current.open, current.high, current.low, current.close))
            result.update(option_range_fraction=(high-low)/close, option_body_fraction=(close-opening)/opening,
                          option_close_location=(close-low)/(high-low) if high>low else .5)
            previous = [path.get(stamp-pd.Timedelta(minutes=5*i)) for i in range(1, 4)]
            if valid_bar(previous[0]):
                result["option_return_1bar"] = close/float(previous[0].close)-1
            # All three intervening bars required: no bridging gaps or other days.
            if all(valid_bar(bar) for bar in previous):
                closes = np.array([float(bar.close) for bar in previous[::-1]]+[close])
                result["option_return_3bars"] = close/closes[0]-1
                result["option_log_vol_3bars"] = float(np.std(np.diff(np.log(closes)), ddof=0))
        records.append(result)
    return pd.DataFrame(records, index=frame.index, columns=PATH_FEATURES, dtype=float)


def prepare(frame, profile):
    result = frame.copy()
    result["risk_utilization"] = result.decision_premium*1.0005*result.lot_size*.25/(profile.capital*.03)
    return result


def block(frame, fold, profile, output=None):
    fit, cal, val = raw_parts(prepare(frame, profile), longer_fold(fold))
    fit = labelled(fit.loc[risk_eligible(fit, profile.capital, 3, 5)])
    cal = labelled(cal.loc[risk_eligible(cal, profile.capital, 3, 5)])
    val = val.loc[risk_eligible(val, profile.capital, 3, 5)].copy()
    features = ALIGNED_FEATURES+EXTRA_FEATURES+(PATH_FEATURES if profile.option_path else [])
    scores = np.full(len(val), np.nan)
    detail = {"profile":asdict(profile), "features":features, "fit_rows":len(fit),
              "fit_days":fit.trading_day.nunique(), "calibration":{}}
    if len(fit)<500 or detail["fit_days"]<100 or fit.label.nunique()<2:
        detail["disabled"] = "insufficient fitting support"
        return val, scores, detail
    # Fit-only schema handling for an entirely unavailable historical column;
    # never substitute future observations or numeric zero for missing candles.
    model_features = [name for name in features if not fit[name].isna().all()]
    detail["model_features"] = model_features
    detail["all_missing_fit_columns"] = [name for name in features if name not in model_features]
    with threadpool_limits(limits=1):
        model = make_estimator("HIST_GRADIENT")
        weights = effective_weights(fit, profile.half_life)
        detail["fit_weight_sum"] = float(weights.sum())
        model.fit(fit[model_features], fit.label.astype(int), sample_weight=weights)
        calibrators = {}
        for side in ("CE", "PE"):
            sample = cal.loc[cal.direction==side]
            mask = (val.direction==side).to_numpy()
            info = {"days":sample.trading_day.nunique(), "rows":len(sample), "enabled":False}
            if info["days"]>=40 and len(sample)>=100 and sample.label.nunique()==2:
                weights = effective_weights(sample)
                calibration = LogisticRegression(C=1., max_iter=2000, random_state=42).fit(
                    quality_logit(model.predict_proba(sample[model_features])[:, 1]), sample.label.astype(int),
                    sample_weight=weights)
                slope = float(calibration.coef_[0, 0])
                info.update(enabled=slope>0, slope=slope, weight_sum=float(weights.sum()))
                calibrators[side] = calibration
                if slope>0 and mask.any():
                    scores[mask] = calibration.predict_proba(
                        quality_logit(model.predict_proba(val.loc[mask, model_features])[:, 1]))[:, 1]
            detail["calibration"][side] = info
        if output:
            import joblib
            joblib.dump({"schema":"option-path-research-v1", "estimator":model,
                         "calibrators":calibrators, "metadata":detail}, output/"candidate.joblib")
    return val, scores, detail


def select(frame, scores, cutoff, capital):
    return select_capital(frame, scores, {"CE":cutoff, "PE":cutoff}, capital, 3, 5)


def metrics_by_window(frame, scores, cutoff, capital):
    selections, windows = [], []
    for fold in FOLDS:
        mask = (frame.fold==fold[0]).to_numpy()
        trades = select(frame.loc[mask], np.asarray(scores)[mask], cutoff, capital)
        selections.append(trades)
        windows.append({"window":fold[0], **summarize(trades, capital, 5)})
    selected = pd.concat(selections)
    return summarize(selected, capital, 5), windows, selected


def rejection_reasons(metrics, capital, windows=None, win_floor=60, strict=False):
    m = metrics
    tests = {"minimum_100_trades":m["trades"]>=100, "coverage_99pct":m["coverage"]>=.99,
             "win_rate":(m["win_rate_pct"] or 0)>win_floor if strict else (m["win_rate_pct"] or 0)>=win_floor,
             "profit_factor_1_3":(m["profit_factor"] or 0)>=1.3,
             "positive_net":m["net_pnl"]>0, "positive_stress":m["stress_net_pnl"]>0,
             "drawdown_10pct_capital":m["max_drawdown"]<=capital*.1,
             "no_daily_overshoot":m["daily_loss_overshoots"]==0,
             "positive_months_60pct":m["active_months"]>0 and m["positive_months"]>=.6*m["active_months"]}
    if windows is not None:
        tests["four_stable_windows"] = sum(w["trades"]>=10 and w["stress_net_pnl"]>0 for w in windows)>=4
    return [name for name, passed in tests.items() if not passed]


def select_policy(frame, scores, capital):
    curve = []
    for cutoff in CUTOFFS:
        m, windows, _ = metrics_by_window(frame, scores, cutoff, capital)
        target = rejection_reasons(m, capital, windows, 70, True)
        economic = rejection_reasons(m, capital, windows)
        curve.append({"cutoff":cutoff, "metrics":m, "windows":windows,
                      "target_rejections":target, "economic_rejections":economic,
                      "target_pass":not target, "economic_pass":not economic})
    for index, point in enumerate(curve):
        neighbours = [] if index in (0, len(curve)-1) else [curve[index-1]["metrics"], curve[index+1]["metrics"]]
        point["neighbours_supported"] = bool(neighbours) and all(
            m["trades"]>=60 and (m["profit_factor"] or 0)>=1.1 and m["stress_net_pnl"]>0
            and (m["win_rate_pct"] or 0)>=(65 if point["target_pass"] else 55) for m in neighbours)
    accepted = [p for p in curve if p["neighbours_supported"] and (p["target_pass"] or p["economic_pass"])]
    return (max(accepted, key=lambda p:(p["target_pass"], p["metrics"]["stress_net_pnl"])) if accepted else None), curve


def scoped(frame, scores, side):
    mask = np.ones(len(frame), dtype=bool) if side=="BOTH" else (frame.direction==side).to_numpy()
    return frame.loc[mask], np.asarray(scores)[mask]


def run(output):
    output = Path(output)
    if output.exists():
        raise ValueError("Refusing to overwrite frozen option-path research")
    output.mkdir(parents=True)
    specification = {"schema":"option-path-research-v1", "code_hash":fingerprint(Path(__file__)),
        "dependency_code_hashes":{str(p):fingerprint(p) for p in [Path("research/effective_day_models.py"),
            Path("research/exit_strategy_study.py"), Path("research/decision_day_specialists.py"),
            Path("research/walk_forward_tuning.py"), Path("research/pooled_day_models.py")]},
        "source_report_hash":fingerprint(SOURCE/"report.json"),
        "label_report_hash":fingerprint(LABEL_SOURCE/"report.json"),
        "label_cache_hash":fingerprint(LABEL_SOURCE/f"{RULE.key}-labels.pkl"),
        "hypothesis":"Same-contract causal option momentum, volatility and candle shape add information beyond static cost inputs",
        "profiles":[asdict(p) for p in PROFILES], "scopes":["BOTH", "CE"], "cutoffs":CUTOFFS,
        "control_features":ALIGNED_FEATURES+EXTRA_FEATURES, "option_features":PATH_FEATURES,
        "feature_definitions":"fractional close returns over1/3bars; population std of3 contiguous log returns; (high-low)/close; (close-open)/open; (close-low)/(high-low), .5 if flat; require positive finite valid OHLC; exact same key/day; no bars after signal; missing stays NaN",
        "missing_inputs":"native partial-NaN routing; exclude entirely missing columns using fit data alone, record actual model schema; no imputation or future backfill",
        "labels":vars(RULE), "risk":"whole historical lot; 3% per trade/5% daily; cash and reserve fees; 25% premium stop budget; one first passing/day; retain unknown outcomes",
        "weights":"one aggregate unit per observed day, optional252-observed-day half-life; no candle-count rescaling",
        "model":"fixed HGB100iterations/7leaves/lr.05/minleaf20/L2=10/no early stopping/seed42",
        "calibration":"separate CE/PE positive-slope Platt C1; >=40days/100rows;6 preceding calendar months; unit per day",
        "folds":[longer_fold(f) for f in FOLDS], "audit":longer_fold(FINAL_FOLD),
        "acceptance":"100trades/>70% net wins/PF1.3/positive net and stress/coverage.99/DD<=10% capital/no daily overshoot/>=60% positive active months/4of6 stress-positive windows with10trades; adjacent cutoffs >=60trades/PF1.1/positive stress/win>=65",
        "partial_track":"separately labelled >=60% win economic track; same gates except neighbours>=55%; never counts as70% success",
        "freeze":"at most one candidate before later audit, priority target then stressed net; audit also positive lower95 mean and paired unscored improvement; enriched candidate also needs positive lower95 improvement over matched44input control at same frozen cutoff",
        "limits":"quarter-reset development ledgers; one lot not full allocation; all history repeatedly inspected; selection-biased results; candle fill proxies not executable quotes",
        "previously_inspected_history":True, "reserved_window_excluded":["2026-10-14", "2026-10-23"],
        "activation_eligible":False, "app_or_registry_changes":False}
    write_json(output/"specification.json", specification)
    spec_hash = fingerprint(output/"specification.json")
    source, source_report = load_source(SOURCE)
    labels = pd.read_pickle(LABEL_SOURCE/f"{RULE.key}-labels.pkl")
    if not labels.index.is_unique or not labels.index.isin(source.index).all():
        raise ValueError("Label cache index mismatch")
    dev = source.loc[labels.index].copy()
    if dev.trading_day.max()>pd.Timestamp("2025-12-31").date():
        raise ValueError("Development label cache crossed reserved stage boundary")
    if not dev[["timestamp", "instrument_key"]].equals(labels[["timestamp", "instrument_key"]]):
        raise ValueError("Label cache identity mismatch")
    dev[OUTCOMES] = labels[OUTCOMES]
    dev = enrich(dev)
    print("Reading development-only verified option paths", flush=True)
    dev[PATH_FEATURES] = option_features(dev, load_paths(dev))
    dev[["timestamp", "instrument_key"]+PATH_FEATURES].to_pickle(output/"development-features.pkl")
    write_json(output/"feature-audit.json", {"rows":len(dev), "days":dev.trading_day.nunique(),
        "missing_fraction":dev[PATH_FEATURES].isna().mean().to_dict(),
        "snapshot_hash":fingerprint(output/"development-features.pkl")})
    results, best = [], None
    for profile in PROFILES:
        frames, scores, diagnostics = [], [], []
        for fold in FOLDS:
            val, prediction, detail = block(dev, fold, profile)
            val["fold"] = fold[0]
            frames.append(val)
            scores.extend(prediction)
            diagnostics.append({"fold":fold[0], **detail})
        frame, scores = pd.concat(frames), np.asarray(scores)
        for side in ("BOTH", "CE"):
            view, estimates = scoped(frame, scores, side)
            chosen, curve = select_policy(view, estimates, profile.capital)
            record = {"profile":asdict(profile), "scope":side, "selected":chosen,
                      "curve":curve, "diagnostics":diagnostics}
            results.append(record)
            if chosen and (best is None or (chosen["target_pass"], chosen["metrics"]["stress_net_pnl"])>
                           (best["selected"]["target_pass"], best["selected"]["metrics"]["stress_net_pnl"])):
                best = record
        write_json(output/f"{profile.key}.json", results[-2:])
        print(f"Completed {profile.key}", flush=True)
    policy = {"profile":best["profile"] if best else None, "scope":best["scope"] if best else None,
              "cutoff":best["selected"]["cutoff"] if best else None, "specification_hash":spec_hash,
              "dataset_hash":source_report["dataset_hash"], "activation_eligible":False}
    write_json(output/"policy.json", policy)
    policy_hash = fingerprint(output/"policy.json")
    comparisons = []
    for record in results:
        if not record["profile"]["option_path"]:
            continue
        profile = Profile(**record["profile"])
        control = Profile(profile.capital, profile.half_life, False)
        control_record = next(r for r in results if r["profile"]==asdict(control) and r["scope"]==record["scope"])
        for enriched, reference in zip(record["curve"], control_record["curve"]):
            comparisons.append({"profile":record["profile"], "scope":record["scope"], "cutoff":enriched["cutoff"],
                                "path50":enriched["metrics"], "control44":reference["metrics"]})
    write_json(output/"matched-comparisons.json", comparisons)
    status, audit = "REJECTED_NO_SUPPORTED_OPTION_PATH_POLICY", None
    if best:
        profile = Profile(**best["profile"])
        later = source.loc[(source.trading_day>=pd.Timestamp("2026-01-08").date()) &
                           (source.trading_day<=pd.Timestamp("2026-10-13").date())].copy()
        paths = load_paths(later)  # Only after policy has been written and frozen.
        later = enrich(relabel(later, paths, RULE))
        later[PATH_FEATURES] = option_features(later, paths)
        frame, scores, detail = block(pd.concat([dev, later]), FINAL_FOLD, profile, output)
        frame, scores = scoped(frame, scores, best["scope"])
        chosen = select(frame, scores, policy["cutoff"], profile.capital)
        baseline = select(frame, np.ones(len(frame)), 0., profile.capital)
        m = summarize(chosen, profile.capital, 5)
        uncertainty = bootstrap_comparison(frame, chosen, baseline)
        rejections = rejection_reasons(m, profile.capital)
        for key in ("mean_net_per_trade_95pct", "paired_net_improvement_per_day_95pct"):
            if uncertainty.get(key, [-1])[0]<=0:
                rejections.append(key)
        control_audit = None
        if profile.option_path:
            reference = Profile(profile.capital, profile.half_life, False)
            ref_frame, ref_scores, ref_detail = block(pd.concat([dev, later]), FINAL_FOLD, reference)
            ref_frame, ref_scores = scoped(ref_frame, ref_scores, best["scope"])
            if not frame.index.equals(ref_frame.index):
                raise ValueError("Matched audit control uses different eligible rows")
            ref_trades = select(ref_frame, ref_scores, policy["cutoff"], profile.capital)
            paired = bootstrap_comparison(frame, chosen, ref_trades)
            control_audit = {"metrics":summarize(ref_trades, profile.capital, 5), "paired":paired, "calibration":ref_detail}
            if paired.get("paired_net_improvement_per_day_95pct", [-1])[0]<=0:
                rejections.append("no_positive_lower95_improvement_over_matched_control")
        economic = not rejections
        target = economic and (m["win_rate_pct"] or 0)>70
        status = ("HISTORICAL_TARGET_MET_FORWARD_REQUIRED" if target else
                  "HISTORICAL_ECONOMIC_IMPROVEMENT_NOT_70_PERCENT" if economic else "REJECTED_LATER_HISTORICAL_AUDIT")
        audit = {"metrics":m, "calibration":detail, "rejections":rejections,
                 "unscored":summarize(baseline, profile.capital, 5), "uncertainty":uncertainty, "matched_control":control_audit}
        chosen.to_json(output/"audit-trades.json", orient="records", date_format="iso", indent=2)
    if fingerprint(output/"specification.json")!=spec_hash or fingerprint(output/"policy.json")!=policy_hash:
        raise ValueError("Frozen specification or policy changed")
    adequate = [{"profile":r["profile"], "scope":r["scope"], **p} for r in results for p in r["curve"] if p["metrics"]["trades"]>=100]
    diagnostic = max(adequate, key=lambda p:p["metrics"]["win_rate_pct"] or 0) if adequate else None
    path_points = [p for p in adequate if p["profile"]["option_path"]]
    path_diagnostic = max(path_points, key=lambda p:p["metrics"]["win_rate_pct"] or 0) if path_points else None
    report = {"status":status, "specification_hash":spec_hash, "policy_hash":policy_hash,
              "activation_eligible":False, "app_model_selection_changed":False, "model_profiles":len(PROFILES),
              "scoped_profiles":len(results), "cutoffs_per_profile":len(CUTOFFS), "selection":best,
              "highest_win_100plus_diagnostic_only":diagnostic, "path50_highest_win_100plus_diagnostic_only":path_diagnostic,
              "audit":audit, "previously_inspected_history":True}
    write_json(output/"report.json", report)
    print({"status":status, "path50_diagnostic":path_diagnostic["metrics"] if path_diagnostic else None,
           "audit":audit["metrics"] if audit else None}, flush=True)
    return report


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
