# Decision-aligned NIFTY ORB experiment — 24 October 2026

## Decision

**Do not promote this candidate.** Alignment is implemented and the fixed
comparison is complete, but the acceptance gates did not pass. No model was
registered, selected or activated by this run. Existing Shadow model #9 and
parallel research reference #11 are unchanged.

Candidate: `quality-aligned-v1-20261024T173926Z`.
Feature schema remains bundle-compatible V6 with an explicit new training
contract, `decision-aligned-v1`; its actual 40 input features are recorded in
`candidate.metadata.json`. Existing V4/V5/V6 artifacts are not relabelled or
overwritten.

## What was corrected

- Every eligible five-minute decision is an opportunity, rather than just the
  first setup each day. Runtime and training share entry-window semantics.
- Setup priority is first-confirmed, matching `selected_setup` in runtime.
  This particular experiment is frozen to ORB CE/PE, with no family selection.
- ATM first, ITM1 fallback uses signal-time evidence, never future exit results.
- Option-volume and OI features are excluded; their historical/live field
  meanings remain unreconciled. Ordinary executable-quote filters are preserved.
- Morning opportunities are not removed because a later part of their session
  has missing candles. Failed exit paths stay unlabelled, not substituted.
- Fitting, calibration, threshold selection and evaluation use separate day
  groups. Classifiers use equal total weight per trading day, not class-balanced
  weights. Repeated observations are not treated as independent trading days.
- Holdout scoring uses the saved bundle's feature list and runtime calibration
  formula, including legacy rank terms where applicable. New runs use separate
  Platt calibration with rank coefficient zero; non-positive calibration slopes
  disable a direction rather than silently flipping it.

## Frozen experiment

One existing shallow CatBoost configuration per direction: 250 iterations,
depth 3, learning rate 0.03, L2 10, seed 42, one thread. Platt calibration uses
logit-quality input and regularised logistic regression C=1, seed 42. The
existing bundle-format ranker is fit on fitting days only, with coefficient
explicitly zero; no estimator or market-regime selection was performed.

| Partition | Actual dates | Opportunities | Complete labels | Days |
|---|---|---:|---:|---:|
| Fit | 3 Jan 2023–3 Apr 2025 | 18,802 | 18,784 | 533 |
| Calibration | 8 Apr–18 Aug 2025 | 2,717 | 2,717 | 83 |
| Threshold tuning | 20 Aug 2025–6 Jan 2026 | 3,279 | 3,273 | 93 |
| Historical evaluation | 8 Jan–13 Oct 2026 | 6,004 | 5,993 | 181 |

Boundary-purge opportunities are excluded from those partitions. Overall:
30,932 opportunities, 30,897 complete labels, 35 censored outcomes. This is
not 30,897 independent trades. No reserved 14–23 October market data was used.

The specification was saved before fitting. Thresholds were chosen only on
tuning data, from the predefined 0.40–0.76 grid, with support, neighbouring
cutoff stability and cost-stress profitability in both tuning halves. CE 0.48
passed tuning. PE had no passing cutoff and remains disabled (sentinel 1.01).
The 48% model cutoff is not a claimed 48% or 56% realised win rate.

| Period / policy | Complete trades | Wins | PF | Net P&L | Double-slippage P&L |
|---|---:|---:|---:|---:|---:|
| Candidate, tuning | 23 | 60.87% | 2.037 | ₹7,593.40 | ₹7,395.27 |
| Candidate, later evaluation | 47 | 48.94% | 1.205 | ₹4,996.22 | ₹4,474.30 |
| Rule-only baseline, same evaluation dates | 181 | 38.67% | 0.776 | −₹28,667.54 | −₹30,689.72 |

Evaluation candidate entries are all CE. Six of ten active months are
profitable. Maximum drawdown is ₹5,047.93 in this one-lot historical replay.
Selected-path coverage is 100% for both policies: no selected censored entry
was skipped or replaced. Eleven unlabelled evaluation opportunities remain in
the decision population even though neither policy selected them first.

Probability diagnostics on 5,993 labelled evaluation opportunities, with equal
total day weight: ROC AUC 0.5585, Brier score 0.2341. This is a diagnostic
point estimate; no statistically established ranking improvement is claimed.
It is not directly comparable as a controlled ablation to the older V5 replay,
because sampling, feature definitions and fitting protocol changed together.

## Failed gates and uncertainty

- 47 completed selected trades is below the predefined minimum of 100.
- Win rate 48.94% is below 55%; PF 1.205 is below 1.30.
- The descriptive 95% interval for mean net P&L per trade is **−₹224.01 to
  +₹462.08**. Positive expectancy is not established.
- The paired candidate-minus-baseline net improvement per observed day has
  a 95% interval of **−₹49.51 to +₹421.92**. Its lower bound is not positive.

Intervals use 10,000 circular moving-block resamples of five observed trading
days, seed 42, retaining no-entry days and excluding any day with a selected
unknown outcome in either policy. No days needed that exclusion in this run.
These intervals are descriptive, not a substitute for an untouched test.

## Remaining limits

The later historical period was already inspected in previous studies. It is
not a fresh validation, regardless of this run's fixed specification. Real
spread, quote age, delta, premium-at-entry and stop-touch option quotes cannot
be reconstructed exactly from five-minute historical candles. The label uses
the established entry-bar/contiguous-path simulator and option bar-close fill
proxies; no exact executable-return claim is made. The comparison takes the
first passing candidate per day at one historical lot, not the full runtime
risk-sizing/multiple-trade portfolio. Forward and portfolio validation remain
mandatory even if later historical point estimates look attractive.

## Operational state and next step

New standard training calls now use the aligned contract. Current stored
Shadow models and raw history remain unchanged. The active service was found
running under the user's VS Code debugger, with no open trades/downloads; it
was deliberately not interrupted. Restart that debug session before using the
updated training route. The offline experiment itself ran the updated code.

Do not relax gates, change cutoffs after viewing evaluation, activate this
candidate, or immediately search another estimator. Collect real decision-time
quote/outcome evidence, reconcile remaining feature/fill proxies, and require
the predefined forward and risk gates before deployment. No higher win rate
has been established by this run.

## Files and reproducibility

- `candidate.spec.json`: pre-fitting specification, SHA-256
  `5f18adf532378d48ed60d2edfb7575ea127bc7831b0ae1203ff68499868baac2`.
- `candidate.metadata.json`: complete metrics, contract, thresholds and blockers.
- `candidate.json`, `candidate.ce.cbm`, `candidate.pe.cbm`,
  `candidate.ranker.json`: frozen bundle; not registered or selected.
- Opportunity-table fingerprint:
  `8961a01b2377b882f670f376a9e0c3fc193a047cb27ebccbc0ae01f816418cd7`.
