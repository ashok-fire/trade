# NIFTY research ledger and continuation

## Decision-parity and risk foundation — 4 November 2026

The user authorised the architecture correction identified by the full-system
audit. No strategy/model was selected, no threshold or saved trading setting
was changed, and no live order was enabled. Runtime and the parallel observer
now persist immutable `decision-snapshot-v2` evidence containing verified-only
spot/futures source hashes, the exact computed feature row, model contract,
settings, option-selection evidence and final decision. The read-only
`/api/decision-parity` report rebuilds captured market snapshots and requires
at least 100 reconstructable snapshots across five sessions with 95% exact
source/feature parity before it can mark the foundation ready for model
selection. Selected and parallel records from the same provider/index/boundary
count as one independent parity unit. Pre-final v1 snapshots remain immutable
but are excluded from v2 acceptance. Snapshot collection begins only after the application process
is restarted with this code; historical records are not fabricated.
Post-restart validation captured three v2 records over two independent NIFTY
boundaries; both reconstructed exactly (100%), but support remains false at
2/100 boundaries and one of five required sessions.

`market_frame_for_boundary` now rejects unverified spot/futures rows, and
backtest underlying exit paths do the same. Position sizing now enforces both
planned premium-stop risk and whole-premium cash affordability, records maximum
premium exposure, and rejects a new trade when its planned stop would exceed
the remaining daily-loss budget. The saved 100000/5%/7%/20-trade settings and
SHADOW artifact9/reference11 remain unchanged. Health now distinguishes the
last completed candle from the last position-monitor quote. A stale fixed-date
Groww test and the Python3.14 ASGI test-harness deadlock were corrected without
weakening application assertions. Full regression plus parity-report test: 241 passed. These changes
establish evidence and risk invariants; they do not establish profitability.

## Fixed multi-strategy batch — 4 November 2026

See [full multi-strategy diagnostic](multi-strategy-20261104/README.md). The
user-requested fixed first-of-five setup policy was tested on the same verified
NIFTY history and one-lot option-cost proxy as a rule-only ORB control. In the
already exposed 2026 period, the union had 179 completed first-daily trades,
39.11% net wins, −₹18,783.24 after estimated costs and −₹20,530.36 at double
slippage; the ORB rule control had 174 completed, 40.23%, −₹13,088.77. No gate
for profitability, coverage, drawdown, stability or supported improvement
passed. The spot-VWAP reversion arm was structurally unreachable because the
stored spot index volume is zero, so it was retained as a rejected data-definition
attempt, not presented as a negative mean-reversion test. No new strategy or
model was selected. Historical quotes remain unavailable for final validation.

## Latest alignment diagnostic — 4 November 2026

See [portfolio replay alignment](portfolio-alignment-20261104/README.md).
New backtests snapshot saved portfolio limits, and eight exact 2–3 November
option contract-days are complete. A fixed replay of artifact 9 **failed
signal/contract parity** with observed SHADOW trades: only one matching entry,
despite seven proxy and eight observed trades. The proxy's positive net is not
model improvement or validation. Current runtime settings were 100000/5%/7%/
20 trades/five consecutive losses at this diagnostic. Further model selection
requires feature/selection/quote parity and independent fresh evidence.

**Indicator warm-up follow-up:** the backtest now reads up to 1,500 prior
verified spot candles with corresponding futures context and excludes those
dates from entries. On the same two inspected days, the corrected proxy yielded
9 trades/−₹1,612.90 and matched 2 of 8 observed signal/contract pairs. The
earlier +₹3,112.66 proxy result was an indicator-history artifact, not model
improvement. Six entries remain unmatched; see the warm-up specification,
report, comparison and limits in the linked ledger. No model was selected.


## Current conclusion — 25 October 2026

No model has established the requested >70% realised net win rate with the
required sample support, economic/risk gates and fresh forward evidence.
Higher development point estimates exist, but must not be promoted as success.
The user authorises continued model/strategy research and offline amount
experiments, not live orders or weakening acceptance after seeing results.

| Study | Main finding | Report |
|---|---|---|
| First-decision specialists | 36 profiles tested; sparse direction/clock/volatility calibration blocks; no100-trade selected policy | `models/research/decision-day-specialists-20261025/report.json` |
| Pooled daily models | Six-month calibration improved support; best100+trade diagnostic44.44% on144trades, PF1.102 | `models/research/pooled-day-models-20261025/report.json` |
| Premium-outcome meta-models | Best100+trade development diagnostic66.99% on103trades, PF1.291; below gates | `models/research/premium-meta-models-20261025/report.json` |
| Frozen provisional CE | Development70.59%/68trades/PF1.831 did NOT generalise: later2026 audit32.26%/31trades/PF0.381/net -15047.89 | `models/research/frozen-ce-meta-audit-20261025/report.json` |
| Effective-day models | Risk-aligned samples, weights summing to days, causal cost inputs and recency; best100+trade diagnostic60.73%/191trades/PF1.189; no qualifying policy | `models/research/effective-day-models-20261025/report.json` |
| Option-price-path models | Eight paired model profiles; selected economic CE development66.37%/113trades/PF1.829 failed later2026 audit42.59%/54trades/PF0.774/net -8397.80 | `models/research/option-path-models-20261025/ASSESSMENT.md` |
| Fixed monthly refresh | Three fixed arms, no parameter search; monthly50 and monthly44 selected identical26trades/38.46%/PF0.631/net -8507.53; rejected | `models/research/monthly-refresh-20261025/README.md` |

All these development figures are selection-biased, not independent final
performance estimates. The 2026 audit has repeatedly been inspected and is not
a fresh test. October14–23 remains reserved and unused. One-trade/day completed
counts are selected decisions, not the count of overlapping training candles.

## Amount diagnostics already completed

The frozen histogram model, CE-only, cutoff0.55, premium stop25%/target15%/hold60
minutes, one historical lot, same six development-quarter ledgers:

| Capital | Selected trades | Net wins | Net P&L | PF |
|---:|---:|---:|---:|---:|
|100000|68|70.59%|14999.95|1.831|
|200000|73|71.23%|18722.03|1.875|
|500000|73|71.23%|18722.03|1.875|
|1000000|73|71.23%|18722.03|1.875|

These are underpowered development diagnostics. Capital changes eligibility;
it did not create a new predictive model. The primary audit remained frozen at
100000 rather than choosing audit capital after seeing these figures. Do not
present71.23%/73trades as the validated target or automatically increase runtime
capital/risk. Cash affordability, historical lot size and fees remain mandatory.

## Important modelling corrections and limits

- Earlier day weights gave equal aggregate influence to each day, but their
  total was rescaled to candle rows. In regularised estimators/calibration that
  does not make regularisation equivalent to an effective day sample size.
  The effective-day study instead uses weights summing to observed days and
  optional126/252-observed-day decay. This is tested in research only.
- Risk eligibility should be causal and aligned across fitting, calibration and
  evaluation. A bigger training table is not automatically a bigger independent
  sample. Preserve unknown first outcomes; never substitute later known wins.
- Causal cost inputs tested: signal premium, historical lot size, estimated
  round-trip charges at unchanged signal price as percent notional, and gross
  premium-stop risk relative to budget. No future fill/exit prices are inputs.
- Native option stop/target labels are candle-based proxies. Research bundles
  using those exits or new features are NOT compatible with current runtime.
  Do not select them into Shadow or Live merely because their files exist.
- Active configured mode is SHADOW, selected model9, parallel reference11;
  saved capital100000/risk3%/daily5%/underlying RR1.5/premium emergency stop25%
  are unchanged. No live order or registry write was made by these studies.

## Completed option-price-path hypothesis

The causal option-premium momentum/volatility study is complete and rejected.
Do not repeat the grid below as a new study. It compared the44-input cost control
with50 inputs, not a new UI/product feature or permission to deploy. Source:
`research/option_path_models.py`;11 focused tests and169 total tests passed.

The frozen protocol was:

1. Use verified stored same-contract option OHLC at or before the closed signal
   candle only. Candidate inputs: return1/3bars, realised log-return volatility
   over3bars, signal range/close, candle body/open, and close-location in range.
   Missing lag history remains missing; never fill from entry or future bars.
   Tests must show future-candle changes cannot change signal-time features.
2. Keep frozen causal ORB opportunities and premium stop25%/target15%/max60min
   labels, conservative gap/stop ordering, double-slippage path replay and fees.
3. Compare44-input control versus50-input option-path model with fixed shallow
   histogram settings and effective-day weights. Keep whole-day chronology,
   prior6month side-specific calibration and positive slopes. No hyperparameter
   search. At most full-history versus252-day decay and100000 versus200000
   research-capital profiles; declare the complete finite grid beforehand.
4. Use matched cash/whole-lot eligibility, max one selected trade/day, unchanged
   percentage risk gates, declared10% capital drawdown ceiling. Require100+
   completed selected trades, >70% net wins for the target, PF1.3, positive net
   and stressed P&L, coverage99%, stable windows/months and supported neighbours.
   If a partial economic track is included, label it separately and never count
   it as meeting70%. Log every rejection and freeze at most one audit candidate.
5. Any later2026 test is still previously inspected history. A final success or
   live-readiness claim needs genuinely fresh frozen quote-based forward data.
   Do not consume October14–23 opportunistically to rescue a failed candidate.

The frozen CE/200000/252-day/cutoff0.45 policy produced54 later2026 trades,
42.59% wins, PF0.774, net -8397.80, stress -10181.67 and only2/10 positive months.
Matched44-input control:56trades,44.64%, PF0.719, net -12341.98. The model lost
less money, but paired95% improvement included zero (-62.17 to146.35 per day).
Missing lag inputs were below0.04%; the final estimator used all50 inputs. The
highest100+-trade development diagnostic was66.45%/152trades, not70% success.

## Completed diagnostic — scheduled refresh, not more inputs

The final model/calibration were frozen from2025 across the entire2026 audit.
That is an observed limitation, not proof that staleness caused the losses.
The distinct causal monthly-refresh study has now completed without adding
indicators or searching another threshold grid. Do not run this protocol again:

1. Freeze the current research profile:200000,3%/5%, CE-only,50 inputs,
   252-observed-day weight decay, fixed HGB and premium25/15/60 exits, cutoff0.45.
   Refresh on the first observed session of each calendar month. Fit only dates
   before the prior six complete calendar months; calibrate on those six months;
   purge the final observed day in fit and calibration, require existing sample
   and positive-slope gates. No changes chosen from that month's outcomes.
2. Compare exactly three arms: frozen50-input policy from this batch, monthly
   refreshed50-input policy, and identically refreshed44-input control. No
   alternative windows, learner, exits, scope, capital or cutoff optimisation.
   Monthly predictions feed one continuous one-lot capital ledger, not a fresh
   account every month; preserve the first selected unknown outcome and halt.
3. Preregister and hash the full protocol before executing. Evaluate only the
   already-exposed Jan8–Oct13,2026 dates; never consume reserved October14–23.
   Include day-block paired uncertainty, costs/stress,100-trade support,
   profitability/drawdown/monthly gates and all rejections. This is a mechanism
   diagnostic on exposed history, NOT fresh proof or authority to deploy even
   if its displayed win rate exceeds70%.
4. If no supported improvement, do not expand the grid. Wait for genuinely new
   completed quote-based Shadow evidence rather than repeating deterministic
   failed searches. No runtime retraining/activation, risk changes or orders.

Result: monthly50 and monthly44 both26trades/38.46% wins/PF0.631/net -8507.53,
stress -8698.41,1/5 positive active months. Frozen50 reproduced exactly at
54trades/42.59%/PF0.774/net -8397.80. Paired95% improvement vs frozen50 crossed
zero (-138.02 to126.98 per observed day), vs monthly44 was exactly[0,0]. The
monthly fits did run for all ten months; all CE calibrators had positive slopes
and adequate support. June onward scores did not reach the fixed0.45 cutoff;
the threshold was not lowered. Full source: `research/monthly_refresh_study.py`.
The evaluation cohort had2767 causal opportunities on109 eligible sessions,
not2767 independent trades. Eight focused and177 total tests passed; all source,
reference, specification and prepared/prediction snapshot hashes verified.

## Next follow-up — fresh evidence only, no offline search queued

### Matched failure audit — 28 October 2026

**Human-authorised follow-up completed:** see
[`protocol-alignment-20261028/README.md`](protocol-alignment-20261028/README.md).
525 missing NIFTY option candles were appended for the seven exact observed
contract-days on26–27 October; download11 complete, verified pre-import backup,
no existing candles/observations/settings/models overwritten. All46 signal/entry
pairs and both fixed geometry replay arms are complete. Geometry changed9 exit
bars and2 net-win labels;5 aligned candle labels disagree with original quote
outcomes. These are diagnostics, not a win-rate improvement or fresh test.
New `parallel-shadow-v2` uses signal-close barriers and frozen one-lot risk/cash
eligibility for daily evaluation, retaining raw ineligible rows and legacy v1
outcomes. Missing eligibility/first outcomes cannot be replaced by later winners.
Model9/reference11, SHADOW,100000/3%/5%/RR1.5 and0.05% slippage are unchanged.
44 focused tests passed; wider regression219 passed with the previously stalled
HTTP integration test deselected. After the human authorised starting the app,
final health verified v2/SHADOW/model9/reference11 without runtime errors.
No retraining or new learner/cutoff search ran.
Future checks must keep v1/v2 separate and treat the earlier missing-option
blocker below as the audit's original state, now resolved by this follow-up.

The human-requested matched audit is complete:
[`matched-failure-20261028/README.md`](matched-failure-20261028/README.md).
116 quote selections,46 reference scores,33 scoreable selected-model scores and
all46 closed-trial accounting/exit checks reproduced. No new model/threshold
change is justified. The reference observer anchors barriers to a current quote,
unlike selected Shadow/training's signal close (median gap0.082ATR, max0.537ATR).
Only22/46 research entries satisfy fixed100000/3% one-lot risk/cash rules; all24
27 October observations are ineligible. Eligible first-daily reference-filter
and unfiltered comparisons are identical on the sole eligible date26 October.
Current-candle reconstruction differs from140 recorded26 October feature rows
in four opening-input fields, without changing setup directions; revisions are
a possible explanation, not an established root cause. Exact same-session option
replay has0/46 signal/entry pairs and remains blocked on missing26–27 October
option history. Future work is a separately approved data import and versioned
protocol alignment, NOT another learner/cutoff grid. Keep legacy observations,
model9/reference11, runtime/risk settings and reserved option outcomes unchanged.
Run-01 export failure was retained; run-02 fixes JSON scalar serialization only.

No objectively supported distinct offline experiment remains queued. Do not
invent another indicator, learner, capital, window or cutoff variant to reuse
the same exposed audit. Keep the hourly follow-up useful as a read-only health
and fresh-evidence check; unchanged healthy/weekend/no-new-outcome states should
not notify the user or start a deterministic rerun.

At the25 October heartbeat, health was OK/SHADOW/model9/reference11, no runtime
error, saved capital100000/risk3%/daily5%/RR1.5. The parallel observations table
was empty, so there were no new completed quote-evidenced outcomes. This is
expected outside weekday entry hours, not a reason to call order endpoints.
Baseline count: zero CLOSED observations. On future checks, inspect new CLOSED
days, quote/exit evidence and protocol compatibility before any research action.
Do not train on a tiny new sample or treat reference11 observations as direct
validation of rejected50-input/native-exit research models. Preserve their
failures. Any new justified hypothesis must be documented/frozen before testing;
70% success still requires sufficient truly fresh selected-trade evidence.

## Fresh session update — 26 October, 12:20 IST

Groww token recovery verified; current candles and parallel collection resumed.
Read-only snapshot: eight family checks, seven NO_SETUP, one OPEN ORB-CE
counterfactual, zero CLOSED outcomes, one distinct observed session. Selected
model9 rejected its first CE candidate (0.3800<0.52); reference11 scored0.4809
against its0.48 ORB-CE gate and opened the research trial. No threshold/model
changes, retraining or real orders. Details: `research/forward-session-20261026.md`.
Fresh completed-outcome baseline remains zero; do not count pending trials as
success or confuse reference11's protocol with rejected offline candidates.

## Completed session review — 26 October

See `research/session-performance-20261026.md` for the stored-data analysis.
Selected model9:1 closed Shadow trade, net +611.60. Reference11:22 closed
overlapping observations,18 net-positive,20 unique entry events but ONE
observed trading day, captured only after12:17IST token recovery. Primary
first-daily ORB and pullback each had1 winning selected decision; both also
passed their frozen gates, so no incremental first-daily selection benefit is
established. All22 quote/fill/accounting audits passed. Two underlying-target
exits still lost money: use net option P&L labels, never target-touch counts.

No automatic retraining occurred and no tuning/model/risk changes were made.
Today's81.82% overlapping-row fraction does NOT establish the requested70%
goal. Retain it as forward evidence; do not train/rescore it as an independent
test or select a new family/cutoff from this one bullish session. The app was
offline at review time, with no open positions or unfinished research trials.

Execute at most one bounded new batch per scheduled follow-up and never repeat
a deterministic failed search. If no justified distinct hypothesis remains,
wait for meaningful new labelled evidence rather than inventing more cutoffs.
The hourly follow-up is an ongoing research mechanism, not a promise that the
70% target is achievable. Keep notifications for meaningful results/failures or
required input only. Preserve source data, existing worktree edits and secrets.
