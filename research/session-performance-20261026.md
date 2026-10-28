# Model performance review — 26 October 2026 (IST)

## Outcome

Positive first forward-observation session, but insufficient evidence for
retuning or the requested validated >70% net-win goal. No automatic model
learning occurred. This review made no model, threshold, runtime, risk, order or
database changes. The app was offline at review time; stored records were
queried with SQLite `PRAGMA query_only=ON`.

The session capture began at12:17:36 IST after token recovery and continued
through the15:05 observation boundary; last exit15:20:09. This is a partial-day
sample, not a full-day test including the opening session.

## Selected Shadow portfolio model9

Frozen version `quality-v5-20261022T153945Z`; one NIFTY22700CE simulated trade,
65units (one lot), entry14:20:15IST at123.60, time exit15:20:00IST at134.05.
Entry score0.536062 exceeded the frozen CE cutoff0.52. Gross679.25, estimated
charges67.65, **net +611.60**. Recomputed fees and net matched stored values.
This is simulation, not broker-realised profit. No live orders were placed.

Of26 recorded decisions:13 rejected by model quality,8 by option selection,
4 had no ORB setup,1 entered. The8 selection rejects comprised6 delta-range
failures and2 stale-ATM/delta-ITM1 failures. They are not all model rejections;
changing the classifier cannot make an invalid quote executable.

The selected model's one winning trade is not a meaningful100% estimated win
rate. Its historical29-trade validation still does not establish live readiness.

## Parallel research reference11 — not the selected portfolio

Frozen version `quality-v6-20261023T101725Z`;140 checks (35boundaries ×4families):
105 NO_SETUP,13 QUOTE_REJECTED,22 CLOSED,0 OPEN and0 INCOMPLETE. There is one
observed trading date. No PE trial opened; no confirmed failed-breakout or
VWAP-reclaim candidate appeared in this captured period.

| Descriptive group | Closed trials | Net-positive / negative | Net-win fraction |
|---|---:|---:|---:|
| ORB CE, all overlapping trials |20|16 /4|80.00%|
| ORB CE, frozen score pass |8|7 /1|87.50%|
| ORB CE, frozen score reject |12|9 /3|75.00%|
| Pullback CE, all overlapping trials |2|2 /0|100.00%|
| All research rows, including duplicates |22|18 /4|81.82%|

These are descriptive row fractions, NOT independent model accuracy, a
portfolio backtest or proof of improvement. Two pullback rows duplicate the
same contract/time/fills as ORB rows, leaving20 unique entry events, many
overlapping on the same bullish move. Summing their hypothetical net P&L
(31590.18) would misrepresent capital use; it is not the account's return.

The predetermined first-executable-per-family/day view yields:

- ORB CE:1 closed selected decision, net +1964.51 after73.24 charges.
- Pullback CE:1 closed selected decision, net +2578.80 after73.20 charges.

Both first decisions also passed the frozen score gate. Thus the day's primary
first-daily comparisons show no selected-decision difference between scored
and unscored policies. Do not add the two families as an independent portfolio
or choose the better family after seeing the outcome.

## Label, quote and accounting audit

All22 closed rows passed9 checks: closed signal before entry, entry/exit option
quote freshness, underlying exit quote freshness as recorded, frozen rounded
entry/exit fill formulas, gross P&L, date-versioned charges, and net P&L.
No failed check was found. Quote timestamp freshness is based on recorded
provider/adapter timestamps, not proof of exchange latency or fillability.

There were20 UNDERLYING_TARGET exits but only18 net-positive rows. Two target
exits still lost100.55 and94.05 because the option exit price was below its
entry price and fees added to the loss. Win labels must use **net option P&L**,
not whether the underlying touched its target. Two other rows stopped out.

Repricing the SAME captured entry/exit quotes at double configured slippage
left18/22 net positive. The first ORB decision fell from1964.51 to1954.76.
This is only a fill-cost sensitivity, NOT a replay of changed stops/paths or
proof of the full double-slippage robustness gate.15-second polling may miss
intrabar touches; historical/live feature and quote limitations remain.

The accounting audit input summary hash (ordered closed-row IDs, timestamps,
fills, net outcomes and same-exit sensitivity) was:
`eace38dc29abe4ac0bff21c6bd588179bf22d27b3bf01b7c329c57808733719e`.

## Tuning decision and next work

**No retuning on this one-session sample.** Runtime loops collect data, score
and monitor exits; they do not call model fitting. Latest model artifacts remain
dated22/23October; model9/reference11 and saved capital100000/risk3%/daily5%/RR1.5
are unchanged. Earlier monthly-retraining research failed, not a deployed job.

Keep today's outcomes as forward evaluation of these frozen models. Do not
train on them and then present their rescore as an independent test. No lower
cutoff, larger stake, different family or new indicator is justified by this
session. Require100+ completed selected decisions with adequate distinct-day
support, positive net/stress evidence and stable periods before claiming the
requested target; do not count22 overlapping trials as22 independent days.

Useful follow-up: preserve unchanged collection through future sessions;
review first-daily quality-versus-unscored results separately by frozen protocol,
quote rejection reasons, net-not-target labels and score calibration. A future
candidate needs a preregistered experiment and genuinely later frozen validation.

One existing diagnostic inconsistency was found: the entered model9 signal's
`checks.confidence` field is false because runtime writes a generic>=0.70 flag,
although the actual model-specific gate correctly accepted0.536062>=0.52.
This is a misleading audit flag, not evidence that the model bypassed its0.52
gate. It was not changed as part of this analysis; do not interpret it as a
request to increase/decrease the trading threshold.
