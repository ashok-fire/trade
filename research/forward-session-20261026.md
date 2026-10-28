# Forward session analysis — 26 October 2026, 12:20 IST

Read-only diagnostic snapshot after the user updated the Groww token. No model,
threshold, risk setting, execution mode, registry row or order was changed.

## Feed and model decisions

- The 401 error cleared. App health was OK, SHADOW/model9/reference11, and the
  parallel observer reported “Observing current session.” Verified spot/futures
  five-minute candles were arriving. A missing position-monitor last_tick is not
  by itself proof that candle synchronisation has failed.
- At12:17:36 IST, reference11 observed an ORB CE setup from the12:10 signal bar
  (closed12:15). Its diagnostic score0.480876 passed its frozen ORB-CE cutoff0.48.
  This is NOT a validated48% win probability, nor evidence of the70% target.
- Selected Shadow model9 rejected the same bar: score0.380014 below its frozen
  CE cutoff0.52. Its next12:20 decision had no confirmed ORB setup. Do not lower
  thresholds to manufacture entries or switch models after one outcome.

## Executable research evidence

The parallel observer opened one simulated NIFTY22600CE trial, expiring27Oct,
65units, entry142.80 from observed ask142.75 plus configured slippage/tick
rounding. Captured quote age was about2.30seconds (limit3), spread about0.245%
(limit1.5%) and delta0.5695 (allowed0.45–0.60). Entry evidence retained the
option and underlying quotes. No real order was placed.

At06:50:38UTC /12:20:38IST: eight family checks across two decision boundaries,
seven NO_SETUP, one OPEN, zero CLOSED. This is ONE observed session and no
completed outcome—not eight independent trades. The first pending daily trial
must not be replaced with a later winner. No realised win rate can be estimated.

## Next analysis

Let the existing15-second quote monitor record the actual stop/target/time exit,
bid-based fill and fees. Review completed quote evidence, rejected candidates,
score bins and day-grouped net outcomes once there is meaningful support.
Preserve frozen model references and protocol identities; overlapping trials
are not portfolio returns. These reference11 underlying-ATR exits do not directly
validate the rejected50-input/native-premium-exit research models.

The hourly research follow-up remains conditional on meaningful new evidence.
No new historical tuning batch or retraining has been started from this tiny
sample. No target/profitability claim is justified; final evidence requirements
remain unchanged. Live trading remains disabled.

UI: `/shadow-research` shows counterfactual trials; `/orders` shows selected
Shadow model decisions. Seeing a research OPEN trial does not mean a portfolio
trade or broker order was opened.
