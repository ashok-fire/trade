# Completed assessment: useful leads, no qualifying strategy

The preregistered study tested eight option-premium exit rules (15%/25% stop,
15%/25% target, 30/60-minute maximum hold), separately for CE and PE and three
fixed conditions from existing trend-strength inputs: 48 combinations total.
Entry opportunities, contracts and features came from the fingerprint-verified
existing causal ORB dataset. No market history was deleted or downloaded.

Nine combinations had positive estimated net P&L in the six 2024–2025 selection
windows: all eight CE trend-aligned combinations and one PE weak-trend variant.
However, **zero passed every preregistered economic, drawdown and stability gate**.
All 48 had at least 100 completed selected trades. The highest aggregate win
rate was 58.39%, not above the user's 70% goal. Selection performance is biased
by the search and must not be presented as independently validated performance.

## Highest stressed-net diagnostic combination, not an accepted winner

- ORB CE, `setup_direction * trend_atr_strength >= 2`.
- Premium stop 15%, premium target 25%, maximum hold 60 minutes.
- 137 one-lot trades; 46.72% net wins; estimated net P&L ₹13,960.46.
- Re-simulated double-slippage net P&L ₹13,344.59.
- Profit factor 1.266, below the fixed 1.30 gate.
- Maximum drawdown ₹12,486.10, above the fixed ₹10,000 gate.
- Positive active months 9/18, below the fixed 60% requirement.

The same direction/condition with legacy underlying exits returned ₹9,599.95
net over 137 trades, PF 1.188, and drawdown ₹12,040.00. The new exits improve
this development point estimate but do not establish a sufficiently stable edge.
In particular the diagnostic combination lost ₹2,560.80 in 2024 Q3 and
₹8,436.53 in 2024 Q4, despite profitable 2025 windows. These regime-dependent
results are a reason to investigate robustness, not permission to select only
the profitable years after the fact.

## What was and was not done

The replay uses exact next-bar open plus slippage, known opening gaps before
intrabar ordering, otherwise conservative stop-before-target treatment, and
deadline-open exits capped at 15:20 IST. Stress replays the entire path with
double slippage because entry-linked stop/target levels can change. Missing
paths remain unlabelled, never replaced by later known winners. Sizing reserves
the existing conservative 25% signal-premium stop risk for every variant,
including the tighter 15% research stops. Maximum one historical lot/day;
development quarter ledgers reset rather than representing compounding.

Per the specification, no new classifier was fitted because there was no
qualifying foundation strategy. No new 2026 candidate audit was performed or
raw 2026 exit paths read by this study. Reserved October 14–23 data was not used.
Existing source-cache fingerprints and frozen specification/selection hashes
were verified. No gates were relaxed or experiment repeated after results.

Application mode remains SHADOW, selected model #9 and parallel reference #11.
Capital ₹100,000, per-trade risk 3%, daily limit 5%, runtime premium emergency
stop 25%, and underlying reward/risk 1.5 are unchanged. Research exit variants
were not copied into runtime. No model promotion or live order occurred.

This study does not exhaust possible strategies or prove that a 58.39% win rate
is a ceiling. It establishes that these specific exits/conditions are not yet
a dependable foundation for the requested model. Historical candle fills and
estimated fees still require genuinely forward executable-quote validation.
