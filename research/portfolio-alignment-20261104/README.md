# Portfolio replay alignment — 4 November 2026

New backtest submissions now freeze the saved daily trade limit, consecutive-loss
limit, premium stop and reward/risk target. Risk input accepts the saved 5% and
the form displays the saved risk and slippage. Older runs without those fields
keep their original rules. No runtime strategy, model, risk setting or order path
was changed. The app was restarted before the market opened and verified healthy
in SHADOW mode with artifact 9 and parallel reference 11.

The exact eight NIFTY option contract-days previously selected or observed on
2–3 November were imported through Groww historical market data. Plan and
source scope were frozen before retrieval. A WAL-consistent SQLite backup was
quick-checked first. All eight returned 75/75 regular-session five-minute bars:
600 new candles, zero missing, zero conflicts, and no existing candles overwritten.
Research download 16 is complete. The single extra 15:30 bar per request was
excluded as outside the regular session. See `option-import-plan.json` and
`option-import-report.json`. Raw responses remain in
`/tmp/nifty-exact-options-20261104/`; they are source evidence, not live quotes.

An isolated SQLite copy was then used for one fixed descriptive replay of
selected artifact 9 over 2–3 November, using the current capital ₹100,000,
5% per-trade risk, 7% daily loss, 20 trades/day, five consecutive losses,
25% premium stop, 1.5 underlying reward/risk, and 0.05% slippage per side.
The specification hash is in `baseline-report.json`; no settings were chosen
after seeing the result. October 14–23 reserved history was not inspected.

| Evidence | Completed trades | Net after estimated costs |
|---|---:|---:|
| Actual selected SHADOW records, 2–3 November | 8 | −₹957.12 |
| Historical candle proxy replay of artifact 9 | 7 | +₹3,112.66 |

Only **one signal timestamp and contract matched** between the two lists. The
historical result is therefore **not evidence of a profitable model**. Its
option candles have no executable bid/ask or quote-freshness record; historical
volume/OI meanings have not been shown equivalent to runtime quote inputs.
Only the eight previously observed contract-days were imported, so alternative
strike availability is unknown. These are two already inspected sessions, not
fresh model validation. See `comparison.json` and `baseline-report.json`.

The next justified work is to establish signal-time feature and contract-choice
parity on fixed, fully covered sessions, then replay quote-aware portfolio rules
before fitting or selecting a replacement. Do not optimise a filter against
these two sessions or promote this candle proxy result. The 27 focused tests
for replay limits, backtest submission, risk and simulated execution passed.

## Indicator warm-up correction

The original replay began its indicators on 2 November, while SHADOW had up to
1,500 earlier spot candles in its rolling context. Saved SHADOW decision features
and a read-only reconstruction showed different RSI and moving-average inputs
at the same timestamps. Backtests now load up to 1,500 prior verified spot bars
and corresponding futures bars for indicator context. The period-only count
still controls the minimum-data gate, and earlier dates cannot produce entries.

A single fixed-specification replay on an isolated SQLite copy, using the same
model, dates and settings, produced these results:

| Evidence | Trades | Net after estimated costs | Matching SHADOW signal + contract |
|---|---:|---:|---:|
| Actual selected SHADOW | 8 | −₹957.12 | 8 by definition |
| Original historical candle proxy | 7 | +₹3,112.66 | 1 of 8 |
| Corrected warm-up candle proxy | 9 | −₹1,612.90 | 2 of 8 |

The original apparent proxy profit disappeared. **Six observed entries still do
not match**, so historical replay remains unsuitable for validating or tuning a
model. Existing stored candles can also differ from the original live decision
snapshot after source revisions. Historical option candles lack executable
bid/ask and quote freshness, historical option volume/OI equivalence is not
established, and only previously observed contracts were imported. The fixed
specification, full result and entry comparison are in `warmup-spec.json`,
`warmup-report.json`, and `warmup-comparison.json`. The specification was frozen
against the source before this replay; the final source refactor only extracted
the same candle query for regression testing. No model or runtime setting was
changed. The focused test suite passed 26 tests after this correction.
