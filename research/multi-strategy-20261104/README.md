# Fixed multi-strategy NIFTY diagnostic — 4 November 2026

The user requested a broader strategy search after the selected ORB-only model
missed an inside-range market move. This research-only batch froze five distinct
rule families and their fixed-priority, earliest-signal-per-day union before
reading outcomes. A rule-only ORB arm used the same 09:45–14:30 IST window and
one-lot option replay. This is **not** a direct comparison with selected model 9.

`specification.json` records source/code SHA-256, all rules, chronological
periods, costs, gates, and reserved exclusions. An initial preflight failed on
a specification key mismatch before any history was loaded; its exact file is
retained as `specification-preflight-failed.json`. The corrected specification
was frozen before the sole outcome run. The output contains every first-daily
signal, including incomplete and risk-ineligible rows, in the seven JSONL files.

The immutable SQLite snapshot was the earlier 4 November morning offline copy.
The study used verified NIFTY spot, front-future and option five-minute candles
through 13 October 2026. October 14–23 and all subsequent days, including the
visible 4 November session that motivated the idea, were excluded. Development
is 2023–2025; the January–13 October 2026 comparison has already been exposed
to earlier research and is **not a fresh final test**. No parameters were tuned
after seeing this output.

| Exposed 2026 rule arm | First signals / completed | Net wins | Net after estimated costs | Double-slippage net |
|---|---:|---:|---:|---:|
| ORB rule control | 183 / 174 | 40.23% | −₹13,088.77 | −₹14,845.62 |
| Rolling 30-minute breakout | 191 / 180 | 38.89% | −₹28,119.19 | −₹29,896.95 |
| VWAP trend pullback | 182 / 170 | 37.65% | −₹35,648.65 | −₹37,316.34 |
| Opening-range edge rejection | 73 / 70 | 40.00% | −₹5,784.13 | −₹6,448.41 |
| VWAP reversion | 0 / 0 | — | — | — |
| Gap fade | 102 / 91 | 40.66% | −₹7,901.92 | −₹8,716.80 |
| **Fixed first-of-five union** | **191 / 179** | **39.11%** | **−₹18,783.24** | **−₹20,530.36** |

The union had 4/10 positive active months, ₹31,264 maximum cash drawdown and
₹12,344 estimated charges on the 179 completed one-lot outcomes. Development
also failed: 727 completed union trades, 40.58% net wins, −₹74,698 versus the
rule-only ORB control's 696 completed, 41.38%, −₹49,031. On 177 known paired
2026 days the mean union-minus-ORB difference was −₹8.07/day; the fixed
day-bootstrap 95% interval was −₹228.23 to +₹224.25/day, with 15 unknown paired
days retained. The union failed every exposed gate except 100-trade support.

**VWAP-reversion data defect:** all 74,169 verified historical NIFTY spot bars
in scope have zero volume. The app's spot VWAP therefore falls back to the
current close, making this arm's previous-close-vs-spot-VWAP condition
unreachable. Its zero signals do not test whether mean reversion works. Any
future study needs a separately frozen futures-VWAP or price-weighted definition;
this failed arm must not be silently repaired and rescored against the same
comparison as if it were preregistered.

Historical option candles have no executable bid/ask, quote freshness, IV or
delta. The next-open fills, conservative five-minute stop-first exits, cash and
one-lot risk checks, Groww charge estimate, and doubled slippage are proxies.
The affordability check uses fixed ₹100,000 reference capital rather than
reducing cash after each loss, so this is not a deployable continuous portfolio
ledger. The preregistration hashed the study script and immutable database;
imported helper versions were not individually hashed. These limits cannot
rescue the negative result or establish a live-fill advantage. No learner was
trained, model was
selected, trading rule was activated, setting or SHADOW mode was changed, or
broker order was sent. Twelve focused tests for the new signal, replay and risk
paths passed. The objective >70% *net* win rate was not demonstrated.
