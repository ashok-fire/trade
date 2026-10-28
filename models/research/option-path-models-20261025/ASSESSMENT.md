# Causal option-price-path study — 25 October 2026

Result: **REJECTED_LATER_HISTORICAL_AUDIT**. The >70% net-win goal was not met.
No model was promoted and no application settings, risk limits, registry rows
or orders were changed. The running app remains SHADOW with selected artifact9
and reference11; saved capital remains100000, not the research profile200000.

## Experiment

Eight fixed model profiles:44-input cost control versus50-input option-path
model, full history versus252-observed-day decay, capital100000 versus200000.
Each had CE-only and BOTH scopes and nine declared score cutoffs:144 selection
points, not144 independent experiments. Six day-purged chronological development
windows, fixed shallow histogram boosting, prior-six-month side calibration,
one aggregate fitting/calibration weight per day, one historical lot and at most
one selected trade/day. Percentage risk limits stayed3%/5%; capital affected
causal whole-lot eligibility and a10%-of-capital drawdown gate, not win labels.

New inputs used exact same-contract/day verified option candles through signal
close:1/3-bar returns, three-bar realised log volatility, signal range/close,
body/open and close location. Missing lag history stayed missing. Development
contained24781 opportunities across707 days; only8 rows lacked the three-bar
history and3 lacked the previous bar. This experiment did not fail because all
option-price inputs were unavailable. The frozen final estimator used all50
declared columns; no fitting column was entirely missing.

## Frozen candidate and later check

The separately declared economic track selected CE-only, capital200000,
252-day decay and cutoff0.45. **0.45 is a model score cutoff, not a claim of45%
or70% realised wins.** Policy was written and hashed before reading later paths.

| Measure | Development selection | Jan8–Oct13,2026 audit | Matched44-input audit control |
|---|---:|---:|---:|
| Completed selected trades |113|54|56|
| Net win rate |66.37%|42.59%|44.64%|
| Profit factor |1.829|0.774|0.719|
| Net P&L after fees |22022.35|-8397.80|-12341.98|
| Double-slippage net P&L |21612.90|-10181.67|-14148.54|
| Maximum drawdown |5707.56|12603.13|14269.90|
| Positive / active months |11/18|2/10|2/10|

The highest100+-trade path-model development point was BOTH at66.45% on152
trades, PF1.510. It lacked adjacent-cutoff support and was not the frozen audit
candidate. Neither development figure is an independent performance estimate.

The frozen candidate lost less money than its matched control, but its paired
95% day-block bootstrap improvement interval crossed zero: -62.17 to146.35 per
observed opportunity day. Its mean net expectancy interval was -526.37 to219.62
per trade. These exposed-history intervals do not establish an edge. Main
rejections include insufficient100-trade audit support, low wins/PF, negative
net and stressed P&L, poor monthly consistency and unsupported paired improvement.
No post-audit cutoff, capital or exit retuning was performed.

## Interpretation and next evidence

Recent option-price inputs helped some development comparisons, but did not
solve temporal generalisation. Increasing capital changed eligible contracts;
it did not produce a validated70% model. Missing option lag history is not the
main explanation in this batch. The audit model and calibration were frozen
from2025 data throughout2026; testing whether scheduled causal refresh helps
is a distinct hypothesis, not an established cause of failure.

The entire2026 audit was already inspected in earlier research. October14–23
was not read or used. All fills are conservative candle proxies, not executable
historical bid/ask quotes. Any eventual acceptance needs genuinely fresh frozen
quote-based forward observations; more searches on this history are not proof.

## Verification and files

- 11 focused tests passed, covering future-outcome invariance, contract/day/gap
 isolation, invalid/missing candles, full50-input fitting, fit-only missing-column
 handling, capital scaling, unknown validation outcomes and freeze-before-audit.
- Full suite: 169 tests passed; existing 307 deprecation warnings.
- specification.json and policy.json freeze the grid and selection; report.json
 contains all rejection reasons, audit calibration, control and uncertainty.
- matched-comparisons.json records every same-capital/decay/scope/cutoff contrast.
- development-features.pkl and feature-audit.json preserve the engineered inputs
 and their hash; the original source and label-cache hashes are in the spec.

Reproduce only into a new unused output folder:

```bash
.venv/bin/python -m research.option_path_models --output models/research/NEW-UNUSED-OPTION-PATH-STUDY
```

Do not rerun the identical grid expecting a different deterministic result.
