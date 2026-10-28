# NiftyPilot

NiftyPilot is a personal Indian index-options bot for NIFTY 50, BANK NIFTY and SENSEX. It supports Upstox and Groww while deliberately keeping the application small: one FastAPI process, one SQLite database, server-rendered pages, no login, no Redis and no Docker.

## Features

- `STOPPED`, `PAPER` and `LIVE` modes stored in SQLite and changed from the dashboard without a restart.
- One selectable execution broker: Upstox or Groww. Broker changes require Stopped mode, no open position and no pending order.
- One active broker across live data, Paper, Live, research download, model training and backtests. Switching broker changes the whole workflow without deleting stored results.
- One common `models/regime.json` model. It can be trained from stored Groww data and later used with the compatible Upstox live feed; broker-specific model folders are not created.
- Completed five-minute candle strategy with spot price structure plus futures VWAP, volume and basis, EMA 9/20/50, ATR, RSI and opening-range features.
- Fail-closed behavior when the model, historical contracts or verified option premiums are missing.
- Dynamic option contracts, non-expiry-day ATM selection with one-strike ITM fallback, and no OTM buying.
- Shared position sizing and risk gates. Initial Live mode is capped at one lot and the portfolio at one open position.
- Paper fills at ask plus slippage; paper exits and open P&L use bid minus slippage.
- Persistent provider-tagged signals, orders and trades with position reconciliation before Live mode.
- Backtests replay verified real option candles and apply selectable Upstox Basic, Upstox Plus or Groww costs.
- Telegram group and optional private-owner alerts through a persistent retry outbox.
- Overview, Orders, Backtests, Telegram and Settings screens.
- Data-readiness matrix with candle coverage, missing intervals, spot/futures alignment, zero-volume percentage and verified contract metadata.
- Background, resumable research downloads with persisted progress. Re-submitting a range safely stores only missing candles.
- Candidate-model registry with confusion matrix, per-class precision/recall, feature importance and explicit activation.
- Historical contract-metadata CSV import for records that cannot be verified from today's broker instrument master.
- Rule-only comparison backtests, ATM-only versus ATM→ITM1 policies, liquidity/spread rejection, entry delay, adaptive slippage and one-minute stop/target resolution when available.
- Full Paper/Live decision audit including rule checks, bid/ask, slippage, delta, IV, risk budget, stop and target.
- Manual pause and position exit controls, one-minute Live reconciliation, immediate risk shutdown alerts and post-restart Live reconfirmation.

The model predicts regime, direction and confidence. It does not invent stop and target prices: the stop is derived from the underlying ATR, the target uses the configured reward/risk ratio, and a separate premium emergency stop protects against option-price shocks.

## Local setup

Python 3.12 is recommended.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

On macOS, install OpenMP for XGBoost first:

```bash
brew install libomp
```

On Windows, create the environment from PowerShell and install the current Microsoft Visual C++ Redistributable if XGBoost reports a missing runtime DLL:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

On Ubuntu, the normal setup above is sufficient. If the XGBoost wheel reports a missing OpenMP library, install the distribution's `libgomp1` package.

Open `http://127.0.0.1:8000`. SQLite tables and default settings are created on first start.

## Broker setup

Open **Settings**, select one broker and configure only the credentials shown for it. Inactive-broker credentials are hidden to avoid using the wrong feed accidentally.

### Upstox

- Analytics Token: read-only market data, contracts, Paper mode and historical downloads.
- Trading Token: order placement, fills, positions and Live reconciliation.

The credentials are intentionally separate. A missing Trading Token disables Live mode but does not interrupt Paper or Backtest. The application validates a replacement token before saving it.

Expired Upstox option contracts and their candles require Upstox Plus. Index candles alone are not enough for this model because cash-index volume is zero. Training now requires aligned, verified index-futures candles for liquidity features. Trustworthy historical CE/PE P&L additionally requires exact expired-option candles and verified historical lot/tick metadata.

### Groww

- Groww Access Token: live data, historical data, Paper mode, orders and reconciliation.
- An active Groww Trading API subscription is required while downloading data or using Groww for Paper/Live. Previously downloaded data does not require the token.
- Groww documents expired F&O candles from 2020. Backtests still require an exact historical contract with its historical lot size and tick size; if those details cannot be verified from the instrument master, NiftyPilot rejects the period instead of using today's values.

Tokens are stored in `data/secrets/` with `0600` permissions. Do not commit or share them.

## Research and backtesting

Use the Backtests page in this order. Download, training and replay automatically use the broker selected in Settings:

1. Download verified five-minute history. The application always requests both spot-index and aligned index-futures candles; enable the option checkbox when historical CE/PE P&L is required.
2. Train a candidate XGBoost model. Candidate files are kept under `models/candidates/`; training does not silently replace the active model.
3. Confirm that the Backtests screen shows enough verified futures candles and, for trade replay, verified historical option candles.
4. Review its chronological validation metrics, confusion matrix, confidence distribution and feature importance, then explicitly activate it. Activation atomically updates the single common `models/regime.json` and archives the previous version.
5. Run a backtest beginning after the model training end date. Run the same period with **Rule-only comparison** to check whether the model adds value.
6. Review warnings, P&L, drawdown, monthly/exit/strike/side breakdowns, doubled-slippage stress and exported CSV trades.

Historical data is always provider-tagged. A Groww backtest cannot accidentally consume Upstox candles and vice versa. The UI submits the active broker on every research step. Model files are deliberately not provider-specific because both feeds are normalized to the same `spot-futures-v2` feature schema. The training provider, date coverage and validation metrics remain recorded in common model metadata for auditability.

The engine never synthesizes option premiums from index prices. When historical bid/ask is unavailable, it uses verified option OHLC with configured two-sided slippage and reports the limitation. A historical Groww symbol may be downloaded even when it has disappeared from today's instrument master, but it is marked unverified and excluded from P&L until its historical lot size is known.

For pending Groww metadata, export the CSV from **Backtests → Historical contract metadata**, fill lot size and tick size from a trustworthy broker/exchange archive, and import it. The importer only updates already-downloaded matching contracts and records `CSV_IMPORT` as the source.

Research downloads run in the application background. The run and its progress survive page navigation. If the process stops, the run is marked interrupted; submit the same provider, indices and dates again to resume because candle uniqueness prevents duplicate storage. The provider may still be queried for an existing date chunk, but existing rows are not inserted twice.

### Switching after a one-month Groww subscription

1. While the Groww subscription is active, save the Groww token and download the desired spot, futures and expired-option history.
2. Keep Groww selected, train the common model and finish the Groww backtests.
3. Keep the SQLite database and `models/regime*` files in your backups.
4. Stop execution and select Upstox. Upstox then becomes the source for new downloads, training and backtests as well as Paper/Live.
5. The common model trained earlier remains available because it records a normalized, provider-neutral feature schema. Historical Groww candles and old results stay in SQLite, but the simplified UI does not mix them into an Upstox run.

Switching broker does not remove candles, model files or previous backtest results. A future data refresh from Groww requires selecting Groww again and supplying a valid subscription/token.

## Paper and live rollout

1. Keep execution Stopped while configuring the broker, Telegram and risk values.
2. Refresh the selected broker's instrument master.
3. Train the model and complete backtests.
4. Run Paper mode for several sessions and verify signals, option selection, fills, exits, P&L and Telegram messages.
5. Complete the configured Paper-session target (20 distinct trading days by default), verify the static-IP acknowledgement, reconcile broker positions, type `LIVE TRADING`, and begin with the enforced one-lot cap.

Only one broker can execute at a time. Existing trades retain their provider, so exits and reconciliation always route back to the broker that opened them.

Live entries are paused after every application restart and require `LIVE TRADING` again. Broker/database positions are reconciled every minute. Use **Pause new entries** to stop new signals while preserving position monitoring, and use the explicit manual-exit control for an open Paper or Live position. A daily-loss or consecutive-loss breach immediately pauses entries and creates an owner alert.

The Paper-session Live gate is enabled by default at 20 distinct dates containing a closed Paper trade. It can be disabled or changed from **Settings → Live safety gate** without restarting. Disabling it skips only this one gate; model, token, feed, static-IP, open-position and reconciliation checks still apply. Every change is written to system events and queued for the owner-private Telegram destination.

## Telegram

1. Open `@BotFather`, run `/newbot`, choose a name and username, and copy the token.
2. Create a private Telegram group with your friend and add the bot. Send one message such as `/start@your_bot_username` in the group.
3. Open the bot privately, press **Start**, and send one message if you want owner-only failure alerts.
4. Find IDs with `curl "https://api.telegram.org/bot<TOKEN>/getUpdates"`. The group `chat.id` is normally a negative number such as `-100...`; the private `chat.id` is the owner ID.
5. Save `TELEGRAM_BOT_TOKEN=<TOKEN>` in `/opt/niftypilot/.env`, run `chmod 600 /opt/niftypilot/.env`, and restart the service.
6. Open **Telegram** in NiftyPilot, enter the group ID and optional owner ID, enable the destinations, save, and use both test buttons.

The token stays in the server environment; destination IDs and queued delivery state stay in SQLite. The outbox retries temporary Telegram failures and avoids duplicate sends. Treat the token as a password and revoke it in BotFather if it is ever exposed.

## Free deployment and static IP

A computer cannot create its own public static IP. For a home Windows, macOS or Ubuntu machine, either buy a static public IPv4 address from the ISP or run NiftyPilot on a cloud VM with a reserved public IP. Dynamic DNS changes a hostname only; it does not create the fixed outbound IP required for broker orders.

Oracle Cloud Always Free is the practical starting option for Paper mode:

1. Create an Always Free Ubuntu VM in the account home region. An Ampere A1 Flex VM with 1 OCPU and 6 GB RAM is a reasonable small configuration for FastAPI and XGBoost.
2. In OCI, open **Networking → IP management → Reserved public IPs**, reserve an IP, and assign it to the VM's private IP.
3. Allow inbound SSH only. Do not expose port 8000.
4. Copy the project to `/opt/niftypilot`, create the `niftypilot` system user and Python environment, then install `requirements.txt` and Ubuntu's `libgomp1` package.
5. Copy `.env.example` to `/opt/niftypilot/.env`, keep `HOST=127.0.0.1`, set the Telegram token, and run `chmod 600 /opt/niftypilot/.env`.
6. Install and start the included service and backup timer using the commands below.
7. Register the VM's reserved public IP with the active broker. Groww uses the API-key page; Upstox uses **My Apps → Static IPs**. Re-check the broker documentation before enabling Live because IP and token rules can change.

Free compute capacity is not guaranteed and Oracle may reclaim idle Always Free instances. Use it first for backtesting and Paper. For real-money Live trading, monitor availability and move to a paid, reliable VM if interruptions occur.

## Server access

Bind only to `127.0.0.1` and access the dashboard through SSH:

```bash
ssh -L 8000:127.0.0.1:8000 ubuntu@SERVER_IP
```

The included `trading-bot.service` assumes `/opt/niftypilot` and a dedicated `niftypilot` system user. The backup script creates a SQLite-consistent copy, runs `PRAGMA quick_check`, copies the active model and its metadata, and keeps seven days of verified backups.

Install the optional nightly systemd timer:

```bash
sudo cp trading-bot.service niftypilot-backup.service niftypilot-backup.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now trading-bot.service niftypilot-backup.timer
systemctl list-timers niftypilot-backup.timer
```

The Settings screen reports backup freshness and verifies that stored broker token files are not readable by other users. `/health` reports database integrity, active-model status, feed age, backup state, secret-file permissions and the current execution state.

## Parallel Shadow evidence

Open `/shadow-research` to compare ORB, PULLBACK, FAILED_BREAKOUT and
VWAP_RECLAIM independently. While enabled in SHADOW mode, the observer checks
fresh closed five-minute spot/futures bars during 09:35–15:10 IST on weekdays.
It only uses market-data APIs: it never sends orders, changes the selected Shadow
model, or promotes a candidate. Entries paused in the main trading controls do
not pause this separate research observer; use its own pause button.

The observer freezes one compatible V6 research artifact in
`parallel_shadow_model_artifact_id`. Scores are diagnostic, not validated win
probabilities; unsupported providers/indices remain unscored. Every executable
candidate is simulated, including research-threshold rejections. One-lot trials
use observed ask/bid with slippage, estimated charges, a 1 ATR underlying stop,
1.5 ATR target, 25% premium stop, and first-observed square-off from 15:20 IST.
Tracking polls every 15 seconds and cannot reconstruct unobserved intrabar
touches. Missing same-day exits are INCOMPLETE, never manufactured labels.
Keep the service and machine awake for the entire session.

The daily comparison selects the first executable observation per
family/direction/day *before* considering outcome completeness. Separate groups
preserve model, provider, index and protocol settings. Overlapping trials are
counterfactual examples, not a portfolio return. Export all evidence from
`/shadow-research/export.jsonl`; `label_net_win` is null unless CLOSED. Use future
chronological, session-grouped research splits; do not randomly split overlapping
observations or tune repeatedly against a held-out period. Collection does not
automatically retrain, activate a model, or establish an improved win rate.

## Decision-aligned training

New option-quality training runs use the versioned `decision-aligned-v1`
contract. They train on every eligible closed-bar opportunity, use the same
first-confirmed setup priority and ATM→ITM1 policy as runtime, and exclude
option-volume/OI model features until historical/live meanings are reconciled.
Quote tradability filters remain unchanged. New runs are single-index only;
multi-index portfolio equivalence has not been established.

Fitting, probability calibration, threshold tuning and historical evaluation
use disjoint chronological day groups with boundary purges. Trading days have
equal total fitting weight. Incomplete exit paths remain unlabelled
opportunities in evaluation; the first selected missing outcome is never
replaced by a later known outcome. Current-session eligibility does not depend
on whether the remainder of that session eventually has complete data.
Frozen holdout evaluation reads each bundle's own feature list and contract,
not legacy module defaults, and requires dates after its research date range.

The fixed NIFTY ORB experiment (no estimator/regime search) is reproducible with:

```bash
.venv/bin/python -m app.aligned_training --output models/research/NEW-UNUSED-RUN/candidate.json
```

It uses fixed fitting/calibration/tuning/evaluation dates through 13 October
2026 and does not consume the reserved 14–23 October period. The specification
is saved before fitting; existing candidate paths are never overwritten. The
24 October result and its blockers are documented in
`models/research/decision-aligned-orb-20261024/README.md`. Do not repeatedly
rerun or tune this experiment against the already inspected evaluation period.

Candidates remain research-only and are never selected or activated by this
command. Historical labels still use candle-based fill proxies, not real
bid/ask or the option quote at an underlying stop touch. Forward quote/fill
evidence and full portfolio/risk validation remain necessary. Runtime refuses
new aligned bundles whose frozen geometry, provider or index scope no longer
matches current settings; legacy bundle behaviour is preserved.

If the service is running under the VS Code debugger, restart that debug
session before using the new training route. Offline experiment commands load
the updated code directly without restarting or disturbing the running app.

## Capital-aware offline tuning

The configured defaults are ₹100,000 capital, 3% gross stop-risk per trade and
5% daily loss entry-gate limit, explicitly requested on 25 October 2026.
The underlying reward/risk target remains 1.5 and the separate premium emergency
stop remains 25%. A stop-risk budget is not a guaranteed maximum realised loss;
fees, slippage and gaps can increase it. Saved personal settings are not silently
overwritten when the application starts. New backtests snapshot the daily limit;
older saved runs without that field retain their original 0.75% assumption.

The bounded offline search is run with a new, unused output directory:

```bash
.venv/bin/python -m research.walk_forward_tuning --output models/research/NEW-UNUSED-TUNING-RUN
```

It compares six fixed shallow CatBoost configurations, six day-grouped forward
validation windows in 2024–2025, and nine shared CE/PE thresholds. Validation
is used for selection, not claimed as an unbiased final test. Selected settings
are frozen before a previously exposed 2026 historical audit. The reserved
14–23 October window is not read. It retains the existing 40 features, ORB
strategy, candle simulator and cost assumptions; there are no new live features.
Capital-aware replay uses one historical lot, not full risk-budget allocation;
signal-close affordability is only a proxy for the later executable entry quote.
Each validation quarter resets its cash ledger, so their concatenated P&L is
not a continuous compounded account return. An incomplete selected exit halts
the affected ledger rather than inventing a zero P&L.

The 25 October study rejected the selected candidate after its 2026 audit:
61 trades, 40.98% wins, PF 0.848, net -₹4,161.37. Details are in
`models/research/walk-forward-capital-20261025/ASSESSMENT.md`. Nothing in this
command calls order endpoints, writes model-registry state, or activates a model.

The subsequent >70% realised-win study is reproducible with a new output path:

```bash
.venv/bin/python -m research.precision_tuning --output models/research/NEW-UNUSED-PRECISION-RUN
```

This uses the fingerprint-verified local dataset cache from the 25 October
capital-aware study, four stronger-regularisation configurations, and nineteen
shared thresholds. A win rate strictly above 70% must have at least 100 complete
selected trades, positive net and stressed P&L, stable windows, acceptable
drawdown, and supported neighbouring thresholds. A 0.70 prediction score or
a handful of wins does not satisfy that target. The completed result is in
`models/research/precision-70-20261025/README.md`: none qualified; the highest
diagnostic win rate with at least 100 selection-validation trades was 53.6%
over 125 trades. No extra candidate was tested on 2026 after this rejection.
Do not repeat the same deterministic search on unchanged evidence.

The next user-authorised strategy study tests eight premium-native stop/target/
holding-period combinations, CE/PE separately, and three fixed conditions from
existing trend-strength features. It makes no runtime strategy changes:

```bash
.venv/bin/python -m research.exit_strategy_study --output models/research/NEW-UNUSED-EXIT-STUDY
```

Its 48 development combinations produced nine positive after-cost point
estimates but no strategy passing the frozen drawdown/stability/economic gates.
Per its preregistered conditional workflow, no classifier was fitted and no
additional 2026 candidate audit was performed. See
`models/research/premium-exits-20261025/ASSESSMENT.md`. This is not evidence
that all further work on the stored data is exhausted, or a win-rate ceiling.

## Ongoing model research

The current experiment ledger, results and next preregistered hypothesis are
in `research/README.md`. Subsequent tests cover first-decision/day learning,
pooled calibration, premium-outcome meta-labels, effective-day regularisation,
causal option-cost inputs and same-contract option-price-path inputs. These are offline research scripts, not new
runtime trading features. No research bundle was activated.

A provisional CE result above70% on68 development trades failed its frozen
2026 audit at32.26% on31 trades. Increasing research capital to200000 produced
71.23% on73 development trades, still insufficient evidence. Neither result is
a validated >70% trading model. Keep development selection separate from audit
and genuinely fresh forward evidence; preserve rejected experiments.

The latest50-input option-price-path candidate reached66.37% on113 development
trades but failed its frozen later2026 audit:42.59% on54trades, PF0.774 and
net -8397.80. It was not activated; see
`models/research/option-path-models-20261025/ASSESSMENT.md`.

The fixed monthly-refresh follow-up also failed:38.46% on26trades, PF0.631,
net -8507.53, identical selected outcomes with44 or50 inputs. No model was
activated. See `models/research/monthly-refresh-20261025/README.md`.
Further follow-ups wait for meaningful fresh quote-based Shadow evidence rather
than expanding rejected searches on the same historical audit.

The Backtests page now includes read-only offline research history from saved
study reports, with development and later historical results kept separate.
Research, model, download and backtest histories use 10 items per page; the
Recent runs panel shows the latest 5 backtests. Pagination never deletes older
records or changes model selection, trading mode or risk settings.

## Verification

```bash
python -m pytest -q
python -m compileall -q app
```

Automated tests never call real order endpoints. Validate both broker integrations in Paper mode, and use a broker sandbox or the smallest allowed quantity before relying on Live execution.
