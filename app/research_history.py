"""Read-only presentation of offline study reports, never a model registry."""
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import re

from .pagination import pagination

MAX_REPORT_BYTES = 2*1024*1024
TITLES = {
    "monthly-refresh":"Monthly retraining",
    "option-path-models":"Option price-path inputs",
    "effective-day-models":"Effective-day and cost inputs",
    "frozen-ce-meta-audit":"Frozen provisional CE audit",
    "premium-meta-models":"Premium-outcome models",
    "pooled-day-models":"Pooled daily models",
    "decision-day-specialists":"First-decision specialists",
    "premium-exits":"Premium stop/target study",
    "precision-70":"Precision >70% study",
    "walk-forward-capital":"Capital-aware walk-forward",
    "profit-objective":"Profit-objective regression",
}


def report_path(root: Path, study_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", study_id):
        raise ValueError("Invalid study identifier")
    root = Path(root).resolve()
    folder = root/study_id
    path = folder/"report.json"
    if folder.is_symlink() or path.is_symlink() or path.resolve().parent.parent!=root:
        raise ValueError("Report must remain inside the research directory")
    return path


def read_report(root, study_id):
    path = report_path(root, study_id)
    with path.open("rb") as stream:
        content = stream.read(MAX_REPORT_BYTES+1)
    if len(content)>MAX_REPORT_BYTES:
        raise ValueError("Report exceeds display size limit")
    data = json.loads(content)
    if not isinstance(data, dict):
        raise ValueError("Report must be an object")
    return data


def metric_summary(value):
    if not isinstance(value, dict):
        return None
    if isinstance(value.get("metrics"), dict):
        return metric_summary(value["metrics"])
    if "trades" not in value:
        return None
    result = {}
    for key in ("trades", "win_rate_pct", "profit_factor", "net_pnl", "stress_net_pnl"):
        item = value.get(key, value.get("double_slippage_net_pnl") if key=="stress_net_pnl" else None)
        result[key] = item if isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item) else None
    return result


def summarize_report(study_id, report, modified):
    development, development_label = None, "Development (selection-biased)"
    selection = report.get("selection") or {}
    candidates = [report.get("development"), report.get("selected_validation"),
                  selection.get("selected") if isinstance(selection, dict) else None, report.get("tuning")]
    for item in candidates:
        development = metric_summary(item)
        if development is not None:
            break
    if development is None:
        for key, value in report.items():
            if key.startswith("highest_win"):
                development = metric_summary(value)
                if development is not None:
                    development_label = "Best development diagnostic only"
                    break
    later = next((item for item in (report.get("audit"), report.get("historical_audit"), report.get("historical_evaluation"))
                  if isinstance(item, dict)), None)
    evaluation = metric_summary(later)
    reasons = report.get("target_rejections") or (later or {}).get("rejections") or report.get("activation_blockers") or []
    # This study is a fixed three-arm mechanism diagnostic, not development tuning.
    if isinstance(report.get("metrics"), dict) and "monthly50" in report["metrics"]:
        evaluation = metric_summary(report["metrics"]["monthly50"])
    status = str(report.get("status") or ("REJECTED" if report.get("activation_blockers") else "COMPLETED"))
    title = next((title for prefix, title in TITLES.items() if study_id.startswith(prefix+"-")), study_id)
    return {"id":study_id, "title":title, "modified":modified, "status":status,
            "development":development, "development_label":development_label,
            "evaluation":evaluation, "rejections":[str(reason) for reason in reasons] if isinstance(reasons, list) else [],
            "rejected":status.startswith("REJECTED"), "read_error":False}


def research_history(root, page=1):
    root = Path(root)
    entries = []
    for path in root.glob("*/report.json"):
        try:
            report_path(root, path.parent.name)
            entries.append((path.stat().st_mtime, path.parent.name))
        except (OSError, ValueError):
            continue
    entries.sort(reverse=True)
    pager = pagination(len(entries), page)
    needed = entries[pager["offset"]:pager["offset"]+pager["per_page"]]
    rows = {}
    for stamp, study_id in needed:
        try:
            rows[study_id] = summarize_report(study_id, read_report(root, study_id), datetime.fromtimestamp(stamp, UTC))
        except (OSError, ValueError):
            rows[study_id] = {"id":study_id, "title":study_id, "modified":datetime.fromtimestamp(stamp, UTC),
                              "status":"REPORT_UNAVAILABLE", "development":None, "evaluation":None,
                              "rejections":["Report could not be read; retained on disk for inspection."], "read_error":True, "rejected":False}
    return {"items":[rows[name] for _, name in entries[pager["offset"]:pager["offset"]+pager["per_page"]]],
            "pagination":pager}
