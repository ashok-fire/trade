from datetime import UTC, date, datetime
from html import unescape
import json
import os
import re
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from sqlalchemy import func, select

from app import dashboard
from app.models import BacktestRun, ModelArtifact, ResearchSyncRun, RuntimeSetting, Setting
from app.pagination import database_page, pagination
from app import research_history as history


def write_report(root, name, report, stamp=1):
    folder = root/name
    folder.mkdir(parents=True)
    path = folder/"report.json"
    path.write_text(json.dumps(report))
    os.utime(path, (stamp, stamp))
    return path


def metric(wins=42.59, trades=54):
    return {"trades":trades, "win_rate_pct":wins, "profit_factor":.774,
            "net_pnl":-8397.8, "stress_net_pnl":-10181.67}


def test_page_defaults_bounds_and_empty_state():
    assert pagination(21)["per_page"]==10
    assert pagination(21, 2)==dict(page=2, pages=3, per_page=10, total=21, offset=10, first=11, last=20)
    assert pagination(21, 999)["page"]==3
    assert pagination(21, -5)["page"]==1
    assert pagination(0, 99)["first"]==pagination(0, 99)["last"]==0


def test_stylesheet_version_changes_with_asset_and_handles_missing_file(tmp_path):
    from app.main import static_asset_version
    folder = tmp_path/"static"
    folder.mkdir()
    path = folder/"app.css"
    path.write_text("body{}")
    first = static_asset_version("app.css", tmp_path)
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns+1000000))
    assert static_asset_version("app.css", tmp_path)!=first
    assert static_asset_version("missing.css", tmp_path)=="0"


@pytest.mark.parametrize("total,current", [(0, 1), (11, 1), (11, 2), (1000, 1), (1000, 50), (1000, 100)])
def test_numbered_pager_active_disabled_and_bounded_links(total, current):
    from app.main import templates
    from starlette.requests import Request
    request = Request({"type":"http", "method":"GET", "scheme":"http", "server":("testserver", 80),
                       "path":"/backtests", "query_string":b"models_page=2&run_id=1", "headers":[]})
    module = templates.env.get_template("partials/pagination.html").make_module({"request":request})
    html = str(module.pagination(pagination(total, current), "research_page", "research-history"))
    assert html.count('aria-current="page"')==1
    assert f'aria-label="Page {current}, current page"' in html
    assert html.count('class="pager-button pager-number')<=7
    if current==1:
        assert 'disabled aria-label="Previous page"' in html
    if current==pagination(total, current)["pages"]:
        assert 'disabled aria-label="Next page"' in html
    if total==1000 and current==50:
        assert 'class="pagination-gap"' in html
        assert 'aria-label="Go to page 1"' in html and 'aria-label="Go to page 100"' in html


def test_report_history_pages_are_stable_and_not_truncated(tmp_path):
    for i in range(23):
        write_report(tmp_path, f"study-{i:02d}", {"status":"REJECTED", "audit":metric()}, stamp=1)
    pages = [history.research_history(tmp_path, page) for page in (1, 2, 3)]
    assert [len(p["items"]) for p in pages]==[10, 10, 3]
    ids = [r["id"] for p in pages for r in p["items"]]
    assert ids==[f"study-{i:02d}" for i in range(22, -1, -1)]
    assert history.research_history(tmp_path, 999)["items"]==pages[-1]["items"]


@pytest.mark.parametrize("report,expected_dev,expected_eval", [
    ({"development":metric(70.59, 68), "audit":metric(32.26, 31)}, 70.59, 32.26),
    ({"selection":{"selected":{"metrics":metric(66.37, 113)}}, "audit":{"metrics":metric()}}, 66.37, 42.59),
    ({"highest_win_100plus_diagnostic_only":{"metrics":metric(60.73, 191)}, "audit":None}, 60.73, None),
    ({"metrics":{"frozen50":metric(), "monthly50":metric(38.46, 26), "monthly44":metric(38.46, 26)}}, None, 38.46),
    ({"tuning":metric(0, 0), "historical_evaluation":metric(0, 0), "activation_blockers":["No entries"]}, 0, 0),
])
def test_schema_adapters_keep_development_separate_from_later_checks(report, expected_dev, expected_eval):
    result = history.summarize_report("study", report, datetime.now(UTC))
    assert (result["development"]["win_rate_pct"] if result["development"] else None)==expected_dev
    assert (result["evaluation"]["win_rate_pct"] if result["evaluation"] else None)==expected_eval


def test_missing_bad_and_nonfinite_metrics_are_not_shown_as_wins(tmp_path):
    assert history.metric_summary({}) is None
    assert history.metric_summary({"trades":10, "win_rate_pct":float("nan")})["win_rate_pct"] is None
    path = write_report(tmp_path, "broken", {})
    path.write_text("{bad")
    result = history.research_history(tmp_path)["items"][0]
    assert result["read_error"] and result["status"]=="REPORT_UNAVAILABLE"
    assert history.research_history(tmp_path/"missing")["pagination"]["total"]==0


def test_report_reader_blocks_traversal_symlinks_nonobjects_and_oversized_files(tmp_path, monkeypatch):
    root = tmp_path/"reports"
    write_report(root, "normal", {"status":"REJECTED"})
    assert history.read_report(root, "normal")["status"]=="REJECTED"
    for name in ("../normal", "normal/report.json", "..", ""):
        with pytest.raises(ValueError):
            history.read_report(root, name)
    outside = tmp_path/"outside"
    outside.mkdir()
    (outside/"report.json").write_text("{}")
    (root/"linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        history.read_report(root, "linked")
    assert history.research_history(root)["pagination"]["total"]==1
    write_report(root, "array", [])
    with pytest.raises(ValueError):
        history.read_report(root, "array")
    monkeypatch.setattr(history, "MAX_REPORT_BYTES", 1)
    with pytest.raises(ValueError, match="size limit"):
        history.read_report(root, "normal")


def seed_histories(session, count=23):
    stamp = datetime(2026, 10, 25, tzinfo=UTC)
    for i in range(count):
        session.add(BacktestRun(configuration_json="{}", status="COMPLETE", start_date=date(2024, 1, 1), end_date=date(2024, 12, 31), created_at=stamp))
        session.add(ResearchSyncRun(provider="GROWW", configuration_json="{}", status="COMPLETE", created_at=stamp))
        session.add(ModelArtifact(version=f"model-{i}", status="SHADOW" if i==0 else "CANDIDATE", model_path="none", metadata_path="none",
            metadata_json=json.dumps({"version":f"model-{i}", "feature_schema":"cost-aware-option-v5"}), created_at=stamp))
    session.commit()


def test_database_paging_latest5_and_selected_shadow_are_independent(session, tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "config", SimpleNamespace(root=tmp_path, model_path=tmp_path/"absent.json"))
    monkeypatch.setattr(dashboard, "research_readiness", lambda session:[])
    seed_histories(session)
    for model in (BacktestRun, ModelArtifact, ResearchSyncRun):
        first, _ = database_page(session, model)
        second, _ = database_page(session, model, 2)
        assert len(first)==len(second)==10
        assert {r.id for r in first}.isdisjoint(r.id for r in second)
    context = dashboard.backtests_context(session, 1, runs_page=2, models_page=2, downloads_page=2)
    assert len(context["recent_runs"])==5
    assert [r.id for r in context["recent_runs"]]==[23, 22, 21, 20, 19]
    assert context["selected"].id==1
    assert context["shadow_model_metadata"]["version"]=="model-0"
    assert context["shadow_model_ready"]
    assert all(context[key]["total"]==23 for key in ("runs_pagination", "models_pagination", "downloads_pagination"))
    assert session.scalar(select(func.count()).select_from(ModelArtifact))==23
    assert session.get(RuntimeSetting, 1).execution_mode=="STOPPED"
    assert session.get(Setting, "paper_capital").value=="100000"


@pytest.mark.asyncio
async def test_backtests_page_renders_pagination_preserves_other_pages_and_exposes_reports_read_only(session, tmp_path, monkeypatch):
    from app import main
    root = tmp_path/"models/research"
    for i in range(12):
        write_report(root, f"study-{i:02d}", {"status":"REJECTED_TEST", "audit":metric(), "development":metric(75, 20)}, stamp=i)
    config = SimpleNamespace(root=tmp_path, model_path=tmp_path/"none.json")
    monkeypatch.setattr(dashboard, "config", config)
    monkeypatch.setattr(main, "config", config)
    monkeypatch.setattr(dashboard, "research_readiness", lambda session:[])
    seed_histories(session)
    main.app.dependency_overrides[main.get_session] = lambda:session
    try:
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://testserver")
        response = await client.get("/backtests?research_page=1&models_page=2&downloads_page=2&runs_page=2&run_id=1")
        assert response.status_code==200
        html = response.text
        assert html.count('data-research-study=')==10
        assert "Offline research history" in html and "10 per page" in html
        assert re.search(r'/static/app.css\?v=\d+', html)
        next_links = [urlsplit(unescape(url)) for url in re.findall(r'href="([^"]+)"', html)
                      if url.endswith("#research-history")]
        assert any(parse_qs(url.query)=={"research_page":["2"], "models_page":["2"], "downloads_page":["2"],
                                         "runs_page":["2"], "run_id":["1"]} for url in next_links)
        latest = html.split('id="latest-backtest-runs"', 1)[1].split("</section>", 1)[0]
        assert latest.count('class="run-item ')==5
        assert "Latest 5" in latest
        assert "Development (selection-biased)" in html
        second = await client.get("/backtests?research_page=2")
        assert second.status_code==200 and second.text.count('data-research-study=')==2
        assert (await client.get("/backtests?research_page=0")).status_code==422
        assert (await client.get("/backtests?research_page=bad")).status_code==422
        report = await client.get("/backtests/research-reports/study-11")
        assert report.status_code==200 and report.json()["audit"]["win_rate_pct"]==42.59
        assert report.headers["x-content-type-options"]=="nosniff"
        assert (await client.get("/backtests/research-reports/missing")).status_code==404
        assert (await client.post("/backtests/research-reports/study-11")).status_code==405
        assert session.get(RuntimeSetting, 1).execution_mode=="STOPPED"
        assert session.scalar(select(func.count()).select_from(ModelArtifact))==23
    finally:
        await client.aclose()
        main.app.dependency_overrides.pop(main.get_session, None)
