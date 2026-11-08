from app.database import DEFAULTS
from app.dashboard import dashboard_context
from app.models import Setting
from html.parser import HTMLParser
from types import SimpleNamespace

from app import dashboard
from starlette.requests import Request


def test_new_install_enables_only_nifty(session):
    assert DEFAULTS["enabled_indices"] == "NIFTY"
    assert session.get(Setting, "enabled_indices").value == "NIFTY"
    assert dashboard_context(session)["enabled_indices"] == {"NIFTY"}


def test_rendered_workflow_forms_default_to_nifty_only(session, tmp_path, monkeypatch):
    from app.main import templates

    class IndexFields(HTMLParser):
        def __init__(self):
            super().__init__()
            self.action = None
            self.forms = {}

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == "form":
                self.action = attrs.get("action")
                self.forms[self.action] = []
            if tag == "input" and attrs.get("name") == "indices":
                self.forms[self.action].append((attrs["value"], "checked" in attrs))

        def handle_endtag(self, tag):
            if tag == "form":
                self.action = None

    monkeypatch.setattr(dashboard, "config", SimpleNamespace(root=tmp_path, model_path=tmp_path/"none.json"))
    monkeypatch.setattr(dashboard, "research_readiness", lambda session: [])
    context = dashboard.backtests_context(session)
    request = Request({"type": "http", "method": "GET", "scheme": "http", "server": ("testserver", 80),
                       "path": "/backtests", "query_string": b"", "headers": []})
    html = templates.env.get_template("backtests.html").render(**context, request=request,
        url_for=lambda *args, **kwargs: "/static/app.css", static_asset_version=lambda _: "1")
    parser = IndexFields()
    parser.feed(html)
    for action in ("/backtests/sync-data", "/backtests/train", "/backtests"):
        assert parser.forms[action] == [("NIFTY", True), ("BANK_NIFTY", False), ("SENSEX", False)]


def test_backtests_reuses_display_only_research_statistics(session, tmp_path, monkeypatch):
    calls = 0

    def readiness(_session):
        nonlocal calls
        calls += 1
        return []

    dashboard.clear_portal_caches()
    monkeypatch.setattr(dashboard, "config", SimpleNamespace(root=tmp_path, model_path=tmp_path / "none.json"))
    monkeypatch.setattr(dashboard, "research_readiness", readiness)

    dashboard.backtests_context(session)
    dashboard.backtests_context(session)

    assert calls == 1
