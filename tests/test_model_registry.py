import json

import pytest

from app.model_registry import activate_candidate, register_candidate, select_shadow_candidate
from app.models import ModelArtifact
from app.strategy import FEATURE_SCHEMA, FEATURE_SCHEMA_V4


class _LoadableModel:
    def __init__(self, _path):
        self.available = True


def test_candidate_requires_explicit_activation_and_is_copied_atomically(session, tmp_path, monkeypatch):
    candidate = tmp_path / "candidates" / "candidate.json"
    candidate.parent.mkdir()
    candidate.write_text("model-bytes")
    metadata = {
        "version": "regime-test",
        "feature_schema": FEATURE_SCHEMA,
        "activation_eligible": True,
    }
    candidate.with_name("candidate.metadata.json").write_text(json.dumps(metadata))

    artifact = register_candidate(session, candidate, metadata)
    assert artifact.status == "CANDIDATE"

    monkeypatch.setattr("app.model_registry.RegimeModel", _LoadableModel)
    active = tmp_path / "regime.json"
    activate_candidate(session, artifact.id, active)

    session.refresh(artifact)
    assert artifact.status == "ACTIVE"
    assert active.read_text() == "model-bytes"
    assert json.loads(active.with_name("regime.metadata.json").read_text())["version"] == "regime-test"
    assert session.query(ModelArtifact).count() == 1


def test_candidate_that_fails_validation_cannot_be_activated(session, tmp_path):
    candidate = tmp_path / "blocked.json"
    candidate.write_text("model-bytes")
    metadata = {
        "version": "regime-blocked",
        "feature_schema": FEATURE_SCHEMA,
        "activation_eligible": False,
        "activation_blockers": ["no qualified directional predictions"],
    }
    candidate.with_name("blocked.metadata.json").write_text(json.dumps(metadata))
    artifact = register_candidate(session, candidate, metadata)

    with pytest.raises(ValueError, match="no qualified directional predictions"):
        activate_candidate(session, artifact.id, tmp_path / "active.json")


def test_blocked_candidate_can_be_selected_for_shadow_without_activation(session, tmp_path, monkeypatch):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    metadata = {
        "feature_schema": FEATURE_SCHEMA,
        "activation_eligible": False,
        "activation_blockers": ["needs forward validation"],
    }
    for path, version in ((first, "shadow-first"), (second, "shadow-second")):
        path.write_text("model-bytes")
        candidate_metadata = {**metadata, "version": version}
        path.with_name(f"{path.stem}.metadata.json").write_text(json.dumps(candidate_metadata))
        register_candidate(session, path, candidate_metadata)
    monkeypatch.setattr("app.model_registry.RegimeModel", _LoadableModel)

    first_artifact = session.query(ModelArtifact).filter_by(version="shadow-first").one()
    second_artifact = session.query(ModelArtifact).filter_by(version="shadow-second").one()
    select_shadow_candidate(session, first_artifact.id)
    assert first_artifact.status == "SHADOW"
    assert not first_artifact.activated_at

    select_shadow_candidate(session, second_artifact.id)
    session.refresh(first_artifact)
    assert first_artifact.status == "CANDIDATE"
    assert second_artifact.status == "SHADOW"


def test_v4_activation_copies_every_bundle_file(session, tmp_path, monkeypatch):
    candidate = tmp_path / "candidates" / "candidate.json"
    candidate.parent.mkdir()
    for name, content in (
        ("candidate.ce.cbm", "ce"),
        ("candidate.pe.cbm", "pe"),
        ("candidate.ranker.json", "ranker"),
    ):
        (candidate.parent / name).write_text(content)
    candidate.write_text(
        json.dumps(
            {
                "quality_models": {
                    "CE": "candidate.ce.cbm",
                    "PE": "candidate.pe.cbm",
                },
                "ranker": "candidate.ranker.json",
            }
        )
    )
    metadata = {
        "version": "quality-test",
        "feature_schema": FEATURE_SCHEMA_V4,
        "activation_eligible": True,
        "bundle_files": [
            "candidate.ce.cbm",
            "candidate.pe.cbm",
            "candidate.ranker.json",
        ],
    }
    candidate.with_name("candidate.metadata.json").write_text(json.dumps(metadata))
    artifact = register_candidate(session, candidate, metadata)
    monkeypatch.setattr("app.model_registry.RegimeModel", _LoadableModel)

    active = tmp_path / "regime.json"
    activate_candidate(session, artifact.id, active)

    manifest = json.loads(active.read_text())
    assert (tmp_path / manifest["quality_models"]["CE"]).read_text() == "ce"
    assert (tmp_path / manifest["quality_models"]["PE"]).read_text() == "pe"
    assert (tmp_path / manifest["ranker"]).read_text() == "ranker"
