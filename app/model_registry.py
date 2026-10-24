from __future__ import annotations

import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import ModelArtifact
from .strategy import FEATURE_SCHEMA_V4, FEATURE_SCHEMA_V5, FEATURE_SCHEMA_V6, SUPPORTED_FEATURE_SCHEMAS, RegimeModel
from .training import load_model_metadata, model_metadata_path


def _bundle_paths(model_path: Path, metadata: dict) -> list[Path]:
    if metadata.get("feature_schema") in {FEATURE_SCHEMA_V4, FEATURE_SCHEMA_V5, FEATURE_SCHEMA_V6} and model_path.exists():
        try:
            manifest = json.loads(model_path.read_text())
            names = list(manifest.get("quality_models", {}).values())
            if manifest.get("ranker"):
                names.append(manifest["ranker"])
            return [model_path.parent / name for name in names]
        except (OSError, json.JSONDecodeError):
            pass
    return [model_path.parent / name for name in metadata.get("bundle_files", [])]


def _remove_model_files(model_path: Path, metadata_path: Path) -> None:
    metadata = load_model_metadata(model_path)
    model_path.unlink(missing_ok=True)
    metadata_path.unlink(missing_ok=True)
    for bundle_path in _bundle_paths(model_path, metadata):
        bundle_path.unlink(missing_ok=True)


def _copy_v4_bundle(source: Path, destination: Path, metadata: dict) -> dict:
    """Copy a V4 bundle and return a manifest valid beside destination."""
    manifest = json.loads(source.read_text())
    copied: dict[str, object] = {"quality_models": {}}
    for direction in ("CE", "PE"):
        source_name = manifest["quality_models"][direction]
        suffix = direction.lower()
        destination_name = f"{destination.stem}.{suffix}.cbm"
        shutil.copy2(source.parent / source_name, destination.parent / destination_name)
        copied["quality_models"][direction] = destination_name
    ranker_name = f"{destination.stem}.ranker.json"
    shutil.copy2(source.parent / manifest["ranker"], destination.parent / ranker_name)
    copied["ranker"] = ranker_name
    return copied


def register_candidate(session: Session, model_path: Path, metadata: dict) -> ModelArtifact:
    artifact = ModelArtifact(
        version=str(metadata["version"]),
        status="CANDIDATE",
        model_path=str(model_path),
        metadata_path=str(model_metadata_path(model_path)),
        metadata_json=json.dumps(metadata, sort_keys=True),
    )
    session.add(artifact)
    session.commit()
    candidates = session.scalars(
        select(ModelArtifact)
        .where(ModelArtifact.status == "CANDIDATE")
        .order_by(ModelArtifact.created_at.desc())
    ).all()
    for obsolete in candidates[5:]:
        obsolete.status = "PRUNED"
        _remove_model_files(Path(obsolete.model_path), Path(obsolete.metadata_path))
    session.commit()
    return artifact


def select_shadow_candidate(session: Session, artifact_id: int) -> ModelArtifact:
    """Select a loadable candidate for forward simulation without activating it."""
    artifact = session.get(ModelArtifact, artifact_id)
    if artifact is None or artifact.status in {"ACTIVE", "ARCHIVED", "PRUNED"}:
        raise ValueError("A selectable model candidate was not found")
    source = Path(artifact.model_path)
    metadata = load_model_metadata(source)
    if (
        not source.exists()
        or not Path(artifact.metadata_path).exists()
        or metadata.get("feature_schema") not in SUPPORTED_FEATURE_SCHEMAS
    ):
        raise ValueError("Candidate files are missing or use an incompatible feature schema")
    if not RegimeModel(source).available:
        raise ValueError("Candidate model cannot be loaded")

    for existing in session.scalars(
        select(ModelArtifact).where(
            ModelArtifact.status == "SHADOW",
            ModelArtifact.id != artifact.id,
        )
    ):
        existing.status = "CANDIDATE"
    artifact.status = "SHADOW"
    session.commit()
    return artifact


def activate_candidate(session: Session, artifact_id: int, active_path: Path) -> ModelArtifact:
    artifact = session.get(ModelArtifact, artifact_id)
    if artifact is None:
        raise ValueError("Model candidate was not found")
    source = Path(artifact.model_path)
    source_metadata = Path(artifact.metadata_path)
    metadata = load_model_metadata(source)
    if not source.exists() or not source_metadata.exists() or metadata.get("feature_schema") not in SUPPORTED_FEATURE_SCHEMAS:
        raise ValueError("Candidate files are missing or use an incompatible feature schema")
    if not metadata.get("activation_eligible", False):
        detail = "; ".join(metadata.get("activation_blockers") or [])
        raise ValueError(
            "Candidate did not pass activation validation gates"
            + (f": {detail}" if detail else "")
        )
    if not RegimeModel(source).available:
        raise ValueError("Candidate model cannot be loaded")

    active_path.parent.mkdir(parents=True, exist_ok=True)
    archive = active_path.parent / "archive"
    archive.mkdir(parents=True, exist_ok=True)
    current = load_model_metadata(active_path)
    if active_path.exists() and current.get("version"):
        archived_model = archive / f"{current['version']}.json"
        archived_metadata = archive / f"{current['version']}.metadata.json"
        if current.get("feature_schema") in {FEATURE_SCHEMA_V4, FEATURE_SCHEMA_V5, FEATURE_SCHEMA_V6}:
            archived_manifest = _copy_v4_bundle(active_path, archived_model, current)
            archived_model.write_text(json.dumps(archived_manifest, indent=2, sort_keys=True))
        else:
            shutil.copy2(active_path, archived_model)
        shutil.copy2(model_metadata_path(active_path), archived_metadata)

    temporary_model = active_path.with_name(f".{active_path.name}.activation")
    temporary_metadata = model_metadata_path(active_path).with_name(
        f".{model_metadata_path(active_path).name}.activation"
    )
    if metadata.get("feature_schema") in {FEATURE_SCHEMA_V4, FEATURE_SCHEMA_V5, FEATURE_SCHEMA_V6}:
        active_manifest = _copy_v4_bundle(source, active_path, metadata)
        temporary_model.write_text(json.dumps(active_manifest, indent=2, sort_keys=True))
    else:
        shutil.copy2(source, temporary_model)
    shutil.copy2(source_metadata, temporary_metadata)
    os.replace(temporary_model, active_path)
    os.replace(temporary_metadata, model_metadata_path(active_path))

    for existing in session.scalars(select(ModelArtifact).where(ModelArtifact.status == "ACTIVE")):
        existing.status = "ARCHIVED"
    artifact.status = "ACTIVE"
    artifact.activated_at = datetime.now(UTC)
    session.commit()

    archived = session.scalars(
        select(ModelArtifact)
        .where(ModelArtifact.status == "ARCHIVED")
        .order_by(ModelArtifact.activated_at.desc(), ModelArtifact.created_at.desc())
    ).all()
    for obsolete in archived[5:]:
        obsolete.status = "PRUNED"
        _remove_model_files(Path(obsolete.model_path), Path(obsolete.metadata_path))
        _remove_model_files(
            archive / f"{obsolete.version}.json",
            archive / f"{obsolete.version}.metadata.json",
        )
    session.commit()
    return artifact
