from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from geoctgan.artifacts import (
    ArtifactDirectories,
    create_unique_run_directory,
    formal_timestamp,
)
from geoctgan.evaluation import EvaluationConfig, _validate_dataset
from geoctgan.utils import clean_user_path, locate_user_input


def test_formal_timestamp_and_unique_run_directory(tmp_path: Path) -> None:
    moment = datetime(2026, 7, 26, 22, 25)
    assert formal_timestamp(moment) == "26Jul26_10.25PM"
    first = create_unique_run_directory(
        tmp_path, "GeoCTGAN", timestamp=moment
    )
    second = create_unique_run_directory(
        tmp_path, "GeoCTGAN", timestamp=moment
    )
    assert first.name == "GeoCTGAN_26Jul26_10.25PM"
    assert second.name == "GeoCTGAN_26Jul26_10.25PM_02"


def test_artifact_directories_are_created(tmp_path: Path) -> None:
    artifacts = ArtifactDirectories.create(tmp_path / "run")
    assert all(
        path.is_dir()
        for path in (
            artifacts.data,
            artifacts.intermediate,
            artifacts.models,
            artifacts.figures,
            artifacts.tables,
            artifacts.metadata,
            artifacts.logs,
        )
    )


def test_locate_user_input_accepts_quotes_and_containing_directory(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "MCI_OPEN_DATA.zip"
    archive.write_bytes(b"test")
    assert clean_user_path(f'"{archive}"') == str(archive)
    assert locate_user_input(
        f'"{tmp_path}"', expected_names=("MCI_OPEN_DATA.zip",)
    ) == archive.resolve()


def test_evaluation_rejects_empty_dataset() -> None:
    with pytest.raises(RuntimeError, match="empty"):
        _validate_dataset("empty", pd.DataFrame(), EvaluationConfig())

