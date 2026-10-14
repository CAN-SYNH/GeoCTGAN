from __future__ import annotations
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
_MONTHS = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')

def formal_timestamp(value: Optional[datetime]=None) -> str:
    value = datetime.now() if value is None else value
    hour = value.strftime('%I').lstrip('0') or '12'
    suffix = value.strftime('%p')
    return f'{value.day:02d}{_MONTHS[value.month - 1]}{value.year % 100:02d}_{hour}.{value.minute:02d}{suffix}'

def create_unique_run_directory(parent: str | Path, prefix: str, *, timestamp: Optional[datetime]=None) -> Path:
    parent = Path(parent).expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    stem = f'{prefix}_{formal_timestamp(timestamp)}'
    candidate = parent / stem
    counter = 2
    while candidate.exists():
        candidate = parent / f'{stem}_{counter:02d}'
        counter += 1
    candidate.mkdir(parents=False, exist_ok=False)
    return candidate

@dataclass(frozen=True)
class ArtifactDirectories:
    root: Path
    data: Path
    intermediate: Path
    models: Path
    figures: Path
    tables: Path
    metadata: Path
    logs: Path

    @classmethod
    def create(cls, root: str | Path) -> 'ArtifactDirectories':
        root = Path(root).expanduser().resolve()
        directories = cls(root=root, data=root / 'Data', intermediate=root / 'Intermediate', models=root / 'Models', figures=root / 'Figures', tables=root / 'Tables', metadata=root / 'Metadata', logs=root / 'Logs')
        for path in (directories.data, directories.intermediate, directories.models, directories.figures, directories.tables, directories.metadata, directories.logs):
            path.mkdir(parents=True, exist_ok=True)
        return directories

    def seed(self, seed: int) -> 'ArtifactDirectories':
        label = f'Seed_{int(seed)}'
        directories = ArtifactDirectories(root=self.root, data=self.data / label, intermediate=self.intermediate / label, models=self.models / label, figures=self.figures / label, tables=self.tables / label, metadata=self.metadata / label, logs=self.logs / label)
        for path in (directories.data, directories.intermediate, directories.models, directories.figures, directories.tables, directories.metadata, directories.logs):
            path.mkdir(parents=True, exist_ok=True)
        return directories
