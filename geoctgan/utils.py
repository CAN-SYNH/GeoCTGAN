from __future__ import annotations
import hashlib
import inspect
import json
import logging
import os
import random
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Iterable, Optional, Sequence
import numpy as np
logger = logging.getLogger(__name__)

def sdv_gpu_constructor_kwargs(synthesizer_class: type, enabled: bool) -> tuple[dict[str, bool], str]:
    if synthesizer_class is None:
        raise ImportError('The requested SDV synthesizer is not installed.')
    try:
        parameters = inspect.signature(synthesizer_class.__init__).parameters
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f'Could not inspect the constructor for {synthesizer_class!r}.') from exc
    for parameter_name in ('enable_gpu', 'cuda'):
        if parameter_name in parameters:
            return ({parameter_name: bool(enabled)}, parameter_name)
    raise RuntimeError(f"{synthesizer_class.__module__}.{synthesizer_class.__name__} exposes neither 'enable_gpu' nor 'cuda'; the installed SDV interface is incompatible.")

def clean_user_path(value: str | Path) -> str:
    text = str(value).strip()
    while len(text) >= 2 and text[0] == text[-1] and (text[0] in {"'", '"'}):
        text = text[1:-1].strip()
    if text.lower().startswith('file://'):
        text = text[7:]
    text = os.path.expandvars(os.path.expanduser(text))
    if os.name == 'nt':
        text = text.replace('/', '\\')
    else:
        text = text.replace('\\', '/')
    return text

def resolve_source_path(base_dir: str | Path, value: str | Path) -> Path:
    cleaned = clean_user_path(value)
    candidate = Path(cleaned)
    if not candidate.is_absolute():
        candidate = Path(clean_user_path(base_dir)) / candidate
    return candidate.expanduser().resolve()

def locate_user_input(value: str | Path, *, expected_names: Sequence[str]=(), allowed_suffixes: Sequence[str]=()) -> Path:
    cleaned = clean_user_path(value)
    candidate = Path(cleaned).expanduser()
    if not candidate.exists():
        raise FileNotFoundError(f'Input path does not exist on this computer: {candidate}. A Windows path cannot be read by a Linux server unless that drive is mounted, and a server path cannot be read by Windows unless it is locally accessible.')
    candidate = candidate.resolve()
    if candidate.is_file():
        return candidate
    names = {name.casefold() for name in expected_names}
    suffixes = {suffix.casefold() for suffix in allowed_suffixes}
    matches = []
    for path in candidate.rglob('*'):
        if not path.is_file():
            continue
        if names and path.name.casefold() in names:
            matches.append(path)
            continue
        if not names and suffixes and (path.suffix.casefold() in suffixes):
            matches.append(path)
    matches = sorted(set(matches))
    if len(matches) == 1:
        return matches[0].resolve()
    if not matches:
        expectation = ', '.join(expected_names or allowed_suffixes)
        raise FileNotFoundError(f'No matching input ({expectation}) was found under {candidate}.')
    raise RuntimeError(f'The supplied directory contains multiple matching inputs. Provide the intended file directly. Candidates: {[str(path) for path in matches[:20]]}')

def ensure_dir(path: str | Path) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    except Exception:
        pass

def derived_seed(base_seed: int, *parts: object) -> int:
    text = '|'.join([str(base_seed), *(str(part) for part in parts)])
    return int(hashlib.sha256(text.encode('utf-8')).hexdigest()[:8], 16)

def _checkpoint_relative_path(value: str | Path, artifact_root: Path) -> tuple[Path, str]:
    path = Path(value).expanduser().resolve()
    try:
        relative = path.relative_to(artifact_root)
    except ValueError as exc:
        raise RuntimeError(f'Checkpoint artifact must remain inside its governed root {artifact_root}: {path}') from exc
    return (path, relative.as_posix())

def checkpoint_artifacts(paths: Iterable[str | Path], *, artifact_root: str | Path) -> list[dict]:
    root = Path(artifact_root).expanduser().resolve()
    values = list(paths)
    if not values:
        raise ValueError('A successful checkpoint must commit at least one artifact.')
    records = []
    seen = set()
    for value in values:
        path, relative = _checkpoint_relative_path(value, root)
        if relative in seen:
            raise ValueError(f'Duplicate checkpoint artifact: {relative}')
        seen.add(relative)
        if not path.is_file():
            raise FileNotFoundError(f'Checkpoint artifact is missing: {path}')
        records.append({'relative_path': relative})
    return records

def load_success_checkpoint(checkpoint_path: str | Path, *, artifact_root: str | Path, required_artifacts: Iterable[str | Path]=()) -> dict | None:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        return None
    try:
        payload = load_output_json(path)
    except (OSError, json.JSONDecodeError):
        return None
    if str(payload.get('status', '')).casefold() != 'success':
        return None
    root = Path(artifact_root).expanduser().resolve()
    artifacts = payload.get('artifacts')
    if not isinstance(artifacts, list) or not artifacts:
        return None
    validated = {}
    for artifact in artifacts:
        relative = str(artifact.get('relative_path', ''))
        if not relative or Path(relative).is_absolute() or relative in validated:
            return None
        artifact_path = (root / Path(relative)).resolve()
        try:
            artifact_path.relative_to(root)
        except ValueError:
            return None
        if not artifact_path.is_file():
            return None
        validated[relative] = str(artifact_path)
    required = {_checkpoint_relative_path(value, root)[1] for value in required_artifacts}
    if not required.issubset(validated):
        return None
    payload['_validated_artifacts'] = validated
    return payload

def write_success_checkpoint(checkpoint_path: str | Path, *, result: dict, artifacts: Iterable[str | Path], artifact_root: str | Path) -> Path:
    path = Path(checkpoint_path).expanduser().resolve()
    atomic_json_dump({'status': 'success', 'result': result, 'artifacts': checkpoint_artifacts(artifacts, artifact_root=artifact_root)}, path)
    return path

def atomic_json_dump(payload, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8') as handle:
        json.dump(output_payload(payload, path), handle, indent=2, default=str)
    temp.replace(path)

def output_run_root(path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    start = target if target.is_dir() else target.parent
    for parent in (start, *start.parents):
        if (parent / 'Metadata').is_dir() and (parent / 'Data').is_dir():
            return parent
    return start

def output_payload(payload, path: str | Path):
    return _output_payload_at_root(payload, output_run_root(path))


def _output_payload_at_root(payload, root: Path):

    def convert(value):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                converted = convert(item)
                if converted is not _OMIT:
                    result[key] = converted
            return result
        if isinstance(value, (list, tuple)):
            return [None if (item := convert(v)) is _OMIT else item for v in value]
        if isinstance(value, Path):
            value = str(value)
        if isinstance(value, str) and (Path(value).is_absolute() or PureWindowsPath(value).is_absolute()):
            try:
                return Path(value).resolve().relative_to(root).as_posix() or '.'
            except (ValueError, OSError):
                return _OMIT
        return value
    cleaned = convert(payload)
    return None if cleaned is _OMIT else cleaned
_OMIT = object()

def resolve_output_path(reference_file: str | Path, value: str | Path) -> Path:
    value = Path(value).expanduser()
    return value.resolve() if value.is_absolute() else (output_run_root(reference_file) / value).resolve()

def load_output_json(path: str | Path):
    path = Path(path)
    payload = json.loads(path.read_text(encoding='utf-8'))
    root = output_run_root(path)

    def restore(value):
        if isinstance(value, dict):
            return {key: item if key == 'relative_path' else restore(item) for key, item in value.items()}
        if isinstance(value, list):
            return [restore(item) for item in value]
        if isinstance(value, str) and (not Path(value).is_absolute()) and ('/' in value or value == '.'):
            candidate = root / value
            if candidate.exists():
                return str(candidate.resolve())
        return value
    return restore(payload)

def write_output_csv(frame, path, *args, **kwargs):
    frame = frame.copy()
    root = output_run_root(path)
    for column in frame.select_dtypes(include=['object', 'string']).columns:
        frame[column] = _output_payload_at_root(frame[column].tolist(), root)
    return frame.to_csv(path, *args, **kwargs)

def atomic_text_dump(text: str, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(text, encoding='utf-8')
    temp.replace(path)

def preserve_prior_file(path: str | Path) -> Optional[Path]:
    path = Path(path)
    if not path.is_file():
        return None
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    archived = path.with_name(f'{path.stem}_prior_{stamp}{path.suffix}')
    counter = 2
    while archived.exists():
        archived = path.with_name(f'{path.stem}_prior_{stamp}_{counter:02d}{path.suffix}')
        counter += 1
    path.replace(archived)
    return archived

def resolve_in_zip(base_dir: str, zip_name: str, inner_ext: Optional[str]=None, contains: Optional[str]=None, exact_member: Optional[str]=None) -> str:
    zip_path = resolve_source_path(base_dir, zip_name)
    if not zip_path.is_file():
        raise FileNotFoundError(zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        names = [name for name in archive.namelist() if not name.endswith('/')]
        if exact_member:
            matches = [name for name in names if name == exact_member]
        else:
            matches = names
            if inner_ext:
                matches = [name for name in matches if name.lower().endswith(inner_ext.lower())]
            if contains:
                matches = [name for name in matches if contains.lower() in name.lower()]
        if not matches:
            raise FileNotFoundError(f'No archive member matched exact_member={exact_member!r}, inner_ext={inner_ext!r}, contains={contains!r} in {zip_path}.')
        if len(matches) > 1:
            raise RuntimeError(f'Ambiguous archive member selection. Configure EVENTS_MEMBER or NETWORK_MEMBER explicitly. Candidates: {matches[:20]}')
        return f'zip://{zip_path}!{matches[0]}'

def setup_logging(log_file=None, level=logging.ERROR):
    logging.basicConfig(level=logging.ERROR, format='%(message)s', force=True)

class StageLedger:

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.exists():
            self.payload = load_output_json(self.path)
            interrupted = False
            for details in self.payload.get('stages', {}).values():
                if details.get('status') == 'RUNNING':
                    details['status'] = 'INTERRUPTED'
                    details['interrupted_utc'] = datetime.now(timezone.utc).isoformat()
                    interrupted = True
            if interrupted:
                atomic_json_dump(self.payload, self.path)
        else:
            self.payload = {'stages': {}}

    def start(self, name: str, **details) -> None:
        self.payload['stages'][name] = {'status': 'RUNNING', 'started_utc': datetime.now(timezone.utc).isoformat(), **details}
        atomic_json_dump(self.payload, self.path)

    def complete(self, name: str, **details) -> None:
        self.payload['stages'][name] = {'status': 'COMPLETE', 'completed_utc': datetime.now(timezone.utc).isoformat(), **details}
        atomic_json_dump(self.payload, self.path)

    def fail(self, name: str, error: str, **details) -> None:
        self.payload['stages'][name] = {'status': 'FAILED', 'failed_utc': datetime.now(timezone.utc).isoformat(), **details}
        atomic_json_dump(self.payload, self.path)

    def is_complete(self, name: str) -> bool:
        return self.payload.get('stages', {}).get(name, {}).get('status') == 'COMPLETE'

    def details(self, name: str) -> dict:
        return dict(self.payload.get('stages', {}).get(name, {}))

def checkpoint_artifact_map(*, artifact_root: str | Path, **paths: str | Path) -> dict:
    records = checkpoint_artifacts(paths.values(), artifact_root=artifact_root)
    return {key: record for key, record in zip(paths.keys(), records)}

def load_validated_stage_artifacts(ledger: StageLedger, stage: str, required: Sequence[str], *, artifact_root: str | Path) -> Optional[dict]:
    if not ledger.is_complete(stage):
        return None
    details = ledger.details(stage)
    artifacts = details.get('artifacts', {})
    root = Path(artifact_root).expanduser().resolve()
    recovered = {}
    for key in required:
        record = artifacts.get(key)
        if not isinstance(record, dict):
            return None
        relative = str(record.get('relative_path', ''))
        if not relative or Path(relative).is_absolute():
            return None
        path = (root / Path(relative)).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            return None
        if not path.is_file():
            return None
        recovered[key] = str(path)
    recovered['details'] = details
    return recovered
