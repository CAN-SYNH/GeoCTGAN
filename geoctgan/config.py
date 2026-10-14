from __future__ import annotations
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from .utils import atomic_json_dump, load_output_json
SELECTION_LOCK_FIELDS = ('SNAP_MAX_M', 'EMB_DIM', 'N_WALKS', 'WALK_LEN', 'COHORT_KIND')
DEFINITIVE_OVERRIDE_FIELDS = {'BASE_DIR', 'OUTPUT_DIR', 'ZIP_EVENTS', 'ZIP_NETWORK', 'EVENTS_MEMBER', 'NETWORK_MEMBER', 'NETWORK_LAYER', 'SELECTED_CONFIG_PATH', 'REQUIRE_SELECTED_CONFIG', 'EPOCHS', 'ENABLE_GPU', 'DEFINITIVE_SEEDS', 'BENCHMARK_SEEDS', 'BENCHMARK_METHODS', 'SEED', 'RESUME_TRAINING', 'RESUME_GENERATION', 'RESUME_CANN', 'MAX_MODEL_CACHE', 'CANN_GROUP_TIME_LIMIT_S', 'CANN_MAX_EDGES', 'CANN_REQUIRE_ORTOOLS', 'CANN_PREFLIGHT_LARGEST_GROUPS', 'PUBLIC_RELEASE_ENABLED', 'PUBLIC_RELEASE_CAPACITY_MODE', 'ALLOW_EXACT_QUOTA_PUBLIC_RELEASE'}
PATH_CONFIG_FIELDS = {'BASE_DIR', 'OUTPUT_DIR', 'ZIP_EVENTS', 'ZIP_NETWORK', 'SELECTED_CONFIG_PATH', 'OUTER_SPLIT_MANIFEST_PATH'}
OPERATIONAL_RECOVERY_FIELDS = {'RESUME_TRAINING', 'RESUME_GENERATION', 'RESUME_CANN', 'MAX_MODEL_CACHE', 'CANN_GROUP_TIME_LIMIT_S', 'CANN_PREFLIGHT_LARGEST_GROUPS'}

def scientific_config_payload(config: 'Config', *, exclude_paths: bool=True) -> Dict[str, Any]:
    excluded = set(OPERATIONAL_RECOVERY_FIELDS)
    if exclude_paths:
        excluded.update(PATH_CONFIG_FIELDS)
    return {key: value for key, value in config.as_dict().items() if key not in excluded}

@dataclass
class Config:
    BASE_DIR: str = 'data'
    OUTPUT_DIR: Optional[str] = None
    ZIP_EVENTS: str = 'MCI_OPEN_DATA.zip'
    ZIP_NETWORK: str = 'RoadNetworkExport.zip'
    EVENTS_MEMBER: Optional[str] = 'MCI_OPEN_DATA.shp'
    NETWORK_MEMBER: Optional[str] = 'RoadNetworkExport/toronto_network.gpkg'
    NETWORK_LAYER: Optional[str] = 'roads_edges'
    START_YEAR: int = 2014
    END_YEAR: int = 2024
    CRS_GEO: str = 'EPSG:4326'
    CRS_METRIC: str = 'EPSG:32617'
    COL_ID: str = 'EVENT_PK'
    COL_DATE: str = 'OCC_DATE'
    COL_LON: str = 'LONG_WGS84'
    COL_LAT: str = 'LAT_WGS84'
    COL_SEG: str = 'segment_id'
    COL_CAT: str = 'MCI_CATEGO'
    COL_OFFENCE: str = 'OFFENCE'
    MAX_SEG_LEN_M: float = 40.0
    SNAP_MAX_M: float = 50.0
    NODE_SNAP_TOL_M: float = 0.75
    REMOVE_REVERSE_DUPLICATES: bool = True
    REQUIRE_AUTHORITATIVE_TOPOLOGY: bool = True
    MAX_ALLOWED_ISOLATED_SEGMENTS: int = 0
    SNAP_CANDIDATES_M: List[float] = field(default_factory=lambda: [10.0, 20.0, 30.0, 50.0])
    EMB_DIM: int = 16
    WALK_LEN: int = 10
    N_WALKS: int = 15
    W2V_WINDOW: int = 10
    W2V_EPOCHS: int = 6
    W2V_WORKERS: int = 1
    EMB_DIM_CANDIDATES: List[int] = field(default_factory=lambda: [8, 16, 32])
    N_WALKS_CANDIDATES: List[int] = field(default_factory=lambda: [5, 15, 30])
    WALK_LEN_CANDIDATES: List[int] = field(default_factory=lambda: [10, 20, 40])
    SENSITIVITY_REFERENCE_WALK_LEN: int = 20
    SENSITIVITY_GRAPH_NODE_LIMIT: int = 80000
    SENSITIVITY_PAIR_SAMPLE: int = 5000
    SENSITIVITY_NEIGHBOR_K: int = 10
    SENSITIVITY_SCREEN_SEED: int = 42
    SENSITIVITY_SEEDS: List[int] = field(default_factory=lambda: [42, 52, 62])
    EMBEDDING_FINALIST_COUNT: int = 3
    EMBEDDING_PILOT_SCHEME: str = 'year'
    COHORT_KIND: str = 'global'
    COHORT_CANDIDATES: List[str] = field(default_factory=lambda: ['year_month', 'year_quarter', 'year', 'global'])
    TEMPORAL_FINALIST_COUNT: int = 4
    TEMPORAL_PILOT_EPOCHS: int = 100
    TEMPORAL_PILOT_YEARS: List[int] = field(default_factory=lambda: [2019, 2020, 2022, 2024])
    TEMPORAL_PILOT_MAX_ROWS_PER_CONDITION: int = 500
    TEMPORAL_BOUNDARY_WINDOW_DAYS: int = 7
    EPOCHS: int = 300
    ENABLE_GPU: bool = True
    PAC: int = 10
    BATCH_TARGET: int = 512
    GEN_LR: float = 0.0001
    DISC_LR: float = 0.0001
    ENFORCE_ROUNDING: bool = True
    MIN_POOL_ROWS: int = 50
    MAX_WHOLE_ROW_RESAMPLE_ROUNDS: int = 60
    MIN_CONDITIONAL_REQUEST: int = 64
    MAX_CONDITIONAL_REQUEST: int = 5000
    CONDITIONAL_REQUEST_SAFETY_FACTOR: float = 1.5
    MAX_TOTAL_SAMPLE_MULTIPLIER: int = 120
    RESUME_TRAINING: bool = True
    RESUME_GENERATION: bool = True
    MAX_MODEL_CACHE: int = 1
    SAMPLE_FACTOR: float = 1.0
    CAPACITY_MODE: str = 'exact_quota'
    K_CANDIDATE_SCHEDULE: List[int] = field(default_factory=lambda: [16, 32, 64, 128, 256])
    CANN_COST_SCALE: int = 100000
    CANN_OBJECTIVE_TOLERANCE: float = 0.0001
    CANN_STABLE_ROUNDS: int = 1
    CANN_FULL_SUPPORT_THRESHOLD: int = 400
    CANN_GROUP_TIME_LIMIT_S: int = 600
    CANN_MAX_EDGES: int = 5000000
    CANN_REQUIRE_ORTOOLS: bool = True
    CANN_PREFLIGHT_LARGEST_GROUPS: int = 3
    RESUME_CANN: bool = True
    UPPER_QUOTA_SLACK: float = 0.25
    MIN_REAL_SEGMENT_SUPPORT: int = 10
    DCR_THRESH_M: float = 5.0
    PUBLIC_RELEASE_ENABLED: bool = True
    PUBLIC_RELEASE_CAPACITY_MODE: str = 'upper_quota'
    ALLOW_EXACT_QUOTA_PUBLIC_RELEASE: bool = False
    PUBLIC_COLUMNS: List[str] = field(default_factory=lambda: ['SYNTHETIC_ID', 'OCC_YEAR', 'OCC_MONTH', 'OCC_DAY', 'OCC_DOW', 'OCC_HOUR', 'MCI_CATEGO', 'OFFENCE', 'PREMISES_T', 'LOCATION_T', 'LONG_WGS84', 'LAT_WGS84'])
    TEST_SIZE: float = 0.2
    INNER_VALIDATION_SIZE: float = 0.2
    SEED: int = 42
    DEFINITIVE_SEEDS: List[int] = field(default_factory=lambda: [42, 52, 62])
    OUTER_SPLIT_MANIFEST_PATH: Optional[str] = None
    REQUIRE_REUSED_OUTER_SPLIT: bool = True
    REQUIRE_SELECTED_CONFIG: bool = True
    SELECTED_CONFIG_PATH: Optional[str] = None
    RUNTIME_SCORE_WEIGHT: float = 0.1
    BENCHMARK_METHODS: List[str] = field(default_factory=lambda: ['GaussianCopula', 'CTGAN_RawLatLon', 'KDE_AttributeBootstrap', 'RoadSupportBootstrap'])
    BENCHMARK_SEEDS: List[int] = field(default_factory=lambda: [42, 52, 62])

    @classmethod
    def from_json(cls, path: str) -> 'Config':
        data = load_output_json(path)
        return cls(**data)

    def update_from_json(self, path: str) -> 'Config':
        with open(path, 'r', encoding='utf-8') as handle:
            data = json.load(handle)
        merged = asdict(self)
        merged.update(data)
        return Config(**merged)

    def save(self, path: str) -> None:
        atomic_json_dump(asdict(self), path)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_env(cls) -> 'Config':
        default = cls()
        alias_mapping = {'BASE_DIR': ('MCI_BASE_DIR', str), 'OUTPUT_DIR': ('MCI_OUTPUT_DIR', str), 'ZIP_EVENTS': ('MCI_ZIP_EVENTS', str), 'ZIP_NETWORK': ('MCI_ZIP_NETWORK', str), 'START_YEAR': ('MCI_START_YEAR', int), 'END_YEAR': ('MCI_END_YEAR', int), 'EPOCHS': ('MCI_EPOCHS', int), 'GEN_LR': ('MCI_GEN_LR', float), 'DISC_LR': ('MCI_DISC_LR', float), 'TEST_SIZE': ('MCI_TEST_SIZE', float), 'SEED': ('MCI_SEED', int)}
        values = asdict(default)
        for field_name, (env_name, caster) in alias_mapping.items():
            raw = os.getenv(env_name)
            if raw is None:
                continue
            try:
                values[field_name] = caster(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f'Invalid value for environment variable {env_name}: {raw!r}') from exc
        for field_name, current in list(values.items()):
            env_name = f'GEOCTGAN_{field_name}'
            raw = os.getenv(env_name)
            if raw is None:
                continue
            try:
                if isinstance(current, bool):
                    normalized = raw.strip().casefold()
                    if normalized not in {'true', 'false', 'yes', 'no', '1', '0'}:
                        raise ValueError('expected true/false, yes/no or 1/0')
                    values[field_name] = normalized in {'true', 'yes', '1'}
                elif isinstance(current, list):
                    try:
                        decoded = json.loads(raw)
                    except json.JSONDecodeError:
                        decoded = [item.strip() for item in raw.split(',') if item.strip()]
                    if not isinstance(decoded, list):
                        raise ValueError('expected a JSON array or comma-separated list')
                    if current:
                        item_type = type(current[0])
                        decoded = [item_type(item) for item in decoded]
                    values[field_name] = decoded
                elif isinstance(current, int):
                    values[field_name] = int(raw)
                elif isinstance(current, float):
                    values[field_name] = float(raw)
                elif isinstance(current, str):
                    values[field_name] = raw
                elif current is None:
                    try:
                        values[field_name] = json.loads(raw)
                    except json.JSONDecodeError:
                        values[field_name] = raw
                else:
                    values[field_name] = json.loads(raw)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f'Invalid value for environment variable {env_name}: {raw!r}') from exc
        return cls(**values)

    def validate(self) -> None:
        if self.COHORT_KIND not in {'year_month', 'year_quarter', 'year', 'global'}:
            raise ValueError(f'Unsupported COHORT_KIND={self.COHORT_KIND!r}')
        if self.CAPACITY_MODE not in {'exact_quota', 'upper_quota'}:
            raise ValueError(f'Unsupported CAPACITY_MODE={self.CAPACITY_MODE!r}')
        if self.PUBLIC_RELEASE_CAPACITY_MODE not in {'exact_quota', 'upper_quota'}:
            raise ValueError(f'Unsupported PUBLIC_RELEASE_CAPACITY_MODE={self.PUBLIC_RELEASE_CAPACITY_MODE!r}')
        if self.PUBLIC_RELEASE_ENABLED and self.PUBLIC_RELEASE_CAPACITY_MODE == 'exact_quota' and (not self.ALLOW_EXACT_QUOTA_PUBLIC_RELEASE):
            raise ValueError('Exact-quota public release is blocked by default because it can disclose fine-grained training segment margins. Set ALLOW_EXACT_QUOTA_PUBLIC_RELEASE=true only after an explicit disclosure-risk assessment.')
        if self.SAMPLE_FACTOR <= 0:
            raise ValueError('SAMPLE_FACTOR must be positive.')
        if not self.K_CANDIDATE_SCHEDULE:
            raise ValueError('K_CANDIDATE_SCHEDULE cannot be empty.')
        if sorted(set(self.K_CANDIDATE_SCHEDULE)) != self.K_CANDIDATE_SCHEDULE:
            raise ValueError('K_CANDIDATE_SCHEDULE must be strictly increasing.')
        if self.PAC < 1:
            raise ValueError('PAC must be at least one.')
        if self.CANN_MAX_EDGES < 1:
            raise ValueError('CANN_MAX_EDGES must be positive.')
        if self.CANN_GROUP_TIME_LIMIT_S < 1:
            raise ValueError('CANN_GROUP_TIME_LIMIT_S must be positive.')
        if self.CANN_PREFLIGHT_LARGEST_GROUPS < 1:
            raise ValueError('CANN_PREFLIGHT_LARGEST_GROUPS must be positive.')
        if self.REQUIRE_AUTHORITATIVE_TOPOLOGY and (not self.NETWORK_LAYER):
            raise ValueError("NETWORK_LAYER must be explicit when authoritative topology is required. For the Toronto export use 'roads_edges'.")
        sensitivity_seeds = list(dict.fromkeys(map(int, self.SENSITIVITY_SEEDS)))
        if len(sensitivity_seeds) < 3:
            raise ValueError('Finalist confirmation requires at least three distinct SENSITIVITY_SEEDS.')
        if int(self.SENSITIVITY_SCREEN_SEED) not in sensitivity_seeds:
            raise ValueError('SENSITIVITY_SCREEN_SEED must also appear in SENSITIVITY_SEEDS so its screening result can be reused during confirmation.')
        if not 1 <= int(self.EMBEDDING_FINALIST_COUNT):
            raise ValueError('EMBEDDING_FINALIST_COUNT must be positive.')
        if not 1 <= int(self.TEMPORAL_FINALIST_COUNT) <= len(set(self.COHORT_CANDIDATES)):
            raise ValueError('TEMPORAL_FINALIST_COUNT must be between one and the number of distinct COHORT_CANDIDATES.')
        if not self.DEFINITIVE_SEEDS:
            raise ValueError('DEFINITIVE_SEEDS cannot be empty.')
        if not self.BENCHMARK_SEEDS:
            raise ValueError('BENCHMARK_SEEDS cannot be empty.')
        if not 0 < self.INNER_VALIDATION_SIZE < 0.5:
            raise ValueError('INNER_VALIDATION_SIZE must lie between 0 and 0.5.')
        if self.TEMPORAL_PILOT_MAX_ROWS_PER_CONDITION < 1:
            raise ValueError('TEMPORAL_PILOT_MAX_ROWS_PER_CONDITION must be positive.')
        if self.MAX_MODEL_CACHE < 1:
            raise ValueError('MAX_MODEL_CACHE must be at least one.')
