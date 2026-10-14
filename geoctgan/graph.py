from __future__ import annotations
from .utils import write_output_csv
import logging
import hashlib
import pickle
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple
import geopandas as gpd
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from .config import Config
from .utils import atomic_json_dump, derived_seed, load_success_checkpoint, write_success_checkpoint
logger = logging.getLogger(__name__)

def _stable_token_hash(value: object) -> int:
    digest = hashlib.blake2b(str(value).encode('utf-8'), digest_size=8, person=b'GeoCTGAN').digest()
    return int.from_bytes(digest, byteorder='little', signed=False)

@dataclass
class GraphArtifacts:
    segment_ids: List[str]
    adjacency: List[np.ndarray]

class DeepWalkCorpus:

    def __init__(self, segment_ids: Sequence[str], adjacency: Sequence[np.ndarray], n_walks: int, walk_length: int, seed: int) -> None:
        self.segment_ids = list(segment_ids)
        self.adjacency = adjacency
        self.n_walks = int(n_walks)
        self.walk_length = int(walk_length)
        self.seed = int(seed)

    def __iter__(self) -> Iterator[List[str]]:
        nodes = np.arange(len(self.segment_ids), dtype=int)
        for pass_index in range(self.n_walks):
            rng = np.random.default_rng(derived_seed(self.seed, 'walk', pass_index))
            order = nodes.copy()
            rng.shuffle(order)
            for start in order:
                current = int(start)
                walk = [self.segment_ids[current]]
                for _ in range(self.walk_length - 1):
                    neighbours = self.adjacency[current]
                    if len(neighbours) == 0:
                        break
                    current = int(rng.choice(neighbours))
                    walk.append(self.segment_ids[current])
                yield walk

def _cluster_endpoints_true_distance(starts: np.ndarray, ends: np.ndarray, tolerance: float) -> Tuple[np.ndarray, np.ndarray, dict]:
    points = np.vstack([starts, ends])
    tree = cKDTree(points)
    pairs = tree.query_pairs(r=float(tolerance), output_type='ndarray')
    parent = np.arange(len(points), dtype=int)

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = int(parent[item])
        return item

    def union(left: int, right: int) -> None:
        root_left, root_right = (find(left), find(right))
        if root_left != root_right:
            parent[root_right] = root_left
    for left, right in pairs:
        union(int(left), int(right))
    roots = np.array([find(index) for index in range(len(points))], dtype=int)
    unique_roots, labels = np.unique(roots, return_inverse=True)
    clusters = defaultdict(list)
    for index, label in enumerate(labels):
        clusters[int(label)].append(index)
    maximum_diameter = 0.0
    for members in clusters.values():
        if len(members) < 2:
            continue
        coordinates = points[members]
        extent = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=2).max()
        maximum_diameter = max(maximum_diameter, float(extent))
    n_segments = len(starts)
    diagnostics = {'cluster_count': int(len(unique_roots)), 'pair_count_within_tolerance': int(len(pairs)), 'maximum_cluster_diameter_m': float(maximum_diameter)}
    return (labels[:n_segments], labels[n_segments:], diagnostics)

def build_graph(cfg: Config, segments_path: str, out_dir: str) -> str:
    output_dir = Path(out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    segments = gpd.read_parquet(segments_path).to_crs(cfg.CRS_METRIC).reset_index(drop=True)
    required = {cfg.COL_SEG, 'source_edge_uid', 'child_index', 'child_count', 'geometry'}
    missing = sorted(required.difference(segments.columns))
    if missing:
        raise KeyError(f'Segment table is missing graph-construction columns: {missing}')
    segment_ids = segments[cfg.COL_SEG].astype(str).tolist()
    n = len(segments)
    adjacency = [set() for _ in range(n)]
    source_groups = segments.groupby('source_edge_uid', sort=False).groups
    for source_edge_uid, indices in source_groups.items():
        subset = segments.loc[list(indices)].sort_values('child_index')
        child_indices = subset['child_index'].astype(int).to_numpy()
        if len(np.unique(child_indices)) != len(child_indices):
            raise RuntimeError(f'Duplicate child_index values for source_edge_uid={source_edge_uid}')
        ordered = subset.index.to_list()
        for left, right in zip(ordered[:-1], ordered[1:]):
            adjacency[left].add(right)
            adjacency[right].add(left)
    authoritative = 'source_u' in segments.columns and 'source_v' in segments.columns and segments['source_u'].notna().any() and segments['source_v'].notna().any()
    if cfg.REQUIRE_AUTHORITATIVE_TOPOLOGY and (not authoritative):
        raise RuntimeError('Authoritative source u/v topology was required but is unavailable.')
    fallback_diagnostics = None
    if authoritative:
        node_to_segments = defaultdict(set)
        for index, row in segments.iterrows():
            if int(row['child_index']) == 0 and pd.notna(row.get('source_u')):
                node_to_segments[f"N_{row.get('source_u')}"].add(int(index))
            if int(row['child_index']) == int(row['child_count']) - 1 and pd.notna(row.get('source_v')):
                node_to_segments[f"N_{row.get('source_v')}"].add(int(index))
    else:
        starts = np.array([[line.coords[0][0], line.coords[0][1]] for line in segments.geometry])
        ends = np.array([[line.coords[-1][0], line.coords[-1][1]] for line in segments.geometry])
        start_labels, end_labels, fallback_diagnostics = _cluster_endpoints_true_distance(starts, ends, cfg.NODE_SNAP_TOL_M)
        node_to_segments = defaultdict(set)
        for index, (start_label, end_label) in enumerate(zip(start_labels, end_labels)):
            node_to_segments[f'F_{int(start_label)}'].add(index)
            node_to_segments[f'F_{int(end_label)}'].add(index)
    for member_indices in node_to_segments.values():
        members = sorted(member_indices)
        for position, left in enumerate(members):
            for right in members[position + 1:]:
                if left != right:
                    adjacency[left].add(right)
                    adjacency[right].add(left)
    adjacency_arrays = [np.array(sorted(values), dtype=np.int32) for values in adjacency]
    degree = np.array([len(values) for values in adjacency_arrays], dtype=int)
    isolated_indices = np.flatnonzero(degree == 0)
    if len(isolated_indices) > cfg.MAX_ALLOWED_ISOLATED_SEGMENTS:
        isolated_path = output_dir / 'isolated_segments.csv'
        write_output_csv(pd.DataFrame({cfg.COL_SEG: np.asarray(segment_ids, dtype=object)[isolated_indices], 'degree': degree[isolated_indices]}), isolated_path, index=False)
        raise RuntimeError(f'Road graph contains {len(isolated_indices)} isolated segments, exceeding MAX_ALLOWED_ISOLATED_SEGMENTS={cfg.MAX_ALLOWED_ISOLATED_SEGMENTS}. DeepWalk cannot train meaningful vectors for isolated nodes. Inspect {isolated_path}.')
    graph_path = output_dir / 'segment_graph.pkl'
    with graph_path.open('wb') as handle:
        pickle.dump({'segment_ids': segment_ids, 'adjacency': adjacency_arrays}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    summary = {'segments': n, 'parent_edge_count': int(segments['source_edge_uid'].nunique()), 'authoritative_topology_used': bool(authoritative), 'mean_degree': float(degree.mean()), 'median_degree': float(np.median(degree)), 'maximum_degree': int(degree.max()), 'isolated_segments': int(len(isolated_indices)), 'fallback_endpoint_diagnostics': fallback_diagnostics}
    atomic_json_dump(summary, output_dir / 'graph_summary.json')
    return str(graph_path)

def load_graph(graph_path: str) -> GraphArtifacts:
    with open(graph_path, 'rb') as handle:
        payload = pickle.load(handle)
    return GraphArtifacts(payload['segment_ids'], payload['adjacency'])

def train_deepwalk(cfg: Config, graph_path: str, out_dir: str, *, dimension: int | None=None, n_walks: int | None=None, walk_length: int | None=None, seed: int | None=None, workers: int | None=None, tag: str='selected') -> Tuple[str, dict]:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    graph = load_graph(graph_path)
    isolated = [index for index, values in enumerate(graph.adjacency) if len(values) == 0]
    if isolated:
        raise RuntimeError(f'DeepWalk refused to train because the graph contains {len(isolated)} isolated nodes.')
    dimension = int(cfg.EMB_DIM if dimension is None else dimension)
    n_walks = int(cfg.N_WALKS if n_walks is None else n_walks)
    walk_length = int(cfg.WALK_LEN if walk_length is None else walk_length)
    seed = int(cfg.SEED if seed is None else seed)
    workers = int(cfg.W2V_WORKERS if workers is None else workers)
    corpus = DeepWalkCorpus(graph.segment_ids, graph.adjacency, n_walks, walk_length, seed)
    start = time.perf_counter()
    model = Word2Vec(sentences=corpus, vector_size=dimension, window=cfg.W2V_WINDOW, min_count=1, workers=max(1, workers), sg=1, epochs=cfg.W2V_EPOCHS, seed=seed, sorted_vocab=1, hashfxn=_stable_token_hash)
    elapsed = time.perf_counter() - start
    missing = [segment_id for segment_id in graph.segment_ids if segment_id not in model.wv]
    if missing:
        raise RuntimeError(f'DeepWalk model lacks {len(missing)} segment vectors: {missing[:10]}')
    vectors = np.vstack([model.wv[segment_id] for segment_id in graph.segment_ids]).astype(np.float32)
    if not np.isfinite(vectors).all():
        raise RuntimeError('DeepWalk produced nonfinite embedding values.')
    embedding = pd.DataFrame(vectors, columns=[f'gemb_{index}' for index in range(dimension)])
    embedding.insert(0, cfg.COL_SEG, graph.segment_ids)
    output = output_dir / f'segment_embeddings_{tag}.parquet'
    embedding.to_parquet(output, index=False)
    model.lifecycle_events = None
    model.wv.lifecycle_events = None
    model.save(str(output_dir / f'deepwalk_{tag}.model'))
    metrics = {'tag': tag, 'dimension': dimension, 'walks_per_node': n_walks, 'walk_length': walk_length, 'window': cfg.W2V_WINDOW, 'epochs': cfg.W2V_EPOCHS, 'seed': seed, 'workers': workers, 'runtime_seconds': elapsed, 'embedding_rows': len(embedding), 'embedding_bytes': int(vectors.nbytes)}
    atomic_json_dump(metrics, output_dir / f'deepwalk_{tag}_runtime.json')
    return (str(output), metrics)

def _bfs_distances(adjacency: Sequence[np.ndarray], source: int, max_depth: int=4) -> dict[int, int]:
    distances = {source: 0}
    queue = deque([source])
    while queue:
        current = queue.popleft()
        depth = distances[current]
        if depth >= max_depth:
            continue
        for neighbour in adjacency[current]:
            neighbour = int(neighbour)
            if neighbour not in distances:
                distances[neighbour] = depth + 1
                queue.append(neighbour)
    return distances

def evaluate_embedding_quality(cfg: Config, graph_path: str, embedding_path: str, *, seed: int) -> dict:
    graph = load_graph(graph_path)
    embedding = pd.read_parquet(embedding_path)
    columns = [column for column in embedding.columns if column.startswith('gemb_')]
    vectors = embedding[columns].to_numpy(dtype=np.float32)
    rng = np.random.default_rng(seed)
    sample_size = min(cfg.SENSITIVITY_PAIR_SAMPLE, len(vectors))
    sampled_nodes = rng.choice(len(vectors), size=sample_size, replace=False)
    nearest = NearestNeighbors(n_neighbors=min(cfg.SENSITIVITY_NEIGHBOR_K + 1, len(vectors))).fit(vectors)
    _, indices = nearest.kneighbors(vectors[sampled_nodes])
    one_hop_recalls, three_hop_recalls, pair_graph, pair_embedding = ([], [], [], [])
    for source, embedding_neighbours in zip(sampled_nodes, indices):
        embedding_set = set(map(int, embedding_neighbours[1:]))
        one_hop = set(map(int, graph.adjacency[int(source)]))
        if one_hop:
            one_hop_recalls.append(len(embedding_set.intersection(one_hop)) / len(one_hop))
        distances = _bfs_distances(graph.adjacency, int(source), max_depth=4)
        three_hop = {node for node, distance in distances.items() if 1 <= distance <= 3}
        if three_hop:
            three_hop_recalls.append(len(embedding_set.intersection(three_hop)) / min(len(three_hop), len(embedding_set)))
        candidates = [(node, distance) for node, distance in distances.items() if distance > 0]
        if candidates:
            target, graph_distance = candidates[int(rng.integers(0, len(candidates)))]
            pair_graph.append(int(graph_distance))
            pair_embedding.append(float(np.linalg.norm(vectors[int(source)] - vectors[int(target)])))
    correlation = np.nan
    if len(pair_graph) > 2 and np.std(pair_graph) > 0 and (np.std(pair_embedding) > 0):
        correlation = float(spearmanr(pair_graph, pair_embedding).statistic)
    return {'one_hop_recall_mean': float(np.mean(one_hop_recalls)) if one_hop_recalls else np.nan, 'three_hop_recall_mean': float(np.mean(three_hop_recalls)) if three_hop_recalls else np.nan, 'graph_embedding_distance_correlation': correlation, 'graph_embedding_distance_correlation_method': 'Spearman', 'sampled_nodes': int(sample_size)}

def _connected_graph_sample(graph: GraphArtifacts, limit: int, seed: int) -> GraphArtifacts:
    if len(graph.segment_ids) <= limit:
        return graph
    rng = np.random.default_rng(seed)
    nonisolated = np.array([i for i, values in enumerate(graph.adjacency) if len(values) > 0], dtype=int)
    if not len(nonisolated):
        raise RuntimeError('Cannot draw a connected sensitivity graph from an entirely isolated graph.')
    start = int(rng.choice(nonisolated))
    selected, selected_set, queue = ([], set(), deque([start]))
    while queue and len(selected) < limit:
        node = int(queue.popleft())
        if node in selected_set:
            continue
        selected_set.add(node)
        selected.append(node)
        neighbours = list(map(int, graph.adjacency[node]))
        rng.shuffle(neighbours)
        queue.extend(neighbours)
    if len(selected) < limit:
        remaining = [i for i in nonisolated if i not in selected_set]
        selected.extend(remaining[:limit - len(selected)])
    selected = np.array(selected[:limit], dtype=int)
    old_to_new = {old: new for new, old in enumerate(selected)}
    adjacency = [np.array(sorted((old_to_new[n] for n in map(int, graph.adjacency[int(old)]) if n in old_to_new)), dtype=np.int32) for old in selected]
    if any((len(values) == 0 for values in adjacency)):
        keep = np.array([len(values) > 0 for values in adjacency], dtype=bool)
        kept_old = selected[keep]
        old_to_new = {old: new for new, old in enumerate(kept_old)}
        adjacency = [np.array(sorted((old_to_new[n] for n in map(int, graph.adjacency[int(old)]) if n in old_to_new)), dtype=np.int32) for old in kept_old]
        selected = kept_old
    return GraphArtifacts([graph.segment_ids[int(i)] for i in selected], adjacency)

def embedding_screen_specifications(cfg: Config) -> list[tuple[int, int, int, str]]:
    specifications = []
    for dimension in cfg.EMB_DIM_CANDIDATES:
        specifications.append((dimension, cfg.N_WALKS, cfg.SENSITIVITY_REFERENCE_WALK_LEN, 'dimension'))
    for walks in cfg.N_WALKS_CANDIDATES:
        specifications.append((cfg.EMB_DIM, walks, cfg.SENSITIVITY_REFERENCE_WALK_LEN, 'walks'))
    for length in cfg.WALK_LEN_CANDIDATES:
        specifications.append((cfg.EMB_DIM, cfg.N_WALKS, length, 'length'))
    seen = set()
    unique = []
    for specification in specifications:
        if specification[:3] not in seen:
            seen.add(specification[:3])
            unique.append(specification)
    return unique


def run_embedding_sensitivity(cfg: Config, graph_path: str, out_dir: str) -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    full_graph = load_graph(graph_path)
    screen_graph = _connected_graph_sample(full_graph, cfg.SENSITIVITY_GRAPH_NODE_LIMIT, cfg.SEED)
    screen_graph_path = out / 'sensitivity_graph.pkl'
    with screen_graph_path.open('wb') as handle:
        pickle.dump({'segment_ids': screen_graph.segment_ids, 'adjacency': screen_graph.adjacency}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    specifications = embedding_screen_specifications(cfg)
    rows, seen = ([], set())
    checkpoint_dir = out / 'candidate_checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for dimension, walks, length, family in specifications:
        key = (dimension, walks, length)
        if key in seen:
            continue
        seen.add(key)
        for seed in [int(cfg.SENSITIVITY_SCREEN_SEED)]:
            tag = f'd{dimension}_w{walks}_l{length}_seed{seed}'
            checkpoint_path = checkpoint_dir / f'{tag}.json'
            checkpoint = load_success_checkpoint(checkpoint_path, artifact_root=out, required_artifacts=[out / f'segment_embeddings_{tag}.parquet', out / f'deepwalk_{tag}_runtime.json', out / f'deepwalk_{tag}.model'])
            if checkpoint is not None:
                result = dict(checkpoint['result'])
                result['embedding_path'] = str((out / f'segment_embeddings_{tag}.parquet').resolve())
                rows.append(result)
                continue
            embedding_path, runtime = train_deepwalk(cfg, str(screen_graph_path), str(out), dimension=dimension, n_walks=walks, walk_length=length, seed=seed, workers=1, tag=tag)
            quality = evaluate_embedding_quality(cfg, str(screen_graph_path), embedding_path, seed=seed)
            result = {'family': family, 'embedding_path': embedding_path, **runtime, **quality}
            runtime_path = out / f'deepwalk_{tag}_runtime.json'
            model_path = out / f'deepwalk_{tag}.model'
            write_success_checkpoint(checkpoint_path, result=result, artifacts=[embedding_path, runtime_path, model_path], artifact_root=out)
            rows.append(result)
    raw = pd.DataFrame(rows)
    for column in ['one_hop_recall_mean', 'three_hop_recall_mean']:
        maximum = raw[column].max()
        raw[f'{column}_loss'] = maximum - raw[column]
    runtime_min, runtime_max = (raw['runtime_seconds'].min(), raw['runtime_seconds'].max())
    raw['runtime_scaled'] = (raw['runtime_seconds'] - runtime_min) / max(runtime_max - runtime_min, 1e-09)
    raw['screen_score'] = 0.45 * raw['one_hop_recall_mean_loss'].fillna(1) + 0.35 * raw['three_hop_recall_mean_loss'].fillna(1) + cfg.RUNTIME_SCORE_WEIGHT * raw['runtime_scaled'] + 0.1 * (1 - raw['graph_embedding_distance_correlation'].clip(lower=-1.0, upper=1.0).fillna(0))
    raw_path = out / 'embedding_sensitivity_runs.csv'
    write_output_csv(raw, raw_path, index=False)
    group_columns = ['dimension', 'walks_per_node', 'walk_length']
    summary = raw.groupby(group_columns, as_index=False).agg(family=('family', 'first'), seeds=('seed', 'nunique'), screen_score_mean=('screen_score', 'mean'), screen_score_sd=('screen_score', 'std'), runtime_seconds_mean=('runtime_seconds', 'mean'), one_hop_recall_mean=('one_hop_recall_mean', 'mean'), three_hop_recall_mean=('three_hop_recall_mean', 'mean'), graph_embedding_distance_correlation=('graph_embedding_distance_correlation', 'mean'))
    summary['screen_score_se'] = summary['screen_score_sd'] / np.sqrt(summary['seeds'].clip(lower=1))
    output = out / 'embedding_sensitivity.csv'
    write_output_csv(summary.sort_values('screen_score_mean'), output, index=False)
    return str(output)
