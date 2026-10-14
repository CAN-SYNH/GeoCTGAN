from __future__ import annotations
import logging
import math
import multiprocessing as mp
import queue
import subprocess
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple
import numpy as np
from sklearn.neighbors import NearestNeighbors
logger = logging.getLogger(__name__)

@dataclass
class FlowResult:
    assigned_segment_ids: np.ndarray
    diagnostics: dict

@lru_cache(maxsize=1)
def ortools_available() -> bool:
    try:
        probe = subprocess.run([sys.executable, '-c', 'from ortools.graph.python import min_cost_flow; min_cost_flow.SimpleMinCostFlow()'], capture_output=True, text=True, timeout=30, check=False)
        return probe.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False

def largest_remainder_counts(observed_counts: Dict[str, int], target_total: int) -> Dict[str, int]:
    items = sorted(((str(key), int(value)) for key, value in observed_counts.items() if int(value) > 0))
    if target_total < 0:
        raise ValueError('target_total cannot be negative.')
    if not items:
        if target_total == 0:
            return {}
        raise ValueError('Cannot allocate a positive target total from empty observed counts.')
    total = sum((value for _, value in items))
    expected = [(key, target_total * value / total) for key, value in items]
    allocated = {key: int(math.floor(value)) for key, value in expected}
    remainder = target_total - sum(allocated.values())
    order = sorted(expected, key=lambda pair: (-(pair[1] - math.floor(pair[1])), pair[0]))
    for key, _ in order[:remainder]:
        allocated[key] += 1
    if sum(allocated.values()) != target_total or any((value < 0 for value in allocated.values())):
        raise RuntimeError('Largest-remainder allocation violated its invariants.')
    return {key: value for key, value in allocated.items() if value > 0}

def build_capacities(observed_counts: Dict[str, int], target_total: int, *, mode: str, upper_slack: float=0.25) -> Dict[str, int]:
    exact = largest_remainder_counts(observed_counts, target_total)
    if mode == 'exact_quota':
        return exact
    if mode != 'upper_quota':
        raise ValueError(f'Unsupported capacity mode: {mode!r}')
    slack_total = int(math.ceil(float(upper_slack) * target_total))
    if slack_total <= 0:
        return exact
    slack = largest_remainder_counts(observed_counts, slack_total)
    result = dict(exact)
    for segment_id, count in slack.items():
        result[segment_id] = result.get(segment_id, 0) + int(count)
    if sum(result.values()) < target_total:
        raise RuntimeError('Upper-capacity total is smaller than the target sample size.')
    return result

def _solve_candidate_flow_direct(candidate_indices: np.ndarray, candidate_distances: np.ndarray, capacities: np.ndarray, cost_scale: int, *, require_ortools: bool) -> Tuple[bool, np.ndarray | None, int | None, dict]:
    if not require_ortools:
        min_cost_flow = None
        ortools_error = None
    else:
        try:
            from ortools.graph.python import min_cost_flow
        except Exception as exc:
            min_cost_flow = None
            ortools_error = exc
        else:
            ortools_error = None
    event_count, candidate_count = candidate_indices.shape
    edge_count = int(event_count * candidate_count)
    if min_cost_flow is None:
        if require_ortools:
            raise ImportError(f'OR-Tools is required for the production CaNN solver but its native extension could not be imported. Install the exact locked environment. Underlying error: {ortools_error!r}') from ortools_error
        if event_count > 2000:
            raise ImportError('The SciPy fallback is restricted to engineering groups of at most 2,000 events.')
        from scipy.optimize import linear_sum_assignment
        slot_segments = np.repeat(np.arange(len(capacities), dtype=int), capacities.astype(int))
        if len(slot_segments) < event_count:
            return (False, None, None, {'solver_status': -1, 'solver_optimal': False, 'candidate_count': candidate_count, 'candidate_edges': edge_count, 'solver_backend': 'scipy_fallback'})
        huge = np.int64(10 ** 15)
        cost_matrix = np.full((event_count, len(slot_segments)), huge, dtype=np.int64)
        rank_lookup = []
        for event_index in range(event_count):
            lookup = {}
            for rank, (segment_index, distance) in enumerate(zip(candidate_indices[event_index], candidate_distances[event_index])):
                lookup[int(segment_index)] = (int(round(float(distance) ** 2 * int(cost_scale))), rank)
            rank_lookup.append(lookup)
            for slot_index, segment_index in enumerate(slot_segments):
                if int(segment_index) in lookup:
                    cost_matrix[event_index, slot_index] = lookup[int(segment_index)][0]
        start = time.perf_counter()
        row_index, column_index = linear_sum_assignment(cost_matrix)
        elapsed = time.perf_counter() - start
        selected_costs = cost_matrix[row_index, column_index]
        if np.any(selected_costs >= huge):
            return (False, None, None, {'solver_status': -2, 'solver_optimal': False, 'solver_runtime_seconds': elapsed, 'candidate_count': candidate_count, 'candidate_edges': edge_count, 'solver_backend': 'scipy_fallback'})
        assigned = slot_segments[column_index].astype(np.int32)
        selected_ranks = np.array([rank_lookup[int(event)][int(segment)][1] for event, segment in zip(row_index, assigned)], dtype=np.int32)
        objective = int(selected_costs.sum())
        return (True, assigned, objective, {'solver_status': 0, 'solver_optimal': True, 'solver_runtime_seconds': elapsed, 'candidate_count': candidate_count, 'candidate_edges': edge_count, 'objective_cost': objective, 'selected_rank_median': float(np.median(selected_ranks + 1)), 'selected_rank_p95': float(np.quantile(selected_ranks + 1, 0.95)), 'boundary_selection_share': float((selected_ranks == candidate_count - 1).mean()), 'solver_backend': 'scipy_fallback'})
    segment_count = len(capacities)
    source = 0
    event_offset = 1
    segment_offset = event_offset + event_count
    sink = segment_offset + segment_count
    solver = min_cost_flow.SimpleMinCostFlow()
    event_segment_arcs: List[Tuple[int, int, int, int]] = []
    for event_index in range(event_count):
        solver.add_arc_with_capacity_and_unit_cost(source, event_offset + event_index, 1, 0)
        for candidate_rank in range(candidate_count):
            segment_index = int(candidate_indices[event_index, candidate_rank])
            distance = float(candidate_distances[event_index, candidate_rank])
            if not np.isfinite(distance):
                continue
            cost = int(round(distance * distance * int(cost_scale)))
            arc = solver.add_arc_with_capacity_and_unit_cost(event_offset + event_index, segment_offset + segment_index, 1, max(cost, 0))
            event_segment_arcs.append((arc, event_index, segment_index, candidate_rank))
    for segment_index, capacity in enumerate(capacities):
        solver.add_arc_with_capacity_and_unit_cost(segment_offset + segment_index, sink, int(capacity), 0)
    solver.set_node_supply(source, event_count)
    solver.set_node_supply(sink, -event_count)
    start = time.perf_counter()
    status = solver.solve()
    elapsed = time.perf_counter() - start
    optimal = status == solver.OPTIMAL
    diagnostics = {'solver_status': int(status), 'solver_optimal': bool(optimal), 'solver_runtime_seconds': elapsed, 'candidate_count': candidate_count, 'candidate_edges': len(event_segment_arcs), 'solver_backend': 'ortools'}
    if not optimal:
        return (False, None, None, diagnostics)
    assigned = np.full(event_count, -1, dtype=np.int32)
    selected_ranks = np.full(event_count, -1, dtype=np.int32)
    for arc, event_index, segment_index, candidate_rank in event_segment_arcs:
        if solver.flow(arc) > 0:
            assigned[event_index] = segment_index
            selected_ranks[event_index] = candidate_rank
    if (assigned < 0).any():
        raise RuntimeError('Min-cost flow returned an optimal status but left events unassigned.')
    diagnostics.update({'objective_cost': int(solver.optimal_cost()), 'selected_rank_median': float(np.median(selected_ranks + 1)), 'selected_rank_p95': float(np.quantile(selected_ranks + 1, 0.95)), 'boundary_selection_share': float((selected_ranks == candidate_count - 1).mean())})
    return (True, assigned, int(solver.optimal_cost()), diagnostics)

def _flow_worker(payload, result_queue) -> None:
    try:
        result_queue.put(('ok', _solve_candidate_flow_direct(**payload)))
    except BaseException as exc:
        result_queue.put(('error', repr(exc)))

def _solve_candidate_flow(candidate_indices: np.ndarray, candidate_distances: np.ndarray, capacities: np.ndarray, cost_scale: int, time_limit_seconds: int, *, max_edges: int, require_ortools: bool) -> Tuple[bool, np.ndarray | None, int | None, dict]:
    edge_count = int(candidate_indices.shape[0] * candidate_indices.shape[1])
    if edge_count > int(max_edges):
        return (False, None, None, {'solver_status': -10, 'solver_optimal': False, 'solver_timed_out': False, 'edge_budget_exceeded': True, 'candidate_count': int(candidate_indices.shape[1]), 'candidate_edges': edge_count})
    payload = {'candidate_indices': candidate_indices, 'candidate_distances': candidate_distances, 'capacities': capacities, 'cost_scale': cost_scale, 'require_ortools': require_ortools}
    context = mp.get_context('spawn')
    result_queue = context.Queue(maxsize=1)
    process = context.Process(target=_flow_worker, args=(payload, result_queue), daemon=True)
    process.start()
    try:
        kind, value = result_queue.get(timeout=float(time_limit_seconds))
    except queue.Empty:
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
        return (False, None, None, {'solver_status': -11, 'solver_optimal': False, 'solver_timed_out': True, 'edge_budget_exceeded': False, 'candidate_count': int(candidate_indices.shape[1]), 'candidate_edges': edge_count, 'solver_backend': 'subprocess_timeout_guard'})
    process.join(timeout=5)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        raise RuntimeError('CaNN solver returned a result but its subprocess did not exit cleanly.')
    if kind == 'error':
        raise RuntimeError(f'CaNN solver subprocess failed: {value}')
    feasible, assigned, objective, diagnostics = value
    diagnostics['solver_timed_out'] = False
    diagnostics['edge_budget_exceeded'] = False
    return (feasible, assigned, objective, diagnostics)

def assign_flow_cann(generated_queries: np.ndarray, segment_ids: Sequence[str], segment_embeddings: np.ndarray, capacities: Dict[str, int], *, capacity_mode: str, candidate_schedule: Sequence[int], cost_scale: int=100000, objective_tolerance: float=0.0001, stable_rounds: int=1, full_support_threshold: int=400, time_limit_seconds: int=600, max_edges: int=5000000, require_ortools: bool=True) -> FlowResult:
    generated_queries = np.asarray(generated_queries, dtype=np.float32)
    segment_embeddings = np.asarray(segment_embeddings, dtype=np.float32)
    segment_ids = np.asarray([str(value) for value in segment_ids], dtype=object)
    if generated_queries.ndim != 2 or segment_embeddings.ndim != 2:
        raise ValueError('generated_queries and segment_embeddings must be two-dimensional.')
    if generated_queries.shape[1] != segment_embeddings.shape[1]:
        raise ValueError('Generated and real embedding dimensions do not match.')
    if not np.isfinite(generated_queries).all() or not np.isfinite(segment_embeddings).all():
        raise ValueError('CaNN received nonfinite embedding values.')
    if require_ortools and (not ortools_available()):
        raise ImportError('OR-Tools is required but not importable. Run the production solver self-test first.')
    id_to_index = {segment_id: index for index, segment_id in enumerate(segment_ids)}
    support_ids = sorted((segment_id for segment_id, value in capacities.items() if int(value) > 0))
    missing = [segment_id for segment_id in support_ids if segment_id not in id_to_index]
    if missing:
        raise KeyError(f'Capacity-bearing segments are missing from the embedding table: {missing[:10]}')
    support_indices = np.array([id_to_index[segment_id] for segment_id in support_ids], dtype=np.int32)
    support_embeddings = segment_embeddings[support_indices]
    support_capacities = np.array([int(capacities[segment_id]) for segment_id in support_ids], dtype=np.int32)
    event_count = len(generated_queries)
    if event_count == 0:
        return FlowResult(np.array([], dtype=object), {'events': 0, 'capacity_violations': 0})
    if support_capacities.sum() < event_count:
        raise RuntimeError(f'CaNN capacity total {int(support_capacities.sum())} is smaller than event count {event_count}.')
    scale = support_embeddings.std(axis=0)
    scale = np.where(scale > 1e-08, scale, 1.0)
    centre = support_embeddings.mean(axis=0)
    support_z = (support_embeddings - centre) / scale
    query_z = (generated_queries - centre) / scale
    nearest = NearestNeighbors(algorithm='auto').fit(support_z)
    schedule = sorted({min(int(value), len(support_ids)) for value in candidate_schedule if int(value) > 0})
    full_edge_count = int(event_count * len(support_ids))
    max_k_by_budget = max(1, min(len(support_ids), int(max_edges) // max(event_count, 1)))
    current = schedule[-1] if schedule else min(16, max_k_by_budget)
    while current < max_k_by_budget:
        current = min(max_k_by_budget, max(current + 1, current * 2))
        schedule.append(int(current))
    if len(support_ids) <= full_support_threshold and full_edge_count <= int(max_edges) and (not schedule or schedule[-1] < len(support_ids)):
        schedule.append(len(support_ids))
    schedule = sorted({k for k in schedule if int(event_count * k) <= int(max_edges)})
    if not schedule:
        raise RuntimeError(f'No CaNN candidate size fits CANN_MAX_EDGES={max_edges:,} for events={event_count:,}, support={len(support_ids):,}.')
    solutions, stable, previous_objective = ([], 0, None)
    best_result = None
    for k in schedule:
        distances, candidates = nearest.kneighbors(query_z, n_neighbors=k, return_distance=True)
        feasible, assigned_local, objective, diagnostics = _solve_candidate_flow(candidates, distances, support_capacities, cost_scale, time_limit_seconds, max_edges=max_edges, require_ortools=require_ortools)
        diagnostics['k'] = int(k)
        solutions.append(diagnostics)
        if not feasible:
            if diagnostics.get('solver_timed_out'):
                raise TimeoutError(f'CaNN group exceeded {time_limit_seconds}s at k={k}; partial checkpoints are preserved.')
            continue
        if previous_objective is not None:
            relative_improvement = (previous_objective - objective) / max(abs(previous_objective), 1)
            diagnostics['relative_objective_improvement'] = float(relative_improvement)
            stable = stable + 1 if relative_improvement <= objective_tolerance else 0
        else:
            diagnostics['relative_objective_improvement'] = np.nan
        previous_objective = objective
        selected_global = support_indices[assigned_local]
        assigned_ids = segment_ids[selected_global]
        counts = {segment_id: 0 for segment_id in support_ids}
        for segment_id in assigned_ids:
            counts[str(segment_id)] += 1
        violations = sum((max(0, counts[s] - capacities[s]) for s in support_ids))
        exact_mismatch = sum((abs(counts[s] - capacities[s]) for s in support_ids))
        if violations:
            raise RuntimeError(f'CaNN capacity violation detected: {violations}')
        if capacity_mode == 'exact_quota' and exact_mismatch:
            raise RuntimeError(f'Exact-quota CaNN did not fill the calibrated count vector: mismatch={exact_mismatch}')
        best_result = FlowResult(assigned_ids.astype(object), {'events': event_count, 'support_segments': len(support_ids), 'capacity_total': int(support_capacities.sum()), 'capacity_mode': capacity_mode, 'final_k': int(k), 'capacity_violations': 0, 'exact_quota_mismatch': int(exact_mismatch), 'full_support_edges': full_edge_count, 'full_support_considered': bool(k == len(support_ids)), 'solutions': solutions, **diagnostics})
        if k == len(support_ids) or stable >= stable_rounds:
            return best_result
    if best_result is not None:
        return best_result
    raise RuntimeError('No feasible CaNN assignment was found within the configured candidate schedule and hard edge budget. Increase K_CANDIDATE_SCHEDULE or CANN_MAX_EDGES only after inspecting the saved group preflight.')

def assign_exact_flow_cann(*args, **kwargs) -> FlowResult:
    kwargs.setdefault('capacity_mode', 'exact_quota')
    return assign_flow_cann(*args, **kwargs)

def assign_capacity_only(event_count: int, capacities: Dict[str, int], *, seed: int) -> FlowResult:
    slots = np.concatenate([np.repeat(str(segment_id), int(count)) for segment_id, count in sorted(capacities.items())])
    if len(slots) < int(event_count):
        raise RuntimeError('CapacityOnly has fewer slots than events.')
    rng = np.random.default_rng(int(seed))
    rng.shuffle(slots)
    assigned = slots[:int(event_count)].astype(object)
    return FlowResult(assigned, {'events': int(event_count), 'capacity_total': int(len(slots)), 'capacity_violations': 0, 'solver_backend': 'capacity_only_slot_shuffle'})

def assign_no_capacity(generated_queries: np.ndarray, segment_ids: Sequence[str], segment_embeddings: np.ndarray) -> FlowResult:
    generated_queries = np.asarray(generated_queries, dtype=np.float32)
    segment_embeddings = np.asarray(segment_embeddings, dtype=np.float32)
    scale = segment_embeddings.std(axis=0)
    scale = np.where(scale > 1e-08, scale, 1.0)
    centre = segment_embeddings.mean(axis=0)
    nearest = NearestNeighbors(n_neighbors=1).fit((segment_embeddings - centre) / scale)
    distances, indices = nearest.kneighbors((generated_queries - centre) / scale)
    ids = np.asarray(list(map(str, segment_ids)), dtype=object)[indices[:, 0]]
    return FlowResult(ids, {'events': len(ids), 'median_distance': float(np.median(distances[:, 0])), 'p95_distance': float(np.quantile(distances[:, 0], 0.95)), 'solver_backend': 'nearest_no_capacity'})

def production_solver_self_test(time_limit_seconds: int=20) -> dict:
    if not ortools_available():
        raise ImportError('OR-Tools is unavailable; the production CaNN path cannot be validated.')
    candidate_indices = np.array([[0, 1], [1, 0]], dtype=np.int32)
    candidate_distances = np.array([[0.1, 1.0], [0.1, 1.0]], dtype=np.float32)
    capacities = np.array([1, 1], dtype=np.int32)
    feasible, assigned, objective, diagnostics = _solve_candidate_flow(candidate_indices, candidate_distances, capacities, 100000, time_limit_seconds, max_edges=100, require_ortools=True)
    if not feasible or assigned is None or sorted(assigned.tolist()) != [0, 1]:
        raise RuntimeError(f'OR-Tools production self-test failed: feasible={feasible}, assigned={assigned}, diagnostics={diagnostics}')
    if diagnostics.get('solver_backend') != 'ortools':
        raise RuntimeError(f'Production self-test used the wrong backend: {diagnostics}')
    return {'objective': objective, **diagnostics}
