from __future__ import annotations
import math
import numpy as np
from shapely.geometry import LineString, Point

def point_on_line_with_signed_offset(line_m: LineString, segment_position: float, signed_offset_m: float) -> Point:
    position = float(np.clip(segment_position, 0.0, 1.0))
    base = line_m.interpolate(position, normalized=True)
    if abs(float(signed_offset_m)) <= 1e-12:
        return base
    epsilon = min(0.0001, 0.25 / max(float(line_m.length), 1.0))
    previous = line_m.interpolate(max(0.0, position - epsilon), normalized=True)
    following = line_m.interpolate(min(1.0, position + epsilon), normalized=True)
    dx, dy = (following.x - previous.x, following.y - previous.y)
    norm = math.hypot(dx, dy)
    if norm <= 1e-12:
        return base
    normal_x, normal_y = (-dy / norm, dx / norm)
    return Point(base.x + float(signed_offset_m) * normal_x, base.y + float(signed_offset_m) * normal_y)

def point_on_line_with_offset(line_m: LineString, seg_pos01: float, offset_m: float, rng: np.random.Generator) -> Point:
    signed = abs(float(offset_m)) if rng.random() >= 0.5 else -abs(float(offset_m))
    return point_on_line_with_signed_offset(line_m, seg_pos01, signed)
