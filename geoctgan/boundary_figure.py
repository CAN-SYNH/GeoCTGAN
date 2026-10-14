from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial.distance import jensenshannon

from .utils import write_output_csv

METRICS = (
    "count_relative_change",
    "category_js_shift",
    "hour_js_shift",
    "centroid_shift_m",
)
ERRORS = tuple(f"{metric}_absolute_error" for metric in METRICS)
METHODS = {"GeoCTGAN-Full": "Full", "GeoCTGAN-Priv": "Priv"}
COLORS = {"Full": "#0072B2", "Priv": "#D55E00"}


def _dates(frame):
    if "OCC_DATE" in frame:
        return pd.to_datetime(frame["OCC_DATE"], errors="coerce")
    return pd.to_datetime(
        {"year": frame["OCC_YEAR"], "month": frame["OCC_MONTH"], "day": frame["OCC_DAY"]},
        errors="coerce",
    )


def _js(left, right):
    levels = sorted(set(left.dropna().astype(str)) | set(right.dropna().astype(str)))
    p = left.astype(str).value_counts().reindex(levels, fill_value=0).to_numpy(float)
    q = right.astype(str).value_counts().reindex(levels, fill_value=0).to_numpy(float)
    if not len(levels) or p.sum() == 0 or q.sum() == 0:
        return float("nan")
    return float(jensenshannon(p / p.sum(), q / q.sum(), base=2) ** 2)


def _changes(frame, boundaries, transformer, longitude, latitude, category):
    dates = _dates(frame)
    rows = []
    for boundary in boundaries:
        before = frame.loc[dates.between(boundary - pd.Timedelta(days=7), boundary - pd.Timedelta(days=1))]
        after = frame.loc[dates.between(boundary, boundary + pd.Timedelta(days=6))]
        first = np.asarray(transformer.transform(before[longitude].mean(), before[latitude].mean()))
        second = np.asarray(transformer.transform(after[longitude].mean(), after[latitude].mean()))
        rows.append({
            "boundary": str(boundary.date()),
            "count_relative_change": (len(after) - len(before)) / max(len(before), 1),
            "category_js_shift": _js(before[category], after[category]),
            "hour_js_shift": _js(before["OCC_HOUR"], after["OCC_HOUR"]),
            "centroid_shift_m": float(np.linalg.norm(second - first)),
        })
    return pd.DataFrame(rows, columns=["boundary", *METRICS])


def boundary_tables(real, frames, cfg=None):
    longitude = getattr(cfg, "longitude_column", "LONG_WGS84")
    latitude = getattr(cfg, "latitude_column", "LAT_WGS84")
    category = getattr(cfg, "category_column", "MCI_CATEGO")
    transformer = Transformer.from_crs(
        getattr(cfg, "geographic_crs", "EPSG:4326"),
        getattr(cfg, "metric_crs", "EPSG:32617"),
        always_xy=True,
    )
    dates = _dates(real).dropna()
    if dates.empty:
        raise ValueError("OCC_DATE or OCC_YEAR/OCC_MONTH/OCC_DAY is required.")
    boundaries = pd.date_range(
        dates.min().to_period("M").to_timestamp(),
        dates.max().to_period("M").to_timestamp(),
        freq="MS",
    )[1:]
    if not len(boundaries):
        raise ValueError("At least two calendar months are required.")
    reference = _changes(real, boundaries, transformer, longitude, latitude, category)
    tables = []
    for (method, seed), frame in sorted(frames.items()):
        if method not in METHODS:
            continue
        synthetic = _changes(frame, boundaries, transformer, longitude, latitude, category)
        table = pd.DataFrame({"profile": METHODS[method], "seed": int(seed), "boundary": reference["boundary"]})
        for metric in METRICS:
            table[f"real_{metric}"] = reference[metric]
            table[f"synthetic_{metric}"] = synthetic[metric]
            table[f"{metric}_absolute_error"] = (synthetic[metric] - reference[metric]).abs()
        tables.append(table)
    if not tables:
        raise ValueError("GeoCTGAN-Full and GeoCTGAN-Priv tables are required.")
    details = pd.concat(tables, ignore_index=True)
    summary = details.groupby(["profile", "seed"], as_index=False)[list(ERRORS)].mean()
    return details, summary


def generate_boundary_figure(real, frames, artifacts, cfg=None):
    details, summary = boundary_tables(real, frames, cfg)
    write_output_csv(details, artifacts.tables / "month_boundary_continuity.csv", index=False)
    write_output_csv(summary, artifacts.tables / "month_boundary_continuity_summary.csv", index=False)
    seeds = sorted(summary["seed"].unique())
    labels = [
        "Relative event-count\nchange error",
        "MCI-composition\nJS-shift error",
        "Hour-composition\nJS-shift error",
        "Centroid-shift error (m)",
    ]
    with plt.rc_context({"font.size": 12, "axes.labelsize": 13, "legend.fontsize": 12}):
        fig, axes = plt.subplots(2, 2, figsize=(10, 7.8))
        x = np.arange(len(seeds))
        for ax, metric, label, letter in zip(axes.flat, ERRORS, labels, "abcd"):
            for offset, (method, profile) in zip([-0.19, 0.19], METHODS.items()):
                values = summary.loc[summary["profile"].eq(profile)].set_index("seed")[metric].reindex(seeds)
                ax.bar(x + offset, values, width=0.38, color=COLORS[profile], edgecolor="#333333", linewidth=0.7, label=method)
            ax.set_xticks(x, [str(seed) for seed in seeds])
            ax.set_xlabel("Seed", labelpad=8)
            ax.set_ylabel(label, labelpad=10)
            ax.set_ylim(bottom=0)
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.text(-0.18, 1.035, letter, transform=ax.transAxes, fontsize=15, fontweight="bold", va="bottom")
        handles, names = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, names, loc="upper center", bbox_to_anchor=(0.53, 1.0), ncol=2, frameon=False)
        fig.subplots_adjust(left=0.13, right=0.98, bottom=0.09, top=0.89, hspace=0.42, wspace=0.39)
        fig.savefig(artifacts.figures / "Figure_S4.png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
