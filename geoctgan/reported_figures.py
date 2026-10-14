from __future__ import annotations

import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

METHODS = ("GeoCTGAN-Full", "GeoCTGAN-Priv")
COLORS = {"Real held-out": "#222222", METHODS[0]: "#0072B2", METHODS[1]: "#D55E00"}
FIELDS = ["seg_pos", "SEG_MAP_DIST_M", "LOCATION_T", "OFFENCE", "PREMISES_T", "OCC_HOUR", "OCC_DOW", "LAT_WGS84", "LONG_WGS84", "DIVISION", "OCC_YEAR", "MCI_CATEGO", "OCC_MONTH"]
LABELS = ["Segment position", "Map distance", "Location type", "Offence", "Premises type", "Hour", "Day of week", "Latitude", "Longitude", "Police division", "Year", "MCI category", "Month"]
NUMERICAL = {"seg_pos", "SEG_MAP_DIST_M", "LAT_WGS84", "LONG_WGS84"}


def categorical_hd(left, right):
    p = left.fillna("<NA>").astype(str).value_counts(normalize=True)
    q = right.fillna("<NA>").astype(str).value_counts(normalize=True)
    index = p.index.union(q.index)
    return float(np.sqrt(np.sum((np.sqrt(p.reindex(index, fill_value=0)) - np.sqrt(q.reindex(index, fill_value=0))) ** 2)) / np.sqrt(2))


def numeric_hd(left, right, bins=30):
    x = pd.to_numeric(left, errors="coerce").to_numpy(float)
    y = pd.to_numeric(right, errors="coerce").to_numpy(float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return categorical_hd(left, right)
    edges[0], edges[-1] = -np.inf, np.inf
    p, q = np.histogram(x, bins=edges)[0].astype(float), np.histogram(y, bins=edges)[0].astype(float)
    p, q = p / p.sum(), q / q.sum()
    return float(np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2))


def cramers_v(left, right):
    table = pd.crosstab(left.fillna("<NA>").astype(str), right.fillna("<NA>").astype(str))
    if min(table.shape) < 2:
        return 0.0
    n = table.to_numpy().sum()
    r, k = table.shape
    phi = max(0.0, chi2_contingency(table, correction=False)[0] / n - (k - 1) * (r - 1) / (n - 1))
    denominator = min(k - (k - 1) ** 2 / (n - 1) - 1, r - (r - 1) ** 2 / (n - 1) - 1)
    return float(np.sqrt(phi / denominator)) if denominator > 0 else 0.0


def association_matrix(frame, fields, kind):
    if kind == "Pearson":
        return frame[fields].apply(pd.to_numeric, errors="coerce").corr().to_numpy(float)
    result = np.eye(len(fields))
    for i in range(len(fields)):
        for j in range(i):
            result[i, j] = result[j, i] = cramers_v(frame[fields[i]], frame[fields[j]])
    return result


def seasonal_matrix(frame, field, levels):
    table = pd.crosstab(frame[field].fillna("<NA>").astype(str), pd.to_numeric(frame.OCC_MONTH, errors="coerce"))
    table = table.reindex(index=levels, columns=range(1, 13), fill_value=0).astype(float)
    return table.div(table.sum(axis=1).replace(0, np.nan), axis=0).fillna(0).to_numpy() * 100


def diagnostic_frames(real, synthetic):
    result = {}
    for label, frame in synthetic.items():
        match = re.fullmatch(r"(GeoCTGAN-Full|GeoCTGAN-Priv)\s*\|\s*seed\s+(\d+)", label)
        if match is None:
            continue
        method, seed = match.group(1), int(match.group(2))
        missing = [field for field in FIELDS if field not in frame]
        if missing:
            internal = synthetic.get(f"GeoCTGAN-MinimumSupport | seed {seed}")
            if internal is None or not set(missing).issubset(internal.columns):
                raise KeyError("The retained internal privacy-stage table is required for the 13-field diagnostics.")
            if not frame.SYNTHETIC_ID.is_unique or not internal.SYNTHETIC_ID.is_unique:
                raise ValueError("Synthetic identifiers must be unique for the diagnostic join.")
            if not frame.SYNTHETIC_ID.isin(internal.SYNTHETIC_ID).all():
                raise ValueError("Public identifiers are absent from the retained privacy-stage table.")
            frame = frame.merge(internal[["SYNTHETIC_ID", *missing]], on="SYNTHETIC_ID", how="left", validate="one_to_one")
        result[method, seed] = frame
    if not result:
        return result
    if any(field not in real for field in FIELDS):
        raise KeyError("The outer-test table lacks fields required for the 13-field diagnostics.")
    seed_sets = [{seed for method, seed in result if method == target} for target in METHODS]
    if not seed_sets[0] or seed_sets[0] != seed_sets[1]:
        raise ValueError("Full and Priv must provide the same generator seeds.")
    return result


def _save(fig, artifacts, name):
    fig.savefig(artifacts.figures / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _panel(ax, text):
    ax.text(-0.09, 1.04, text, transform=ax.transAxes, fontweight="bold", fontsize=14, va="bottom")


def figure2(real, frames, artifacts):
    rows = [{"method": method, "seed": seed, "feature": field, "hellinger_distance": numeric_hd(real[field], frame[field]) if field in NUMERICAL else categorical_hd(real[field], frame[field])} for (method, seed), frame in frames.items() for field in FIELDS]
    table = pd.DataFrame(rows)
    table.to_csv(artifacts.tables / "main_figure_hellinger.csv", index=False)
    fig, axes = plt.subplots(2, 1, figsize=(10, 11), gridspec_kw={"height_ratios": [1, 3], "hspace": 0.43})
    values = [table.loc[table.method.eq(method), "hellinger_distance"].to_numpy() for method in METHODS]
    boxes = axes[0].boxplot(values, tick_labels=METHODS, patch_artist=True, showfliers=False, widths=0.5, medianprops={"color": "black", "linewidth": 1.5})
    for patch, method in zip(boxes["boxes"], METHODS):
        patch.set_facecolor(COLORS[method])
        patch.set_alpha(0.75)
    rng = np.random.default_rng(521)
    for i, values_for_method in enumerate(values):
        axes[0].scatter(i + 1 + rng.uniform(-0.12, 0.12, len(values_for_method)), values_for_method, color="#444444", alpha=0.5, s=14)
    axes[0].set_ylabel("Hellinger distance")
    summary = table.groupby(["method", "feature"]).hellinger_distance.agg(["mean", "min", "max"])
    for offset, method in zip([-0.21, 0.21], METHODS):
        selected = summary.loc[method].loc[FIELDS]
        mean = selected["mean"].to_numpy()
        axes[1].barh(np.arange(len(FIELDS)) + offset, mean, height=0.42, color=COLORS[method], label=method, edgecolor="#333333", xerr=[mean - selected["min"], selected["max"] - mean], capsize=3)
    axes[1].set_yticks(np.arange(len(FIELDS)), LABELS)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Hellinger distance")
    axes[1].legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, frameon=False)
    for ax, label in zip(axes, "ab"):
        _panel(ax, label)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="x" if label == "b" else "y", color="#DDDDDD")
    fig.subplots_adjust(left=0.26, right=0.98, top=0.96)
    _save(fig, artifacts, "Figure_2")


def figure3(real, frames, artifacts):
    fields = [["OCC_HOUR", "LONG_WGS84", "LAT_WGS84", "SEG_MAP_DIST_M", "seg_pos"], ["MCI_CATEGO", "DIVISION", "PREMISES_T", "OCC_DOW", "OCC_MONTH"]]
    labels = [["Hour", "Lon.", "Lat.", "Map dist.", "Seg. pos."], ["MCI", "Division", "Premises", "Weekday", "Month"]]
    fig = plt.figure(figsize=(13, 15))
    grid = fig.add_gridspec(4, 4, width_ratios=[1, 1, 1, 0.055], hspace=0.75, wspace=0.3)
    cells = []
    for group, kind in enumerate(["Pearson", "CramersV"]):
        row = 2 * group
        matrices = {"Real held-out": association_matrix(real, fields[group], kind)}
        for method in METHODS:
            matrices[method] = np.mean([association_matrix(frame, fields[group], kind) for (m, seed), frame in frames.items() if m == method], axis=0)
        for name, matrix in matrices.items():
            for i, left in enumerate(fields[group]):
                for j, right in enumerate(fields[group]):
                    cells.append({"matrix": kind, "dataset": name, "row": left, "column": right, "value": matrix[i, j]})
        for col, (name, matrix) in enumerate(matrices.items()):
            ax = fig.add_subplot(grid[row, col])
            image = ax.imshow(matrix, vmin=-1 if group == 0 else 0, vmax=1, cmap="RdBu_r" if group == 0 else "Purples")
            ax.set_xticks(range(5), labels[group], rotation=45, ha="right")
            ax.set_yticks(range(5), labels[group] if col == 0 else [""] * 5)
            ax.set_title(name, pad=22)
            if col == 0:
                _panel(ax, "ac"[group])
        fig.colorbar(image, cax=fig.add_subplot(grid[row, 3]), label="Pearson r" if group == 0 else "Cramér’s V")
        differences = [matrices[method] - matrices["Real held-out"] for method in METHODS]
        limit = max(0.01, max(float(np.nanmax(np.abs(matrix))) for matrix in differences))
        for col, (method, matrix) in enumerate(zip(METHODS, differences), 1):
            ax = fig.add_subplot(grid[row + 1, col])
            image = ax.imshow(matrix, vmin=-limit, vmax=limit, cmap="RdBu_r")
            ax.set_xticks(range(5), labels[group], rotation=45, ha="right")
            ax.set_yticks(range(5), labels[group] if col == 1 else [""] * 5)
            ax.set_title(f"{method} vs Real", pad=22)
            if col == 1:
                _panel(ax, "bd"[group])
        fig.colorbar(image, cax=fig.add_subplot(grid[row + 1, 3]), label="Difference")
    pd.DataFrame(cells).to_csv(artifacts.tables / "main_figure_association_matrices.csv", index=False)
    fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.09)
    _save(fig, artifacts, "Figure_3")


def figure4(real, frames, artifacts):
    offences = sorted(real.OFFENCE.fillna("<NA>").astype(str).unique())
    levels = set(real.PREMISES_T.fillna("<NA>").astype(str))
    premises = [level for level in ["Outside", "Apartment", "Commercial", "House", "Other", "Transit", "Educational"] if level in levels]
    premises += sorted(levels - set(premises))
    years = pd.to_numeric(real.OCC_YEAR, errors="raise").astype(int)
    months = pd.MultiIndex.from_product([range(years.min(), years.max() + 1), range(1, 13)], names=["OCC_YEAR", "OCC_MONTH"])
    collections = {"Real held-out": [real], **{method: [frame for (m, seed), frame in frames.items() if m == method] for method in METHODS}}
    rows = []
    seasonal = {}
    for field, categories in [("OFFENCE", offences), ("PREMISES_T", premises)]:
        for name, group in collections.items():
            matrix = np.mean([seasonal_matrix(frame, field, categories) for frame in group], axis=0)
            seasonal[field, name] = matrix
            for i, category in enumerate(categories):
                for month in range(1, 13):
                    rows.append({"dataset": name, "field": field, "category": category, "month": month, "percentage": matrix[i, month - 1]})
    pd.DataFrame(rows).to_csv(artifacts.tables / "main_figure_seasonal_shares.csv", index=False)
    fig = plt.figure(figsize=(15, max(16, len(offences) * 0.27 + 7)))
    grid = fig.add_gridspec(4, 4, height_ratios=[max(5, len(offences) / 7), 1.8, 2.2, 2.2], width_ratios=[1, 1, 1, 0.045], hspace=0.48, wspace=0.18)
    for row, (field, categories) in enumerate([("OFFENCE", offences), ("PREMISES_T", premises)]):
        vmax = max(float(seasonal[field, name].max()) for name in collections)
        for col, name in enumerate(collections):
            ax = fig.add_subplot(grid[row, col])
            image = ax.imshow(seasonal[field, name], aspect="auto", vmin=0, vmax=vmax, cmap="YlOrRd")
            ax.set_yticks(range(len(categories)), categories if col == 0 else [""] * len(categories))
            ax.set_xticks(range(12), range(1, 13))
            ax.set_xlabel("Month")
            if row == 0:
                ax.set_title(name, pad=22)
            if col == 0:
                _panel(ax, "ab"[row])
        fig.colorbar(image, cax=fig.add_subplot(grid[row, 3]), label="Within-category monthly share (%)")
    curve_rows = []
    for row, field in [(2, "year_month"), (3, "OCC_HOUR")]:
        ax = fig.add_subplot(grid[row, :3])
        for name, group in collections.items():
            arrays = []
            for frame in group:
                series = frame.groupby(["OCC_YEAR", "OCC_MONTH"]).size().reindex(months, fill_value=0) if field == "year_month" else frame.groupby("OCC_HOUR").size().reindex(range(24), fill_value=0)
                arrays.append(series.to_numpy(float) / series.sum() * 100)
            arrays = np.asarray(arrays)
            x = np.arange(len(arrays[0]))
            mean = arrays.mean(axis=0)
            if len(arrays) > 1:
                ax.fill_between(x, arrays.min(axis=0), arrays.max(axis=0), color=COLORS[name], alpha=0.10)
            style = {"Real held-out": (3.2, "-"), METHODS[0]: (2.2, "--"), METHODS[1]: (1.4, ":")}[name]
            ax.plot(x, mean, color=COLORS[name], linewidth=style[0], linestyle=style[1], label=name)
            for i in x:
                curve_rows.append({"dataset": name, "field": field, "index": int(i), "mean_percentage": mean[i], "minimum_percentage": arrays[:, i].min(), "maximum_percentage": arrays[:, i].max()})
        if field == "year_month":
            ax.set_xticks(np.arange(0, len(months), 24), np.arange(years.min(), years.max() + 1, 2))
            ax.set_xlabel("Year")
        else:
            ax.set_xlabel("Hour of day")
        ax.set_ylabel("Share of events (%)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#DDDDDD")
        _panel(ax, "cd"[row - 2])
        ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.01), frameon=False)
    pd.DataFrame(curve_rows).to_csv(artifacts.tables / "main_figure_temporal_curves.csv", index=False)
    fig.subplots_adjust(left=0.31, right=0.95, bottom=0.06, top=0.97)
    _save(fig, artifacts, "Figure_4")


def generate_reported_figures(real, synthetic, artifacts, cfg=None):
    from .boundary_figure import generate_boundary_figure

    frames = diagnostic_frames(real, synthetic)
    if frames:
        with plt.rc_context({"font.size": 11, "axes.labelsize": 12, "legend.fontsize": 11}):
            figure2(real, frames, artifacts)
            figure3(real, frames, artifacts)
            figure4(real, frames, artifacts)
            generate_boundary_figure(real, frames, artifacts, cfg)
