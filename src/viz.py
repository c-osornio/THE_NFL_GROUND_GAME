from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


sns.set_theme(style="whitegrid", context="talk")


def _save(fig: plt.Figure, out_dir: str | Path, name: str) -> Path:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir) / f"{name}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_injury_rate_by_surface(df_rates: pd.DataFrame, out_dir: str | Path) -> Path:
    # df_rates columns: Surface, injuries, exposures
    rates = df_rates.copy()
    rates["rate_per_1k"] = 1000.0 * rates["injuries"] / rates["exposures"].clip(lower=1)
    # 95% CI using normal approximation on log rate
    z = 1.959963984540054
    se_log_rate = np.sqrt(1.0 / rates["injuries"].clip(lower=1))
    rates["lcl"] = np.exp(np.log(rates["rate_per_1k"].replace(0, np.nan)) - z * se_log_rate)
    rates["ucl"] = np.exp(np.log(rates["rate_per_1k"].replace(0, np.nan)) + z * se_log_rate)

    fig, ax = plt.subplots(figsize=(8, 6))
    order = rates.sort_values("rate_per_1k")["Surface"].tolist()
    sns.barplot(
        data=rates,
        x="Surface",
        y="rate_per_1k",
        order=order,
        palette="Set2",
        ax=ax,
    )
    # error bars
    for i, row in rates.set_index("Surface").loc[order].reset_index().iterrows():
        ax.errorbar(i, row["rate_per_1k"], yerr=[[row["rate_per_1k"] - row["lcl"]], [row["ucl"] - row["rate_per_1k"]]], fmt="none", ecolor="black", capsize=4)
    ax.set_title("Injury Rate by Surface (per 1,000 exposures)")
    ax.set_xlabel("Surface")
    ax.set_ylabel("Rate per 1,000")
    return _save(fig, out_dir, "figure_1_injury_rate_by_surface")


def plot_severity_distribution(inj_playlist: pd.DataFrame, out_dir: str | Path) -> Optional[Path]:
    if "DaysMissed" not in inj_playlist.columns:
        return None
    df = inj_playlist.copy()
    # coerce to numeric; drop missing
    df["DaysMissed"] = pd.to_numeric(df["DaysMissed"], errors="coerce")
    df = df.dropna(subset=["DaysMissed"])  # type: ignore[arg-type]
    if "Surface" not in df.columns:
        df["Surface"] = "Unknown"

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.ecdfplot(data=df, x="DaysMissed", hue="Surface", ax=ax)
    ax.set_title("Severity Distribution by Surface (ECDF of Days Missed)")
    ax.set_xlabel("Days Missed")
    ax.set_ylabel("Cumulative proportion")
    ax.legend(title="Surface")
    return _save(fig, out_dir, "figure_2_severity_distribution")


def plot_position_irr(df_counts: pd.DataFrame, out_dir: str | Path) -> Optional[Path]:
    # df_counts must have columns: PositionGroup, Surface, injuries, exposures
    required = {"PositionGroup", "Surface", "injuries", "exposures"}
    if not required.issubset(df_counts.columns):
        return None
    # compute rates and IRRs (Surface baseline = grass if present)
    pos_groups = []
    irr_values = []
    for pos, g in df_counts.groupby("PositionGroup", observed=True):
        g = g.set_index("Surface")
        if not {"grass", "synthetic"}.issubset(g.index.astype(str).str.lower()):
            continue
        try:
            grass = g.loc[g.index.astype(str).str.lower() == "grass"].iloc[0]
            turf = g.loc[g.index.astype(str).str.lower() == "synthetic"].iloc[0]
        except Exception:
            continue
        rate_grass = turf_rate = None  # placeholders
        rate_grass = 1000.0 * grass["injuries"] / max(grass["exposures"], 1)
        turf_rate = 1000.0 * turf["injuries"] / max(turf["exposures"], 1)
        irr = (turf["injuries"] / max(turf["exposures"], 1)) / (grass["injuries"] / max(grass["exposures"], 1)) if grass["injuries"] and turf["injuries"] else np.nan
        pos_groups.append(pos)
        irr_values.append((irr, turf_rate, rate_grass))

    if not pos_groups:
        return None

    plot_df = pd.DataFrame(
        {
            "PositionGroup": pos_groups,
            "IRR_turf_vs_grass": [v[0] for v in irr_values],
            "Rate_turf": [v[1] for v in irr_values],
            "Rate_grass": [v[2] for v in irr_values],
        }
    ).dropna()

    fig, ax = plt.subplots(figsize=(10, 6))
    order = plot_df.sort_values("IRR_turf_vs_grass", ascending=False)["PositionGroup"]
    sns.barplot(data=plot_df, x="IRR_turf_vs_grass", y="PositionGroup", order=order, palette="crest", ax=ax)
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1)
    ax.set_title("Position-specific IRR: Turf vs Grass")
    ax.set_xlabel("Injury Rate Ratio (IRR)")
    ax.set_ylabel("Position Group")
    return _save(fig, out_dir, "figure_3_position_irr")


def plot_field_heatmap(track_df: pd.DataFrame, out_dir: str | Path) -> Optional[Path]:
    # Requires columns x, y; will KDE all points if present
    if not {"x", "y"}.issubset(track_df.columns):
        return None
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.kdeplot(data=track_df, x="x", y="y", fill=True, cmap="magma", levels=50, thresh=0.05)
    ax.set_title("Field Spatial Density (All Points)")
    ax.set_xlabel("Field X (yards)")
    ax.set_ylabel("Field Y (yards)")
    return _save(fig, out_dir, "figure_4_field_heatmap")