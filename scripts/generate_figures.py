import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data_io import (
    compute_exposures,
    count_injuries_by,
    ensure_output_dir,
    load_injury_record,
    load_play_list,
    load_player_track,
)
from src.viz import (
    plot_injury_rate_by_surface,
    plot_severity_distribution,
    plot_position_irr,
    plot_field_heatmap,
)


def synthesize_demo_data(seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    # PlayList-like exposures
    positions = ["RB", "WR", "DB", "LB", "OL", "DL", "QB", "TE"]
    surfaces = ["grass", "synthetic"]
    n_rows = 200_000
    play_list = pd.DataFrame(
        {
            "PlayKey": np.arange(n_rows),
            "PositionGroup": rng.choice(positions, size=n_rows, p=[0.12, 0.16, 0.18, 0.14, 0.14, 0.12, 0.08, 0.06]),
            "Surface": rng.choice(surfaces, size=n_rows, p=[0.55, 0.45]),
        }
    )
    # InjuryRecord-like: generate injuries with slightly higher risk on synthetic
    base_rate = 0.00045  # per play
    surface_rr = {"grass": 1.0, "synthetic": 1.25}
    probs = (
        play_list["Surface"].map(surface_rr).values * base_rate
    )
    injuries_mask = rng.random(n_rows) < probs
    injured_keys = play_list.loc[injuries_mask, "PlayKey"].values
    injury_record = pd.DataFrame(
        {
            "PlayKey": injured_keys,
            "Surface": play_list.loc[injuries_mask, "Surface"].values,
            "PositionGroup": play_list.loc[injuries_mask, "PositionGroup"].values,
            "DaysMissed": rng.integers(1, 30, size=injuries_mask.sum()),
        }
    )

    # Tracking-like: random scatter on a 120x53.3 field
    n_track = 100_000
    player_track = pd.DataFrame(
        {
            "PlayKey": rng.choice(play_list["PlayKey"], size=n_track),
            "x": rng.uniform(0, 120, size=n_track),
            "y": rng.uniform(0, 53.3, size=n_track),
        }
    )
    return injury_record, play_list, player_track


def main(
    injury_csv: Optional[str],
    plays_csv: Optional[str],
    track_csv: Optional[str],
    out_dir: str,
    demo: bool,
):
    ensure_output_dir(out_dir)

    if demo:
        injury_record, play_list, player_track = synthesize_demo_data()
    else:
        if not (injury_csv and plays_csv):
            raise SystemExit("Please provide --injury-csv and --plays-csv, or use --demo")
        injury_record = load_injury_record(injury_csv)
        play_list = load_play_list(plays_csv)
        player_track = load_player_track(track_csv) if track_csv else pd.DataFrame()

    # Compute rates by surface
    by_surface_inj = count_injuries_by(injury_record, group_cols=("Surface",))
    by_surface_exp = compute_exposures(play_list, group_cols=("Surface",))
    rates = by_surface_inj.merge(by_surface_exp, on=["Surface"], how="inner")

    fig1 = plot_injury_rate_by_surface(rates, out_dir)

    # Merge to get severity by surface
    inj_playlist = play_list.merge(
        injury_record[["PlayKey", "Surface", "DaysMissed"]], on="PlayKey", how="inner"
    )
    fig2 = plot_severity_distribution(inj_playlist, out_dir)

    # Position-specific IRR
    inj_by_pos_surface = count_injuries_by(injury_record, group_cols=("PositionGroup", "Surface"))
    exp_by_pos_surface = compute_exposures(play_list, group_cols=("PositionGroup", "Surface"))
    counts = inj_by_pos_surface.merge(exp_by_pos_surface, on=["PositionGroup", "Surface"], how="inner")
    fig3 = plot_position_irr(counts, out_dir)

    # Field heatmap (if tracking present)
    fig4 = plot_field_heatmap(player_track if isinstance(player_track, pd.DataFrame) else pd.DataFrame(), out_dir)

    print("Saved figures:")
    for f in [fig1, fig2, fig3, fig4]:
        if f is not None:
            print(f" - {f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate portfolio-ready figures for NFL Ground Game analysis")
    parser.add_argument("--injury-csv", type=str, default=None)
    parser.add_argument("--plays-csv", type=str, default=None)
    parser.add_argument("--track-csv", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="reports/figures")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    main(args.injury_csv, args.plays_csv, args.track_csv, args.out_dir, args.demo)