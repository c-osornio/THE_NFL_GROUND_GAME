import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


def ensure_output_dir(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def _downcast_integer_series(series: pd.Series) -> pd.Series:
    if series.isna().any():
        return series.astype(pd.Int32Dtype())
    min_val = series.min()
    max_val = series.max()
    if min_val >= 0:
        if max_val <= np.iinfo(np.uint8).max:
            return series.astype(np.uint8)
        if max_val <= np.iinfo(np.uint16).max:
            return series.astype(np.uint16)
        if max_val <= np.iinfo(np.uint32).max:
            return series.astype(np.uint32)
    else:
        if np.iinfo(np.int8).min <= min_val and max_val <= np.iinfo(np.int8).max:
            return series.astype(np.int8)
        if np.iinfo(np.int16).min <= min_val and max_val <= np.iinfo(np.int16).max:
            return series.astype(np.int16)
        if np.iinfo(np.int32).min <= min_val and max_val <= np.iinfo(np.int32).max:
            return series.astype(np.int32)
    return series


def reduce_mem_usage(df: pd.DataFrame, category_threshold: float = 0.5) -> pd.DataFrame:
    optimized = df.copy()
    for col in optimized.columns:
        col_data = optimized[col]
        if pd.api.types.is_integer_dtype(col_data):
            optimized[col] = _downcast_integer_series(col_data)
        elif pd.api.types.is_float_dtype(col_data):
            optimized[col] = col_data.astype(np.float32)
        elif pd.api.types.is_object_dtype(col_data):
            # Convert to category if reasonable cardinality
            num_unique = col_data.nunique(dropna=True)
            if 0 < num_unique < category_threshold * len(col_data):
                optimized[col] = col_data.astype("category")
    return optimized


def safe_read_csv(
    file_path: str | Path,
    usecols: Optional[Iterable[str]] = None,
    dtype: Optional[Dict[str, str]] = None,
    parse_dates: Optional[Iterable[str]] = None,
    chunksize: Optional[int] = None,
) -> pd.DataFrame | Iterable[pd.DataFrame]:
    file_path = str(file_path)
    engine = "pyarrow" if parse_dates is None else None
    if chunksize is not None:
        return pd.read_csv(
            file_path,
            usecols=usecols,
            dtype=dtype,
            parse_dates=parse_dates,
            chunksize=chunksize,
            engine=engine,
        )
    return pd.read_csv(
        file_path,
        usecols=usecols,
        dtype=dtype,
        parse_dates=parse_dates,
        engine=engine,
    )


def load_injury_record(path: str | Path) -> pd.DataFrame:
    df = safe_read_csv(path)
    df = reduce_mem_usage(df)
    # Common normalizations
    for cat_col in [
        "BodyPart",
        "Surface",
        "StadiumType",
        "RosterPosition",
        "PositionGroup",
    ]:
        if cat_col in df.columns and not pd.api.types.is_categorical_dtype(df[cat_col]):
            df[cat_col] = df[cat_col].astype("category")
    return df


def load_play_list(path: str | Path) -> pd.DataFrame:
    df = safe_read_csv(path)
    df = reduce_mem_usage(df)
    for cat_col in [
        "RosterPosition",
        "PositionGroup",
        "StadiumType",
        "FieldType",
        "PlayKey",
        "GameID",
    ]:
        if cat_col in df.columns and not pd.api.types.is_categorical_dtype(df[cat_col]):
            df[cat_col] = df[cat_col].astype("category")
    return df


def load_player_track(path: str | Path, chunksize: Optional[int] = None) -> pd.DataFrame | Iterable[pd.DataFrame]:
    usecols = None
    df_or_iter = safe_read_csv(path, usecols=usecols, chunksize=chunksize)
    if chunksize is None:
        df = reduce_mem_usage(df_or_iter)  # type: ignore[arg-type]
        for cat_col in ["PlayKey", "event"]:
            if cat_col in df.columns and not pd.api.types.is_categorical_dtype(df[cat_col]):
                df[cat_col] = df[cat_col].astype("category")
        return df
    return df_or_iter


def compute_exposures(
    play_list: pd.DataFrame,
    group_cols: Tuple[str, ...] = ("Surface",),
) -> pd.DataFrame:
    missing = [c for c in group_cols if c not in play_list.columns]
    if missing:
        raise ValueError(f"Missing grouping columns in play_list: {missing}")
    exposures = (
        play_list.assign(exposure=1).groupby(list(group_cols), observed=True)["exposure"].sum().reset_index()
    )
    exposures = exposures.rename(columns={"exposure": "exposures"})
    return exposures


def count_injuries_by(
    injury_df: pd.DataFrame,
    group_cols: Tuple[str, ...] = ("Surface",),
) -> pd.DataFrame:
    missing = [c for c in group_cols if c not in injury_df.columns]
    if missing:
        raise ValueError(f"Missing grouping columns in injury_df: {missing}")
    counts = injury_df.assign(injury=1).groupby(list(group_cols), observed=True)["injury"].sum().reset_index()
    counts = counts.rename(columns={"injury": "injuries"})
    return counts


def merge_injury_playlist(
    play_list: pd.DataFrame,
    injury_df: pd.DataFrame,
    on: Tuple[str, ...] = ("PlayKey",),
) -> pd.DataFrame:
    for key in on:
        if key not in play_list.columns or key not in injury_df.columns:
            raise ValueError(f"Missing merge key '{key}' in provided DataFrames")
    merged = play_list.merge(injury_df, on=list(on), how="inner", validate="m:1")
    return merged