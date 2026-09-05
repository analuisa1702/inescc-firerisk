"""
Comparação espacial de mapas classificados.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.windows import Window


DIFF_COLORMAP = {
    0: (255, 255, 255, 0),
    1: (33, 102, 172, 255),
    2: (146, 197, 222, 255),
    3: (247, 247, 247, 255),
    4: (244, 165, 130, 255),
    5: (178, 24, 43, 255)
}


AGREEMENT_COLORMAP = {
    0: (255, 255, 255, 0),
    1: (220, 220, 220, 255),
    2: (67, 147, 195, 255),
    3: (244, 109, 67, 255),
    4: (165, 0, 38, 255)
}


def _windows(src, block_size=1024):
    for row in range(0, src.height, block_size):
        height = min(block_size, src.height - row)

        for col in range(0, src.width, block_size):
            width = min(block_size, src.width - col)
            yield Window(col, row, width, height)


def _check_grid(src_a, src_b):
    if src_a.shape != src_b.shape:
        raise ValueError("Os rasters têm dimensões diferentes.")

    if src_a.crs != src_b.crs:
        raise ValueError("Os rasters têm CRS diferentes.")

    if not src_a.transform.almost_equals(
        src_b.transform,
        precision=1e-8
    ):
        raise ValueError("Os rasters têm transformações diferentes.")


def compare_classes(raster_a, raster_b, output, **metadata):
    """Cria a matriz de transição, indicadores e mapa de diferença."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)

    transition = np.zeros((5, 5), dtype="int64")
    same = up = down = up_two = down_two = 0
    abs_sum = signed_sum = valid_count = 0

    with rio.open(raster_a) as src_a, rio.open(raster_b) as src_b:
        _check_grid(src_a, src_b)

        profile = src_a.profile.copy()
        profile.update(
            dtype="uint8",
            nodata=0,
            compress="lzw",
            BIGTIFF="IF_SAFER"
        )

        with rio.open(output, "w", **profile) as dst:
            for window in _windows(src_a):
                a = src_a.read(1, window=window)
                b = src_b.read(1, window=window)
                valid = (a >= 1) & (a <= 5) & (b >= 1) & (b <= 5)
                grouped = np.zeros(a.shape, dtype="uint8")

                if valid.any():
                    av = a[valid].astype("int16")
                    bv = b[valid].astype("int16")
                    diff = bv - av

                    index = (av - 1) * 5 + (bv - 1)
                    transition += np.bincount(
                        index,
                        minlength=25
                    ).reshape(5, 5)

                    same += int((diff == 0).sum())
                    up += int((diff > 0).sum())
                    down += int((diff < 0).sum())
                    up_two += int((diff >= 2).sum())
                    down_two += int((diff <= -2).sum())
                    abs_sum += float(np.abs(diff).sum())
                    signed_sum += float(diff.sum())
                    valid_count += int(diff.size)

                    values = np.full(diff.shape, 3, dtype="uint8")
                    values[diff <= -2] = 1
                    values[diff == -1] = 2
                    values[diff == 1] = 4
                    values[diff >= 2] = 5
                    grouped[valid] = values

                dst.write(grouped, 1, window=window)

            dst.write_colormap(1, DIFF_COLORMAP)

    summary = pd.DataFrame([{
        **metadata,
        "valid_cells": valid_count,
        "same_pct": 100.0 * same / valid_count if valid_count else np.nan,
        "up_pct": 100.0 * up / valid_count if valid_count else np.nan,
        "down_pct": 100.0 * down / valid_count if valid_count else np.nan,
        "up_two_or_more_pct": (
            100.0 * up_two / valid_count
            if valid_count
            else np.nan
        ),
        "down_two_or_more_pct": (
            100.0 * down_two / valid_count
            if valid_count
            else np.nan
        ),
        "mean_absolute_class_difference": (
            abs_sum / valid_count
            if valid_count
            else np.nan
        ),
        "mean_signed_class_difference": (
            signed_sum / valid_count
            if valid_count
            else np.nan
        ),
        "difference_raster": str(output)
    }])

    matrix = pd.DataFrame(
        transition,
        index=[f"A_{value}" for value in range(1, 6)],
        columns=[f"B_{value}" for value in range(1, 6)]
    )

    return summary, matrix


def priority_agreement(raster_a, raster_b, output, **metadata):
    """Compara as classes prioritárias Alta e Muito Alta."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)

    counts = np.zeros(5, dtype="int64")

    with rio.open(raster_a) as src_a, rio.open(raster_b) as src_b:
        _check_grid(src_a, src_b)
        cell_area_m2 = abs(src_a.transform.a * src_a.transform.e)

        profile = src_a.profile.copy()
        profile.update(
            dtype="uint8",
            nodata=0,
            compress="lzw",
            BIGTIFF="IF_SAFER"
        )

        with rio.open(output, "w", **profile) as dst:
            for window in _windows(src_a):
                a = src_a.read(1, window=window)
                b = src_b.read(1, window=window)
                valid = (a >= 1) & (a <= 5) & (b >= 1) & (b <= 5)
                result = np.zeros(a.shape, dtype="uint8")

                priority_a = valid & (a >= 4)
                priority_b = valid & (b >= 4)

                result[valid & ~priority_a & ~priority_b] = 1
                result[priority_a & ~priority_b] = 2
                result[~priority_a & priority_b] = 3
                result[priority_a & priority_b] = 4

                counts += np.bincount(
                    result.ravel(),
                    minlength=5
                )[:5]

                dst.write(result, 1, window=window)

            dst.write_colormap(1, AGREEMENT_COLORMAP)

    neither, only_a, only_b, both = [int(v) for v in counts[1:5]]
    priority_a_count = only_a + both
    priority_b_count = only_b + both

    summary = pd.DataFrame([{
        **metadata,
        "neither_km2": neither * cell_area_m2 / 1_000_000,
        "only_a_km2": only_a * cell_area_m2 / 1_000_000,
        "only_b_km2": only_b * cell_area_m2 / 1_000_000,
        "common_km2": both * cell_area_m2 / 1_000_000,
        "priority_a_km2": priority_a_count * cell_area_m2 / 1_000_000,
        "priority_b_km2": priority_b_count * cell_area_m2 / 1_000_000,
        "persistence_pct_a": (
            100.0 * both / priority_a_count
            if priority_a_count
            else np.nan
        ),
        "agreement_raster": str(output)
    }])

    return summary


def priority_frequency(rasters, output, threshold=4):
    """Conta em quantos cenários cada célula é Alta ou Muito Alta."""

    rasters = [Path(path) for path in rasters]
    output = Path(output)

    if not rasters:
        raise ValueError("A lista de rasters está vazia.")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)

    sources = [rio.open(path) for path in rasters]

    try:
        for src in sources[1:]:
            _check_grid(sources[0], src)

        profile = sources[0].profile.copy()
        profile.update(
            dtype="uint8",
            nodata=255,
            compress="lzw",
            BIGTIFF="IF_SAFER"
        )

        with rio.open(output, "w", **profile) as dst:
            for window in _windows(sources[0]):
                arrays = [
                    src.read(1, window=window)
                    for src in sources
                ]

                valid = np.ones(arrays[0].shape, dtype=bool)

                for array in arrays:
                    valid &= (array >= 1) & (array <= 5)

                frequency = np.full(
                    arrays[0].shape,
                    255,
                    dtype="uint8"
                )

                if valid.any():
                    stack = np.stack(arrays)
                    frequency[valid] = (
                        stack[:, valid] >= threshold
                    ).sum(axis=0).astype("uint8")

                dst.write(frequency, 1, window=window)
    finally:
        for src in sources:
            src.close()

    return str(output)


def seasonal_effect_summary(changes, agreement):
    """Cria uma síntese direta da alteração territorial provocada pelo SSR."""

    keys = [
        "pilot",
        "comparison_id",
        "effect",
        "scenario_a",
        "scenario_b",
        "product",
        "map_a",
        "map_b"
    ]

    summary = changes.merge(
        agreement,
        on=keys,
        how="inner",
        suffixes=("_change", "_priority")
    )

    valid_area = (
        summary["neither_km2"]
        + summary["only_a_km2"]
        + summary["only_b_km2"]
        + summary["common_km2"]
    )

    summary["priority_change_km2"] = (
        summary["priority_b_km2"]
        - summary["priority_a_km2"]
    )
    summary["priority_a_pct"] = (
        100.0 * summary["priority_a_km2"] / valid_area
    )
    summary["priority_b_pct"] = (
        100.0 * summary["priority_b_km2"] / valid_area
    )
    summary["priority_change_pp"] = (
        summary["priority_b_pct"]
        - summary["priority_a_pct"]
    )

    columns = [
        *keys,
        "same_pct",
        "up_pct",
        "down_pct",
        "mean_absolute_class_difference",
        "mean_signed_class_difference",
        "priority_a_km2",
        "priority_b_km2",
        "priority_change_km2",
        "priority_a_pct",
        "priority_b_pct",
        "priority_change_pp",
        "persistence_pct_a"
    ]

    return summary[columns]
