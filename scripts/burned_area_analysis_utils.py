"""
Cruzamento dos mapas classificados com as áreas ardidas de 2025.
"""

import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.windows import Window

from classification_utils import CLASS_LABELS


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


def burned_area_by_class(
    class_raster,
    burned_raster,
    positive_value=1,
    **metadata
):
    """Calcula a área ardida total e a sua distribuição pelas classes."""

    burned_by_class = np.zeros(6, dtype="int64")
    model_cells = 0
    burned_total = 0
    burned_modelled = 0

    with rio.open(class_raster) as cls_src, rio.open(burned_raster) as ba_src:
        _check_grid(cls_src, ba_src)
        cell_area_m2 = abs(cls_src.transform.a * cls_src.transform.e)

        for window in _windows(cls_src):
            classes = cls_src.read(1, window=window)
            burned = ba_src.read(1, window=window)
            valid_model = (classes >= 1) & (classes <= 5)
            is_burned = burned == positive_value

            model_cells += int(valid_model.sum())
            burned_total += int(is_burned.sum())
            burned_modelled += int((valid_model & is_burned).sum())

            if (valid_model & is_burned).any():
                burned_by_class += np.bincount(
                    classes[valid_model & is_burned],
                    minlength=6
                )[:6]

    support = pd.DataFrame([{
        **metadata,
        "modelled_area_km2": model_cells * cell_area_m2 / 1_000_000,
        "burned_area_total_ha": burned_total * cell_area_m2 / 10_000,
        "burned_area_modelled_ha": burned_modelled * cell_area_m2 / 10_000,
        "burned_cells_total": burned_total,
        "burned_cells_modelled": burned_modelled,
        "burned_coverage_pct": (
            100.0 * burned_modelled / burned_total
            if burned_total
            else np.nan
        )
    }])

    rows = []

    for class_code in range(1, 6):
        cells = int(burned_by_class[class_code])

        rows.append({
            **metadata,
            "class_code": class_code,
            "class_label": CLASS_LABELS[class_code],
            "burned_cells": cells,
            "burned_area_ha": cells * cell_area_m2 / 10_000,
            "burned_area_pct": (
                100.0 * cells / burned_modelled
                if burned_modelled
                else np.nan
            )
        })

    return support, pd.DataFrame(rows)


def burned_area_top_classes(distribution):
    """Resume a área ardida na classe 5 e nas classes 4+5."""

    keys = [
        "pilot",
        "map_id",
        "scenario",
        "product",
        "reference"
    ]
    rows = []

    for values, group in distribution.groupby(keys, dropna=False):
        record = dict(zip(keys, values))
        very_high = group.loc[group["class_code"] == 5]
        high_very_high = group.loc[group["class_code"].isin([4, 5])]

        record.update({
            "very_high_burned_area_ha": very_high["burned_area_ha"].sum(),
            "very_high_burned_area_pct": very_high["burned_area_pct"].sum(),
            "high_very_high_burned_area_ha": (
                high_very_high["burned_area_ha"].sum()
            ),
            "high_very_high_burned_area_pct": (
                high_very_high["burned_area_pct"].sum()
            )
        })
        rows.append(record)

    return pd.DataFrame(rows)


def compare_binary_rasters(raster_a, raster_b, **metadata):
    """Compara as máscaras positivas de dois rasters binários alinhados."""

    positive_a = positive_b = common = only_a = only_b = 0

    with rio.open(raster_a) as src_a, rio.open(raster_b) as src_b:
        _check_grid(src_a, src_b)

        for window in _windows(src_a):
            a = src_a.read(1, window=window)
            b = src_b.read(1, window=window)
            mask_a = _valid_binary(a, src_a.nodata) & (a > 0)
            mask_b = _valid_binary(b, src_b.nodata) & (b > 0)

            positive_a += int(mask_a.sum())
            positive_b += int(mask_b.sum())
            common += int((mask_a & mask_b).sum())
            only_a += int((mask_a & ~mask_b).sum())
            only_b += int((~mask_a & mask_b).sum())

    return pd.DataFrame([{
        **metadata,
        "positive_cells_a": positive_a,
        "positive_cells_b": positive_b,
        "common_positive_cells": common,
        "only_a_cells": only_a,
        "only_b_cells": only_b,
        "identical_positive_mask": only_a == 0 and only_b == 0,
        "raster_a": str(raster_a),
        "raster_b": str(raster_b)
    }])


def _valid_binary(array, nodata):
    valid = np.isfinite(array)

    if nodata is not None:
        try:
            nodata_is_nan = np.isnan(nodata)
        except TypeError:
            nodata_is_nan = False

        if not nodata_is_nan:
            valid &= array != nodata

    return valid
