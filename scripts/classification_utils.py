"""
Classificação percentílica regional e estatísticas por classe.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.windows import Window


CLASS_LABELS = {
    1: "Muito Baixa",
    2: "Baixa",
    3: "Moderada",
    4: "Alta",
    5: "Muito Alta"
}


CLASS_COLORMAP = {
    0: (255, 255, 255, 0),
    1: (26, 150, 65, 255),
    2: (166, 217, 106, 255),
    3: (255, 255, 191, 255),
    4: (253, 174, 97, 255),
    5: (215, 25, 28, 255)
}


def _windows(src, block_size=1024):
    for row in range(0, src.height, block_size):
        height = min(block_size, src.height - row)

        for col in range(0, src.width, block_size):
            width = min(block_size, src.width - col)
            yield Window(col, row, width, height)


def _valid(array, nodata):
    valid = np.isfinite(array)

    if nodata is not None:
        try:
            nodata_is_nan = np.isnan(nodata)
        except TypeError:
            nodata_is_nan = False

        if not nodata_is_nan:
            valid &= array != nodata

    return valid


def percentile_breaks(
    raster,
    percentiles=(0.2, 0.4, 0.6, 0.8),
    n_bins=100000,
    block_size=1024
):
    """
    Calcula limites percentílicos regionais por histograma.

    Valores iguais recebem sempre a mesma classe. Por esse motivo,
    as classes podem não ocupar exactamente 20% da região.
    """

    raster = Path(raster)

    if not raster.exists():
        raise FileNotFoundError(f"Raster não encontrado: {raster}")

    vmin = np.inf
    vmax = -np.inf
    n_valid = 0

    with rio.open(raster) as src:
        for window in _windows(src, block_size):
            array = src.read(1, window=window).astype("float64")
            values = array[_valid(array, src.nodata)]

            if not values.size:
                continue

            vmin = min(vmin, float(values.min()))
            vmax = max(vmax, float(values.max()))
            n_valid += int(values.size)

    if n_valid == 0:
        raise ValueError(f"Raster sem valores válidos: {raster}")

    if vmin == vmax:
        raise ValueError(f"Raster constante: {raster}")

    edges = np.linspace(vmin, vmax, n_bins + 1, dtype="float64")
    histogram = np.zeros(n_bins, dtype="int64")

    with rio.open(raster) as src:
        for window in _windows(src, block_size):
            array = src.read(1, window=window).astype("float64")
            values = array[_valid(array, src.nodata)]

            if values.size:
                counts, _ = np.histogram(values, bins=edges)
                histogram += counts

    cumulative = np.cumsum(histogram)
    breaks = []

    for percentile in percentiles:
        target = percentile * n_valid
        index = int(np.searchsorted(cumulative, target, side="left"))
        index = min(index, n_bins - 1)
        breaks.append(float(edges[index + 1]))

    return np.asarray(breaks, dtype="float64")


def classify_raster(raster, breaks, output, block_size=1024):
    """Classifica um raster contínuo em cinco classes relativas."""

    raster = Path(raster)
    output = Path(output)
    breaks = np.asarray(breaks, dtype="float64")

    if breaks.size != 4:
        raise ValueError("São necessários quatro limites de classe.")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)

    counts = np.zeros(6, dtype="int64")

    with rio.open(raster) as src:
        profile = src.profile.copy()
        profile.update(
            dtype="uint8",
            count=1,
            nodata=0,
            compress="lzw",
            tiled=True,
            blockxsize=256,
            blockysize=256,
            BIGTIFF="IF_SAFER"
        )

        with rio.open(output, "w", **profile) as dst:
            for window in _windows(src, block_size):
                array = src.read(1, window=window).astype("float64")
                valid = _valid(array, src.nodata)
                classes = np.zeros(array.shape, dtype="uint8")

                classes[valid] = (
                    np.searchsorted(
                        breaks,
                        array[valid],
                        side="left"
                    ) + 1
                ).astype("uint8")

                dst.write(classes, 1, window=window)
                counts += np.bincount(
                    classes.ravel(),
                    minlength=6
                )[:6]

            dst.write_colormap(1, CLASS_COLORMAP)

    return counts


def breaks_table(map_id, raster, breaks, counts=None):
    """Cria a tabela de limites e ocupação das classes."""

    lower = [-np.inf, *breaks]
    upper = [*breaks, np.inf]
    total = None if counts is None else counts[1:6].sum()
    rows = []

    for class_code in range(1, 6):
        pixels = None if counts is None else int(counts[class_code])
        percentage = (
            None
            if total in [None, 0]
            else 100.0 * pixels / total
        )

        rows.append({
            "map_id": map_id,
            "raster": str(raster),
            "class_code": class_code,
            "class_label": CLASS_LABELS[class_code],
            "lower_bound": lower[class_code - 1],
            "upper_bound": upper[class_code - 1],
            "pixels_region": pixels,
            "percentage_region": percentage,
            "method": "regional_percentiles_20_40_60_80"
        })

    return pd.DataFrame(rows)


def class_area_table(class_raster, **metadata):
    """Calcula a área e a percentagem ocupada por cada classe."""

    counts = np.zeros(6, dtype="int64")

    with rio.open(class_raster) as src:
        cell_area_m2 = abs(src.transform.a * src.transform.e)

        for window in _windows(src):
            array = src.read(1, window=window)
            counts += np.bincount(
                array.ravel(),
                minlength=6
            )[:6]

    valid_cells = int(counts[1:6].sum())
    rows = []

    for class_code in range(1, 6):
        cells = int(counts[class_code])
        area_km2 = cells * cell_area_m2 / 1_000_000

        rows.append({
            **metadata,
            "class_code": class_code,
            "class_label": CLASS_LABELS[class_code],
            "cells": cells,
            "area_km2": area_km2,
            "percentage_modelled": (
                100.0 * cells / valid_cells
                if valid_cells
                else np.nan
            )
        })

    return pd.DataFrame(rows)
