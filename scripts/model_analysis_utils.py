"""
Interpretação dos LRi, importância RF e incerteza entre réplicas.
"""

from pathlib import Path
import gc

import joblib
import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.windows import Window

from analysis_config import (
    ELEVATION_CLASSES,
    FEATURE_LABELS,
    LULC_CLASSES,
    LULC_MODELLED_CLASSES,
    SLOPE_CLASSES
)


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


def _check_sources(sources):
    reference = sources[0]

    for src in sources[1:]:
        if src.shape != reference.shape:
            raise ValueError("As variáveis têm dimensões diferentes.")

        if src.crs != reference.crs:
            raise ValueError("As variáveis têm CRS diferentes.")

        if not src.transform.almost_equals(
            reference.transform,
            precision=1e-8
        ):
            raise ValueError("As variáveis têm transformações diferentes.")


def raster_valid_cell_count(raster):
    """Conta as células válidas de um raster por blocos."""

    count = 0

    with rio.open(raster) as src:
        for window in _windows(src):
            array = src.read(1, window=window)
            count += int(_valid(array, src.nodata).sum())

    return count


def lri_by_class(class_raster, lri_raster, variable, **metadata):
    """Relaciona cada classe explicativa com o respetivo valor LRi."""

    labels = (
        ELEVATION_CLASSES
        if variable == "dem"
        else SLOPE_CLASSES
    )
    counts = {}

    with rio.open(class_raster) as cls_src, rio.open(lri_raster) as lri_src:
        _check_sources([cls_src, lri_src])

        for window in _windows(cls_src):
            classes = cls_src.read(1, window=window)
            lri = lri_src.read(1, window=window).astype("float64")
            valid = (
                _valid(classes, cls_src.nodata)
                & _valid(lri, lri_src.nodata)
            )

            if not valid.any():
                continue

            pairs = np.column_stack([
                classes[valid].astype("int32"),
                np.round(lri[valid], 8)
            ])

            values, frequency = np.unique(
                pairs,
                axis=0,
                return_counts=True
            )

            for value, n_cells in zip(values, frequency):
                key = (int(value[0]), float(value[1]))
                counts[key] = counts.get(key, 0) + int(n_cells)

    rows = []

    for (class_code, lri), cells in sorted(counts.items()):
        rows.append({
            **metadata,
            "variable": variable,
            "variable_label": (
                "Altitude" if variable == "dem" else "Declive"
            ),
            "class_code": class_code,
            "class_label": labels.get(class_code, "Classe não definida"),
            "lri": lri,
            "cells": cells
        })

    return pd.DataFrame(rows)


def lri_lulc_by_period(
    lulc_raster,
    burned_raster,
    reference_raster,
    lulc_year,
    burned_period,
    weight,
    total_cells=None,
    **metadata
):
    """Calcula os LRi das classes LULC para um período do modelo.

    A expressão reproduz o cálculo usado no modelo: proporção de células
    ardidas em cada classe dividida pela proporção regional de células
    ardidas. Os rasters de áreas ardidas usam valores positivos nas células
    ardidas e NoData nas restantes.
    """

    class_cells = {code: 0 for code in LULC_MODELLED_CLASSES}
    burned_cells = {code: 0 for code in LULC_MODELLED_CLASSES}
    total_burned = 0

    if total_cells is None:
        total_cells = raster_valid_cell_count(reference_raster)

    with rio.open(reference_raster) as ref_src, \
         rio.open(lulc_raster) as lulc_src, \
         rio.open(burned_raster) as burned_src:
        _check_sources([ref_src, lulc_src, burned_src])

        for window in _windows(lulc_src):
            lulc = lulc_src.read(1, window=window)
            burned = burned_src.read(1, window=window)

            valid_lulc = _valid(lulc, lulc_src.nodata)
            valid_burned = _valid(burned, burned_src.nodata)
            burned_mask = valid_burned & (burned > 0)

            total_burned += int(burned_mask.sum())

            for class_code in LULC_MODELLED_CLASSES:
                class_mask = (
                    valid_lulc
                    & (lulc == class_code)
                )
                class_cells[class_code] += int(class_mask.sum())
                burned_cells[class_code] += int(
                    (class_mask & burned_mask).sum()
                )

    if total_cells == 0:
        raise ValueError(f"Raster de referência sem células válidas: {reference_raster}")

    if total_burned == 0:
        raise ValueError(f"Raster sem células ardidas: {burned_raster}")

    regional_rate = total_burned / total_cells
    rows = []

    for class_code in LULC_MODELLED_CLASSES:
        cells = class_cells[class_code]
        burned_count = burned_cells[class_code]
        class_rate = burned_count / cells if cells else np.nan
        lri = class_rate / regional_rate if cells else np.nan

        rows.append({
            **metadata,
            "lulc_year": lulc_year,
            "burned_period": burned_period,
            "weight": weight,
            "class_code": class_code,
            "class_label": LULC_CLASSES[class_code],
            "class_cells": cells,
            "burned_cells": burned_count,
            "class_burned_rate": class_rate,
            "regional_burned_rate": regional_rate,
            "lri": lri,
            "lulc_raster": str(lulc_raster),
            "burned_raster": str(burned_raster)
        })

    return pd.DataFrame(rows)


def rf_model_configuration(models_xlsx):
    """Lê os modelos e as variáveis a partir de models.xlsx."""

    models_xlsx = Path(models_xlsx)

    if not models_xlsx.exists():
        raise FileNotFoundError(f"Excel não encontrado: {models_xlsx}")

    models = pd.read_excel(models_xlsx, sheet_name="models")
    models = models.loc[models["status"] == "run"].copy()

    if models.empty:
        raise ValueError("Não existem modelos com status='run'.")

    model_name = models.iloc[0]["name"]
    features = pd.read_excel(models_xlsx, sheet_name=model_name)

    model_paths = models["model"].astype(str).tolist()
    feature_names = features["predfeat"].astype(str).tolist()
    feature_folder = Path(models.iloc[0]["predfeat"])
    feature_paths = [str(feature_folder / name) for name in feature_names]

    return model_paths, feature_names, feature_paths


def rf_feature_importance(model_paths, feature_names, **metadata):
    """Calcula a importância e o ranking das componentes em cada réplica."""

    rows = []

    for replicate, model_path in enumerate(model_paths, start=1):
        model = joblib.load(model_path)
        importance = np.asarray(model.feature_importances_, dtype="float64")

        if importance.size != len(feature_names):
            raise ValueError(
                f"Número de importâncias inesperado em: {model_path}"
            )

        order = np.argsort(-importance)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, importance.size + 1)

        for feature, value, rank in zip(
            feature_names,
            importance,
            ranks
        ):
            rows.append({
                **metadata,
                "replicate": replicate,
                "feature": feature,
                "feature_label": FEATURE_LABELS.get(feature, feature),
                "importance_type": "impurity_based_feature_importance",
                "importance": float(value),
                "rank": int(rank),
                "model": str(model_path)
            })

    detail = pd.DataFrame(rows)

    summary = (
        detail.groupby(
            [
                *metadata.keys(),
                "feature",
                "feature_label",
                "importance_type"
            ],
            dropna=False
        )
        .agg(
            importance_mean=("importance", "mean"),
            importance_sd=("importance", "std"),
            rank_mean=("rank", "mean"),
            first_rank_frequency=(
                "rank",
                lambda values: int((values == 1).sum())
            )
        )
        .reset_index()
    )

    return detail, summary


def rf_probability_percentiles(
    model_paths,
    feature_rasters,
    percentile=0.8,
    n_bins=10000,
    block_size=256,
    predict_batch_size=50000,
    class_value=1,
    n_jobs=1
):
    """Calcula o percentil regional de cada réplica RF por blocos.

    Apenas um modelo é mantido em memória de cada vez.
    """

    sources = [rio.open(path) for path in feature_rasters]
    edges = np.linspace(0.0, 1.0, n_bins + 1, dtype="float64")

    try:
        _check_sources(sources)
        thresholds = []

        for model_number, model_path in enumerate(model_paths, start=1):
            model = joblib.load(model_path)

            if hasattr(model, "set_params"):
                model.set_params(n_jobs=n_jobs)

            classes = list(model.classes_)

            if class_value not in classes:
                raise ValueError(
                    f"Classe {class_value} ausente em {model_path}."
                )

            class_index = classes.index(class_value)
            histogram = np.zeros(n_bins, dtype="int64")

            for window in _windows(sources[0], block_size):
                arrays = [
                    src.read(1, window=window).astype("float32")
                    for src in sources
                ]

                valid = np.ones(arrays[0].shape, dtype=bool)

                for array, src in zip(arrays, sources):
                    valid &= _valid(array, src.nodata)

                if not valid.any():
                    continue

                x = np.column_stack([
                    array[valid] for array in arrays
                ]).astype("float32", copy=False)

                for batch_start in range(
                    0,
                    x.shape[0],
                    predict_batch_size
                ):
                    batch_end = min(
                        batch_start + predict_batch_size,
                        x.shape[0]
                    )
                    probability = model.predict_proba(
                        x[batch_start:batch_end]
                    )[:, class_index]

                    counts, _ = np.histogram(probability, bins=edges)
                    histogram += counts

                del arrays, valid, x

            total = int(histogram.sum())

            if total == 0:
                raise ValueError(
                    f"Não foram produzidas probabilidades em {model_path}."
                )

            target = percentile * total
            index = int(np.searchsorted(
                np.cumsum(histogram),
                target,
                side="left"
            ))
            threshold = float(edges[min(index, n_bins - 1)])
            thresholds.append(threshold)

            print(
                "Modelo",
                f"{model_number}/{len(model_paths)}",
                "| percentil:",
                round(threshold, 6)
            )

            del model, histogram
            gc.collect()

        return thresholds
    finally:
        for src in sources:
            src.close()


def _create_zero_raster(path, profile, reference, dtype):
    """Cria um raster temporário inicializado a zero."""

    path = Path(path)
    path.unlink(missing_ok=True)

    output_profile = profile.copy()
    output_profile.update(
        dtype=dtype,
        nodata=None,
        compress="lzw",
        BIGTIFF="IF_SAFER"
    )

    with rio.open(path, "w", **output_profile) as dst:
        for window in _windows(reference, 256):
            zeros = np.zeros(
                (int(window.height), int(window.width)),
                dtype=dtype
            )
            dst.write(zeros, 1, window=window)


def rf_uncertainty_byblocks(
    model_paths,
    feature_rasters,
    thresholds,
    output_mean,
    output_sd,
    output_frequency,
    block_size=256,
    predict_batch_size=50000,
    class_value=1,
    n_jobs=1,
    nodata=-9999.0
):
    """Calcula média, desvio-padrão e frequência de Muito Alta.

    Os modelos são aplicados sequencialmente. As somas são guardadas em
    rasters temporários, evitando manter as dez Random Forest em memória.
    """

    if len(model_paths) != len(thresholds):
        raise ValueError(
            "Cada modelo deve ter um limite regional correspondente."
        )

    n_models = len(model_paths)

    if n_models < 2:
        raise ValueError("São necessários pelo menos dois modelos.")

    outputs = [
        Path(output_mean),
        Path(output_sd),
        Path(output_frequency)
    ]

    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.unlink(missing_ok=True)

    temporary = [
        outputs[0].with_name(outputs[0].stem + "_sum_tmp.tif"),
        outputs[0].with_name(outputs[0].stem + "_sumsq_tmp.tif"),
        outputs[0].with_name(outputs[0].stem + "_frequency_tmp.tif")
    ]

    sources = [rio.open(path) for path in feature_rasters]

    try:
        _check_sources(sources)
        reference = sources[0]

        _create_zero_raster(
            temporary[0],
            reference.profile,
            reference,
            "float64"
        )
        _create_zero_raster(
            temporary[1],
            reference.profile,
            reference,
            "float64"
        )
        _create_zero_raster(
            temporary[2],
            reference.profile,
            reference,
            "uint8"
        )

        for model_number, (model_path, threshold) in enumerate(
            zip(model_paths, thresholds),
            start=1
        ):
            print(
                "Aplicar modelo",
                f"{model_number}/{n_models}",
                "|",
                Path(model_path).name
            )

            model = joblib.load(model_path)

            if hasattr(model, "set_params"):
                model.set_params(n_jobs=n_jobs)

            classes = list(model.classes_)

            if class_value not in classes:
                raise ValueError(
                    f"Classe {class_value} ausente em {model_path}."
                )

            class_index = classes.index(class_value)

            with rio.open(temporary[0], "r+") as sum_dst, \
                 rio.open(temporary[1], "r+") as sumsq_dst, \
                 rio.open(temporary[2], "r+") as freq_dst:

                for window in _windows(reference, block_size):
                    arrays = [
                        src.read(1, window=window).astype("float32")
                        for src in sources
                    ]

                    valid = np.ones(arrays[0].shape, dtype=bool)

                    for array, src in zip(arrays, sources):
                        valid &= _valid(array, src.nodata)

                    if not valid.any():
                        continue

                    x = np.column_stack([
                        array[valid] for array in arrays
                    ]).astype("float32", copy=False)
                    probability = np.empty(x.shape[0], dtype="float32")

                    for batch_start in range(
                        0,
                        x.shape[0],
                        predict_batch_size
                    ):
                        batch_end = min(
                            batch_start + predict_batch_size,
                            x.shape[0]
                        )
                        probability[batch_start:batch_end] = (
                            model.predict_proba(
                                x[batch_start:batch_end]
                            )[:, class_index]
                        )

                    sum_array = sum_dst.read(1, window=window)
                    sumsq_array = sumsq_dst.read(1, window=window)
                    freq_array = freq_dst.read(1, window=window)

                    sum_array[valid] += probability
                    sumsq_array[valid] += probability.astype("float64") ** 2
                    freq_array[valid] += (
                        probability >= threshold
                    ).astype("uint8")

                    sum_dst.write(sum_array, 1, window=window)
                    sumsq_dst.write(sumsq_array, 1, window=window)
                    freq_dst.write(freq_array, 1, window=window)

                    del (
                        arrays,
                        valid,
                        x,
                        probability,
                        sum_array,
                        sumsq_array,
                        freq_array
                    )

            del model
            gc.collect()

        profile_float = reference.profile.copy()
        profile_float.update(
            dtype="float32",
            nodata=nodata,
            compress="lzw",
            BIGTIFF="IF_SAFER"
        )

        profile_frequency = reference.profile.copy()
        profile_frequency.update(
            dtype="uint8",
            nodata=255,
            compress="lzw",
            BIGTIFF="IF_SAFER"
        )

        with rio.open(temporary[0]) as sum_src, \
             rio.open(temporary[1]) as sumsq_src, \
             rio.open(temporary[2]) as freq_src, \
             rio.open(outputs[0], "w", **profile_float) as mean_dst, \
             rio.open(outputs[1], "w", **profile_float) as sd_dst, \
             rio.open(outputs[2], "w", **profile_frequency) as freq_dst:

            for window in _windows(reference, block_size):
                arrays = [
                    src.read(1, window=window).astype("float32")
                    for src in sources
                ]
                valid = np.ones(arrays[0].shape, dtype=bool)

                for array, src in zip(arrays, sources):
                    valid &= _valid(array, src.nodata)

                mean_array = np.full(
                    arrays[0].shape,
                    nodata,
                    dtype="float32"
                )
                sd_array = np.full(
                    arrays[0].shape,
                    nodata,
                    dtype="float32"
                )
                frequency_array = np.full(
                    arrays[0].shape,
                    255,
                    dtype="uint8"
                )

                if valid.any():
                    sums = sum_src.read(1, window=window)
                    sums_sq = sumsq_src.read(1, window=window)
                    frequency = freq_src.read(1, window=window)

                    means = sums[valid] / n_models
                    variance = (
                        sums_sq[valid] - (sums[valid] ** 2) / n_models
                    ) / (n_models - 1)
                    variance = np.maximum(variance, 0.0)

                    mean_array[valid] = means.astype("float32")
                    sd_array[valid] = np.sqrt(variance).astype("float32")
                    frequency_array[valid] = frequency[valid]

                mean_dst.write(mean_array, 1, window=window)
                sd_dst.write(sd_array, 1, window=window)
                freq_dst.write(frequency_array, 1, window=window)

        return [str(output) for output in outputs]
    finally:
        for src in sources:
            src.close()

        for path in temporary:
            path.unlink(missing_ok=True)

        gc.collect()

def rf_uncertainty_summary(sd_raster, frequency_raster, n_models, **metadata):
    """Resume o desvio-padrão e a frequência em toda a área-piloto."""

    sd_values = []
    frequency_counts = np.zeros(n_models + 1, dtype="int64")

    with rio.open(sd_raster) as sd_src, rio.open(frequency_raster) as fr_src:
        _check_sources([sd_src, fr_src])
        cell_area_m2 = abs(sd_src.transform.a * sd_src.transform.e)

        for window in _windows(sd_src):
            sd = sd_src.read(1, window=window).astype("float64")
            frequency = fr_src.read(1, window=window)
            valid_sd = _valid(sd, sd_src.nodata)
            valid_frequency = frequency != fr_src.nodata

            if valid_sd.any():
                sd_values.append(sd[valid_sd])

            if valid_frequency.any():
                frequency_counts += np.bincount(
                    frequency[valid_frequency],
                    minlength=n_models + 1
                )[:n_models + 1]

    values = np.concatenate(sd_values) if sd_values else np.array([])
    total = int(frequency_counts.sum())

    return pd.DataFrame([{
        **metadata,
        "valid_cells": total,
        "sd_mean": float(values.mean()) if values.size else np.nan,
        "sd_median": float(np.median(values)) if values.size else np.nan,
        "sd_p95": float(np.percentile(values, 95)) if values.size else np.nan,
        "frequency_0_2_pct": (
            100.0 * frequency_counts[0:3].sum() / total
            if total else np.nan
        ),
        "frequency_3_7_pct": (
            100.0 * frequency_counts[3:min(8, n_models + 1)].sum() / total
            if total else np.nan
        ),
        "frequency_8_10_pct": (
            100.0 * frequency_counts[max(0, n_models - 2):].sum() / total
            if total else np.nan
        ),
        "frequency_10_pct": (
            100.0 * frequency_counts[n_models] / total
            if total else np.nan
        ),
        "frequency_10_area_km2": (
            frequency_counts[n_models] * cell_area_m2 / 1_000_000
        ),
        "sd_raster": str(sd_raster),
        "frequency_raster": str(frequency_raster)
    }])


def rf_uncertainty_very_high_summary(
    sd_raster,
    frequency_raster,
    class_raster,
    n_models,
    **metadata
):
    """Resume a estabilidade apenas na classe Muito Alta do mapa médio."""

    sd_values = []
    frequency_counts = np.zeros(n_models + 1, dtype="int64")

    with rio.open(sd_raster) as sd_src, \
         rio.open(frequency_raster) as fr_src, \
         rio.open(class_raster) as cls_src:
        _check_sources([sd_src, fr_src, cls_src])
        cell_area_m2 = abs(sd_src.transform.a * sd_src.transform.e)

        for window in _windows(sd_src):
            sd = sd_src.read(1, window=window).astype("float64")
            frequency = fr_src.read(1, window=window)
            classes = cls_src.read(1, window=window)

            valid = (
                _valid(sd, sd_src.nodata)
                & (frequency != fr_src.nodata)
                & (classes == 5)
            )

            if not valid.any():
                continue

            sd_values.append(sd[valid])
            frequency_counts += np.bincount(
                frequency[valid],
                minlength=n_models + 1
            )[:n_models + 1]

    values = np.concatenate(sd_values) if sd_values else np.array([])
    total = int(frequency_counts.sum())

    return pd.DataFrame([{
        **metadata,
        "very_high_cells": total,
        "very_high_area_km2": total * cell_area_m2 / 1_000_000,
        "sd_mean_very_high": (
            float(values.mean()) if values.size else np.nan
        ),
        "sd_median_very_high": (
            float(np.median(values)) if values.size else np.nan
        ),
        "frequency_0_2_pct_very_high": (
            100.0 * frequency_counts[0:3].sum() / total
            if total else np.nan
        ),
        "frequency_3_7_pct_very_high": (
            100.0 * frequency_counts[3:min(8, n_models + 1)].sum() / total
            if total else np.nan
        ),
        "frequency_8_10_pct_very_high": (
            100.0 * frequency_counts[max(0, n_models - 2):].sum() / total
            if total else np.nan
        ),
        "frequency_10_pct_very_high": (
            100.0 * frequency_counts[n_models] / total
            if total else np.nan
        ),
        "frequency_10_area_km2_very_high": (
            frequency_counts[n_models] * cell_area_m2 / 1_000_000
        ),
        "class_raster": str(class_raster)
    }])


def raster_value_frequency(raster, value_name="value", decimals=8, **metadata):
    """Conta os valores únicos de um raster por blocos."""

    counts = {}

    with rio.open(raster) as src:
        for window in _windows(src):
            array = src.read(1, window=window).astype("float64")
            values = np.round(
                array[_valid(array, src.nodata)],
                decimals
            )

            if not values.size:
                continue

            unique, frequency = np.unique(values, return_counts=True)

            for value, cells in zip(unique, frequency):
                key = float(value)
                counts[key] = counts.get(key, 0) + int(cells)

    return pd.DataFrame([
        {
            **metadata,
            value_name: value,
            "cells": cells
        }
        for value, cells in sorted(counts.items())
    ])


def rf_uncertainty_local_fast(
    model_paths,
    feature_rasters,
    threshold,
    output_sd,
    output_frequency,
    block_size=262144,
    class_value=1,
    n_jobs=2,
    nodata=-9999.0
):
    """Calcula a incerteza local das réplicas RF de forma simples.

    As variáveis do município são lidas uma única vez. Depois, cada modelo
    é aplicado sequencialmente à mesma matriz de pixels válidos. O limiar
    de Muito Alta é comum a todas as réplicas e corresponde ao limite
    regional do mapa médio final.
    """

    output_sd = Path(output_sd)
    output_frequency = Path(output_frequency)

    for output in [output_sd, output_frequency]:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.unlink(missing_ok=True)

    sources = [rio.open(path) for path in feature_rasters]

    try:
        _check_sources(sources)
        reference = sources[0]

        arrays = [
            src.read(1).astype("float32")
            for src in sources
        ]

        valid = np.ones(arrays[0].shape, dtype=bool)

        for array, src in zip(arrays, sources):
            valid &= _valid(array, src.nodata)

        if not valid.any():
            raise ValueError("Não existem pixels válidos no município.")

        x = np.column_stack([
            array[valid] for array in arrays
        ]).astype("float32", copy=False)

        del arrays
        gc.collect()

        sums = np.zeros(x.shape[0], dtype="float64")
        sums_sq = np.zeros(x.shape[0], dtype="float64")
        frequency = np.zeros(x.shape[0], dtype="uint8")

        for model_number, model_path in enumerate(model_paths, start=1):
            print(
                "Aplicar modelo",
                f"{model_number}/{len(model_paths)}",
                "|",
                Path(model_path).name
            )

            model = joblib.load(model_path)

            if hasattr(model, "set_params"):
                model.set_params(n_jobs=n_jobs)

            classes = list(model.classes_)

            if class_value not in classes:
                raise ValueError(
                    f"Classe {class_value} ausente em {model_path}."
                )

            class_index = classes.index(class_value)

            for start in range(0, x.shape[0], block_size):
                end = min(start + block_size, x.shape[0])
                probability = model.predict_proba(x[start:end])[:, class_index]

                sums[start:end] += probability
                sums_sq[start:end] += probability.astype("float64") ** 2
                frequency[start:end] += (
                    probability >= threshold
                ).astype("uint8")

            del model
            gc.collect()

        n_models = len(model_paths)
        variance = (
            sums_sq - (sums ** 2) / n_models
        ) / (n_models - 1)
        variance = np.maximum(variance, 0.0)

        sd_values = np.sqrt(variance).astype("float32")

        sd_array = np.full(valid.shape, nodata, dtype="float32")
        frequency_array = np.full(valid.shape, 255, dtype="uint8")

        sd_array[valid] = sd_values
        frequency_array[valid] = frequency

        profile_sd = reference.profile.copy()
        profile_sd.update(
            dtype="float32",
            nodata=nodata,
            compress="lzw",
            BIGTIFF="IF_SAFER"
        )

        profile_frequency = reference.profile.copy()
        profile_frequency.update(
            dtype="uint8",
            nodata=255,
            compress="lzw",
            BIGTIFF="IF_SAFER"
        )

        with rio.open(output_sd, "w", **profile_sd) as dst:
            dst.write(sd_array, 1)

        with rio.open(output_frequency, "w", **profile_frequency) as dst:
            dst.write(frequency_array, 1)

        return str(output_sd), str(output_frequency)
    finally:
        for src in sources:
            src.close()

        gc.collect()


def very_high_threshold(class_breaks_xlsx, map_id, area=None):
    """Lê o limite inferior da classe Muito Alta de um mapa regional."""

    table = pd.read_excel(class_breaks_xlsx, sheet_name="Class_breaks")
    mask = (
        (table["map_id"] == map_id)
        & (table["class_code"] == 5)
    )

    if area is not None:
        mask &= table["area"] == area

    rows = table.loc[mask]

    if rows.empty:
        raise ValueError(
            f"Limite de Muito Alta não encontrado: {area} {map_id}"
        )

    if len(rows) > 1:
        raise ValueError(
            f"Limite de Muito Alta ambíguo: {area} {map_id}"
        )

    return float(rows.iloc[0]["lower_bound"])
