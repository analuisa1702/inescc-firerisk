"""
Funções auxiliares para os cenários sazonais C7 e C8.
"""

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rasterio as rio


def _valid_mask(array, nodata):
    """Identifica os valores válidos de um bloco raster."""

    valid = np.isfinite(array)

    if nodata is not None:
        try:
            nodata_is_nan = np.isnan(nodata)
        except TypeError:
            nodata_is_nan = False

        if not nodata_is_nan:
            valid = valid & (array != nodata)

    return valid


def _raster_windows(src, block_size=1024):
    """Gera janelas raster com dimensão máxima definida."""

    from rasterio.windows import Window

    for row_off in range(0, src.height, block_size):
        height = min(
            block_size,
            src.height - row_off
        )

        for col_off in range(0, src.width, block_size):
            width = min(
                block_size,
                src.width - col_off
            )

            yield Window(
                col_off=col_off,
                row_off=row_off,
                width=width,
                height=height
            )


def _check_open_rasters(reference, others):
    """Confirma que vários rasters abertos usam a mesma grelha."""

    for name, src in others.items():
        if src.shape != reference.shape:
            raise ValueError(
                f"Dimensão incompatível em {name}: "
                f"{src.shape} != {reference.shape}"
            )

        if src.crs != reference.crs:
            raise ValueError(
                f"CRS incompatível em {name}: "
                f"{src.crs} != {reference.crs}"
            )

        if not src.transform.almost_equals(
            reference.transform,
            precision=1e-8
        ):
            raise ValueError(
                f"Transformação incompatível em {name}."
            )


def _pick_name(names, candidates):
    """Procura um nome de variável ou coordenada."""

    for candidate in candidates:
        if candidate in names:
            return candidate

    for name in names:
        name_lower = name.lower()

        if any(
            candidate in name_lower
            for candidate in candidates
        ):
            return name

    return None


def dsr_to_ssr_raster(
    netcdf_files,
    year,
    reference,
    output,
    variable=None,
    nodata=-9999.0
):
    """
    Soma o DSR entre 1 de abril e 15 de junho e alinha o SSR
    com um raster de referência.
    """

    import xarray as xr
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject, Resampling

    netcdf_files = [
        Path(path)
        for path in netcdf_files
    ]

    if not netcdf_files:
        raise ValueError(
            f"Não existem ficheiros NetCDF para {year}."
        )

    datasets = [
        xr.open_dataset(path)
        for path in netcdf_files
    ]

    try:
        time_name = _pick_name(
            list(datasets[0].coords),
            ["valid_time", "time"]
        )

        if time_name is None:
            raise ValueError(
                "Não foi encontrada uma coordenada temporal."
            )

        if len(datasets) == 1:
            dataset = datasets[0]
        else:
            dataset = xr.concat(
                datasets,
                dim=time_name
            ).sortby(time_name)

        lon_name = _pick_name(
            list(dataset.coords),
            ["longitude", "lon"]
        )

        lat_name = _pick_name(
            list(dataset.coords),
            ["latitude", "lat"]
        )

        if lon_name is None or lat_name is None:
            raise ValueError(
                "Não foram encontradas as coordenadas latitude/longitude."
            )

        longitude = dataset[lon_name]

        if float(longitude.max()) > 180:
            dataset = dataset.assign_coords({
                lon_name: (
                    (longitude + 180) % 360
                ) - 180
            }).sortby(lon_name)

        if variable is None:
            variable = _pick_name(
                list(dataset.data_vars),
                [
                    "fire_daily_severity_rating",
                    "fire_daily_severity_index",
                    "daily_severity",
                    "severity",
                    "dsr"
                ]
            )

        if variable is None:
            raise ValueError(
                "Define variable com o nome da variável DSR."
            )

        dsr = dataset[variable]
        time = dsr[time_name]

        period = (
            (time.dt.year == year)
            & (
                (time.dt.month == 4)
                | (time.dt.month == 5)
                | (
                    (time.dt.month == 6)
                    & (time.dt.day <= 15)
                )
            )
        )

        selected = dsr.where(
            period,
            drop=True
        )

        if selected.sizes.get(time_name, 0) == 0:
            raise ValueError(
                f"Não existem valores DSR para {year}."
            )

        ssr = selected.sum(
            time_name,
            skipna=True,
            min_count=1
        ).squeeze(drop=True)

        extra_dims = [
            dim
            for dim in ssr.dims
            if dim not in [lat_name, lon_name]
        ]

        if extra_dims:
            raise ValueError(
                "O SSR ainda contém dimensões adicionais: "
                f"{extra_dims}"
            )

        ssr = ssr.transpose(
            lat_name,
            lon_name
        ).load()

        lons = ssr[lon_name].values
        lats = ssr[lat_name].values
        array = ssr.values.astype("float32")

        if lats[0] < lats[-1]:
            array = array[::-1, :]
            lats = lats[::-1]

        dx = abs(float(lons[1] - lons[0]))
        dy = abs(float(lats[1] - lats[0]))

        src_transform = from_bounds(
            float(lons.min() - dx / 2),
            float(lats.min() - dy / 2),
            float(lons.max() + dx / 2),
            float(lats.max() + dy / 2),
            array.shape[1],
            array.shape[0]
        )

        output = Path(output)
        output.parent.mkdir(
            parents=True,
            exist_ok=True
        )
        output.unlink(missing_ok=True)

        with rio.open(reference) as ref:
            profile = ref.profile.copy()

            destination = np.full(
                (ref.height, ref.width),
                nodata,
                dtype="float32"
            )

            reproject(
                source=array,
                destination=destination,
                src_transform=src_transform,
                src_crs="EPSG:4326",
                dst_transform=ref.transform,
                dst_crs=ref.crs,
                src_nodata=np.nan,
                dst_nodata=nodata,
                resampling=Resampling.bilinear,
                init_dest_nodata=True
            )

            destination[
                ref.read_masks(1) == 0
            ] = nodata

        profile.update(
            driver="GTiff",
            dtype="float32",
            count=1,
            nodata=nodata,
            compress="lzw",
            BIGTIFF="IF_SAFER"
        )

        with rio.open(
            output,
            "w",
            **profile
        ) as dst:
            dst.write(destination, 1)

        print("Ano:", year)
        print("Variável DSR:", variable)
        print("Dias utilizados:", selected.sizes[time_name])
        print("SSR criado:", output)

    finally:
        for dataset in datasets:
            dataset.close()

    return str(output)


def create_yearly_burned_target(
    vector,
    reference,
    output,
    year,
    date_field="initialdat",
    nodata=-1,
    block_size=1024
):
    """
    Cria um target anual de verão.

    Valores:
    -1 = área excluída;
     0 = não ardida;
     1 = ardida entre 15 de junho e 15 de setembro.
    """

    import geopandas as gpd
    from rasterio.features import rasterize
    from rasterio.windows import bounds, transform
    from shapely import make_valid
    from shapely.geometry import box

    vector = Path(vector)
    output = Path(output)

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    output.unlink(missing_ok=True)

    gdf = None

    if vector.exists():
        gdf = gpd.read_file(vector)

        if date_field in gdf.columns:
            dates = pd.to_datetime(
                gdf[date_field],
                errors="coerce"
            )

            summer = (
                (dates.dt.year == year)
                & (
                    (
                        (dates.dt.month == 6)
                        & (dates.dt.day >= 15)
                    )
                    | (dates.dt.month == 7)
                    | (dates.dt.month == 8)
                    | (
                        (dates.dt.month == 9)
                        & (dates.dt.day <= 15)
                    )
                )
            )

            gdf = gdf[
                dates.notna() & summer
            ].copy()
        else:
            print(
                year,
                "- sem campo de data; "
                "são usados todos os polígonos."
            )

    with rio.open(reference) as ref:
        profile = ref.profile.copy()
        profile.pop("predictor", None)
        profile.update(
            driver="GTiff",
            dtype="int16",
            count=1,
            nodata=nodata,
            compress="lzw",
            BIGTIFF="IF_SAFER"
        )

        if gdf is not None and len(gdf):
            gdf = gdf.to_crs(ref.crs)
            gdf["geometry"] = gdf.geometry.apply(
                lambda geometry: (
                    make_valid(geometry)
                    if geometry is not None
                    else None
                )
            )

            gdf = gdf[
                gdf.geometry.notna()
                & ~gdf.geometry.is_empty
            ].copy()

        n_polygons = 0 if gdf is None else len(gdf)
        spatial_index = (
            gdf.sindex
            if n_polygons
            else None
        )

        counts = {}

        with rio.open(
            output,
            "w",
            **profile
        ) as dst:
            for window in _raster_windows(
                ref,
                block_size=block_size
            ):
                valid = ref.read_masks(
                    1,
                    window=window
                ) > 0

                target = np.full(
                    valid.shape,
                    nodata,
                    dtype="int16"
                )
                target[valid] = 0

                if spatial_index is not None:
                    window_bounds = bounds(
                        window,
                        ref.transform
                    )
                    window_box = box(*window_bounds)

                    candidate_ids = list(
                        spatial_index.query(
                            window_box,
                            predicate="intersects"
                        )
                    )

                    if candidate_ids:
                        geometries = [
                            geometry
                            for geometry in gdf.geometry.iloc[
                                candidate_ids
                            ]
                            if geometry is not None
                            and not geometry.is_empty
                        ]

                        if geometries:
                            burned = rasterize(
                                [
                                    (geometry, 1)
                                    for geometry in geometries
                                ],
                                out_shape=target.shape,
                                transform=transform(
                                    window,
                                    ref.transform
                                ),
                                fill=0,
                                dtype="int16"
                            )

                            target[
                                valid & (burned == 1)
                            ] = 1

                values, frequencies = np.unique(
                    target,
                    return_counts=True
                )

                for value, frequency in zip(
                    values,
                    frequencies
                ):
                    value = int(value)
                    frequency = int(frequency)
                    counts[value] = (
                        counts.get(value, 0)
                        + frequency
                    )

                dst.write(
                    target,
                    1,
                    window=window
                )

    print("Ano:", year)
    print("Polígonos de verão:", n_polygons)
    print("Target criado:", output)
    print("Valores:", counts)

    return str(output)


def build_common_samples(
    years,
    target_folder,
    ssr_folder,
    susceptibility_lr,
    susceptibility_rf,
    output_csv,
    n_burned=200,
    n_unburned=200,
    seed=42,
    block_size=1024
):
    """
    Cria uma amostra anual comum aos cenários C7 e C8.

    A seleção é realizada na interseção das células válidas de:
    target, SSR, suscetibilidade LR e suscetibilidade RF.
    """

    target_folder = Path(target_folder)
    ssr_folder = Path(ssr_folder)
    output_csv = Path(output_csv)

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    rows_out = []

    with rio.open(susceptibility_lr) as lr_src, rio.open(
        susceptibility_rf
    ) as rf_src:
        _check_open_rasters(
            lr_src,
            {"suscetibilidade RF": rf_src}
        )

        for year in years:
            target_path = target_folder / (
                f"rst_ba_{year}_target.tif"
            )
            ssr_path = ssr_folder / (
                f"ssr_abs_{year}.tif"
            )

            with rio.open(target_path) as target_src, rio.open(
                ssr_path
            ) as ssr_src:
                _check_open_rasters(
                    lr_src,
                    {
                        "target": target_src,
                        "SSR": ssr_src
                    }
                )

                counts = {0: 0, 1: 0}

                for window in _raster_windows(
                    lr_src,
                    block_size=block_size
                ):
                    target = target_src.read(
                        1,
                        window=window
                    )
                    ssr = ssr_src.read(
                        1,
                        window=window
                    )
                    lr = lr_src.read(
                        1,
                        window=window
                    )
                    rf = rf_src.read(
                        1,
                        window=window
                    )

                    valid = (
                        np.isin(target, [0, 1])
                        & _valid_mask(ssr, ssr_src.nodata)
                        & _valid_mask(lr, lr_src.nodata)
                        & _valid_mask(rf, rf_src.nodata)
                    )

                    counts[0] += int(
                        np.count_nonzero(
                            valid & (target == 0)
                        )
                    )
                    counts[1] += int(
                        np.count_nonzero(
                            valid & (target == 1)
                        )
                    )

                requested = {
                    0: int(n_unburned),
                    1: int(n_burned)
                }

                for value in requested:
                    if requested[value] > counts[value]:
                        raise ValueError(
                            f"{year}: classe {value} tem "
                            f"{counts[value]} células, mas foram "
                            f"pedidas {requested[value]}."
                        )

                rng = np.random.default_rng(
                    seed + int(year)
                )

                selected_positions = {
                    value: np.sort(
                        rng.choice(
                            counts[value],
                            size=requested[value],
                            replace=False
                        )
                    )
                    for value in requested
                }

                seen = {0: 0, 1: 0}
                selected_year = []

                for window in _raster_windows(
                    lr_src,
                    block_size=block_size
                ):
                    target = target_src.read(
                        1,
                        window=window
                    )
                    ssr = ssr_src.read(
                        1,
                        window=window
                    )
                    lr = lr_src.read(
                        1,
                        window=window
                    )
                    rf = rf_src.read(
                        1,
                        window=window
                    )

                    valid = (
                        np.isin(target, [0, 1])
                        & _valid_mask(ssr, ssr_src.nodata)
                        & _valid_mask(lr, lr_src.nodata)
                        & _valid_mask(rf, rf_src.nodata)
                    )

                    flat_target = target.ravel()
                    flat_valid = valid.ravel()

                    for value in [0, 1]:
                        local_indices = np.flatnonzero(
                            flat_valid
                            & (flat_target == value)
                        )

                        n_local = local_indices.size

                        if not n_local:
                            continue

                        start = seen[value]
                        end = start + n_local
                        positions = selected_positions[value]

                        lower = np.searchsorted(
                            positions,
                            start,
                            side="left"
                        )
                        upper = np.searchsorted(
                            positions,
                            end,
                            side="left"
                        )

                        if upper > lower:
                            chosen = local_indices[
                                positions[lower:upper] - start
                            ]

                            local_rows, local_cols = np.unravel_index(
                                chosen,
                                target.shape
                            )

                            global_rows = (
                                local_rows
                                + int(window.row_off)
                            )
                            global_cols = (
                                local_cols
                                + int(window.col_off)
                            )

                            x_coords, y_coords = rio.transform.xy(
                                target_src.transform,
                                global_rows,
                                global_cols,
                                offset="center"
                            )

                            selected_year.append(
                                pd.DataFrame({
                                    "year": year,
                                    "row": global_rows,
                                    "col": global_cols,
                                    "x": x_coords,
                                    "y": y_coords,
                                    "burned": value,
                                    "susc_lr": lr.ravel()[chosen],
                                    "susc_rf": rf.ravel()[chosen],
                                    "ssr_abs": ssr.ravel()[chosen]
                                })
                            )

                        seen[value] = end

                year_df = pd.concat(
                    selected_year,
                    ignore_index=True
                )

                year_counts = year_df.groupby(
                    "burned"
                ).size().to_dict()

                print(
                    year,
                    "- amostra:",
                    year_counts,
                    "- disponíveis:",
                    counts
                )

                rows_out.append(year_df)

    samples = pd.concat(
        rows_out,
        ignore_index=True
    )

    samples.sort_values(
        ["year", "burned", "row", "col"],
        inplace=True
    )
    samples.reset_index(
        drop=True,
        inplace=True
    )

    samples.to_csv(
        output_csv,
        index=False
    )

    print("Amostra guardada:", output_csv)
    print("Dimensão:", samples.shape)

    return samples


def train_logistic_model(
    table,
    features,
    target,
    output_model,
    max_iter=5000
):
    """
    Treina uma regressão logística binária sem regularização.
    """

    from sklearn.linear_model import LogisticRegression

    x = table[features].to_numpy(
        dtype="float64"
    )
    y = table[target].to_numpy(
        dtype="int16"
    )

    model = LogisticRegression(
        C=1e12,
        solver="lbfgs",
        max_iter=max_iter,
        tol=1e-8
    )

    model.fit(x, y)
    model.seasonal_features_ = list(features)

    output_model = Path(output_model)
    output_model.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        output_model
    )

    print("Modelo criado:", output_model)
    print("Iterações:", model.n_iter_[0])

    return model


def apply_logistic_byblocks(
    model_file,
    feature_rasters,
    validity_rasters,
    output_probability,
    output_class=None,
    class_value=1,
    threshold=0.5,
    block_size=1024,
    predict_batch_size=250000,
    probability_nodata=101.0,
    class_nodata=-1
):
    """
    Aplica uma regressão logística por blocos.

    validity_rasters define a máscara comum usada em todos os modelos.
    """

    model = joblib.load(model_file)

    feature_sources = [
        rio.open(path)
        for path in feature_rasters
    ]
    validity_sources = [
        rio.open(path)
        for path in validity_rasters
    ]

    try:
        reference = validity_sources[0]

        others = {}

        for i, src in enumerate(feature_sources):
            others[f"feature_{i + 1}"] = src

        for i, src in enumerate(validity_sources[1:]):
            others[f"validity_{i + 2}"] = src

        _check_open_rasters(
            reference,
            others
        )

        probability_profile = reference.profile.copy()
        probability_profile.pop("predictor", None)
        probability_profile.update(
            driver="GTiff",
            dtype="float32",
            count=1,
            nodata=probability_nodata,
            compress="lzw",
            BIGTIFF="IF_SAFER"
        )

        class_profile = reference.profile.copy()
        class_profile.pop("predictor", None)
        class_profile.update(
            driver="GTiff",
            dtype="int16",
            count=1,
            nodata=class_nodata,
            compress="lzw",
            BIGTIFF="IF_SAFER"
        )

        output_probability = Path(
            output_probability
        )
        output_probability.parent.mkdir(
            parents=True,
            exist_ok=True
        )
        output_probability.unlink(
            missing_ok=True
        )

        if output_class:
            output_class = Path(output_class)
            output_class.parent.mkdir(
                parents=True,
                exist_ok=True
            )
            output_class.unlink(
                missing_ok=True
            )

        classes = list(model.classes_)

        if class_value not in classes:
            raise ValueError(
                f"A classe {class_value} não existe no modelo."
            )

        class_index = classes.index(class_value)
        windows = list(
            _raster_windows(
                reference,
                block_size=block_size
            )
        )

        class_context = (
            rio.open(
                output_class,
                "w",
                **class_profile
            )
            if output_class
            else None
        )

        with rio.open(
            output_probability,
            "w",
            **probability_profile
        ) as probability_dst:
            try:
                for i, window in enumerate(
                    windows,
                    start=1
                ):
                    validity_arrays = [
                        src.read(
                            1,
                            window=window
                        )
                        for src in validity_sources
                    ]

                    valid = np.ones(
                        validity_arrays[0].shape,
                        dtype=bool
                    )

                    for array, src in zip(
                        validity_arrays,
                        validity_sources
                    ):
                        valid = valid & _valid_mask(
                            array,
                            src.nodata
                        )

                    probability = np.full(
                        valid.shape,
                        probability_nodata,
                        dtype="float32"
                    )

                    classification = np.full(
                        valid.shape,
                        class_nodata,
                        dtype="int16"
                    )

                    if valid.any():
                        feature_arrays = [
                            src.read(
                                1,
                                window=window
                            )
                            for src in feature_sources
                        ]

                        x = np.column_stack([
                            array[valid]
                            for array in feature_arrays
                        ]).astype(
                            "float64",
                            copy=False
                        )

                        probabilities = np.empty(
                            x.shape[0],
                            dtype="float32"
                        )

                        for start in range(
                            0,
                            x.shape[0],
                            predict_batch_size
                        ):
                            end = min(
                                start + predict_batch_size,
                                x.shape[0]
                            )

                            probabilities[start:end] = (
                                model.predict_proba(
                                    x[start:end]
                                )[:, class_index]
                            )

                        probability[valid] = probabilities
                        classification[valid] = (
                            probabilities >= threshold
                        ).astype("int16")

                    probability_dst.write(
                        probability,
                        1,
                        window=window
                    )

                    if class_context:
                        class_context.write(
                            classification,
                            1,
                            window=window
                        )

                    if i % 50 == 0 or i == len(windows):
                        print(
                            "Blocos:",
                            f"{i}/{len(windows)}"
                        )

            finally:
                if class_context:
                    class_context.close()

    finally:
        for src in feature_sources:
            src.close()

        for src in validity_sources:
            src.close()

    print(
        "Probabilidade criada:",
        output_probability
    )

    if output_class:
        print(
            "Classificação criada:",
            output_class
        )

    return (
        str(output_probability),
        str(output_class) if output_class else None
    )


def classification_metrics(
    observed,
    probability,
    threshold=0.5
):
    """Calcula as métricas da validação binária sobre pontos."""

    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        roc_auc_score
    )

    observed = np.asarray(observed)
    probability = np.asarray(probability)
    predicted = (
        probability >= threshold
    ).astype("int16")

    tn, fp, fn, tp = confusion_matrix(
        observed,
        predicted,
        labels=[0, 1]
    ).ravel()

    return {
        "auc_roc": float(roc_auc_score(
            observed,
            probability
        )),
        "accuracy": float(accuracy_score(
            observed,
            predicted
        )),
        "correctly_predicted_pct": float(
            accuracy_score(
                observed,
                predicted
            ) * 100
        ),
        "sensitivity": float(
            tp / (tp + fn)
        ) if (tp + fn) else np.nan,
        "specificity": float(
            tn / (tn + fp)
        ) if (tn + fp) else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }
