"""Funções auxiliares para leitura, recorte e harmonização de dados LULC."""

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio


def _normalise_code(series):
    """Normaliza códigos lidos como texto ou número."""

    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def _clean_geometries(gdf):
    """Remove geometrias nulas e vazias."""

    return gdf.loc[
        gdf.geometry.notna()
        & ~gdf.geometry.is_empty
    ].copy()


def _get_layer(file, layer=None):
    """Confirma ou identifica a camada vetorial de um GeoPackage."""

    layers = pyogrio.list_layers(file)
    names = layers[:, 0].tolist()

    if layer is not None:
        if layer not in names:
            raise ValueError(
                f"A layer '{layer}' não existe em {file}. "
                f"Layers disponíveis: {names}"
            )

        return layer

    spatial_layers = [
        name
        for name, geometry_type in layers
        if geometry_type is not None
    ]

    if "T_POLIGONOS" in spatial_layers:
        return "T_POLIGONOS"

    if len(spatial_layers) == 1:
        return spatial_layers[0]

    raise ValueError(
        f"Não foi possível identificar automaticamente a layer de {file}. "
        f"Layers vetoriais disponíveis: {spatial_layers}"
    )


def read_lulc(files, aoi, layers=None, where=None):
    """Lê um ou vários ficheiros LULC e recorta-os à AOI."""

    if isinstance(files, (str, Path)):
        files = [str(files)]

    if layers is None:
        layers = [None] * len(files)
    elif isinstance(layers, str):
        layers = [layers] * len(files)

    if not files:
        raise ValueError("Não foram definidos ficheiros para este produto.")

    if len(files) != len(layers):
        raise ValueError("Cada ficheiro deve ter uma layer correspondente.")

    if aoi.crs is None:
        raise ValueError("A área de estudo não tem CRS definido.")

    parts = []
    target_crs = None
    aoi_target = None

    for file, layer in zip(files, layers):
        if not Path(file).exists():
            raise FileNotFoundError(f"Ficheiro não encontrado: {file}")

        layer = _get_layer(file, layer)
        sample = gpd.read_file(file, layer=layer, rows=1)

        if sample.crs is None:
            raise ValueError(f"O ficheiro não tem CRS definido: {file}")

        if target_crs is None:
            target_crs = sample.crs
            aoi_target = aoi.to_crs(target_crs).dissolve()

        aoi_source = aoi.to_crs(sample.crs).dissolve()

        options = {
            "layer": layer,
            "mask": aoi_source.geometry.iloc[0],
        }

        if where:
            options["where"] = where

        part = pyogrio.read_dataframe(file, **options)
        part = _clean_geometries(part)

        if part.empty:
            continue

        if part.crs != target_crs:
            part = part.to_crs(target_crs)

        parts.append(part)

    if not parts:
        raise ValueError("Nenhuma feição interseta a área de estudo.")

    lulc = gpd.GeoDataFrame(
        pd.concat(parts, ignore_index=True),
        geometry="geometry",
        crs=target_crs,
    )

    lulc = gpd.clip(lulc, aoi_target)
    lulc = _clean_geometries(lulc).reset_index(drop=True)

    if lulc.empty:
        raise ValueError("O recorte exato não contém feições LULC.")

    return lulc


def harmonize_lulc(lulc, lookup, product, code_field, label_field=None):
    """Aplica a tabela de harmonização e filtra as classes do modelo."""

    required = {
        "product",
        "code_original",
        "id_harm",
        "label_harmonizada",
        "incluir_modelo",
        "obs",
    }

    missing_fields = required - set(lookup.columns)

    if missing_fields:
        raise ValueError(
            "Campos em falta na tabela de harmonização: "
            f"{sorted(missing_fields)}"
        )

    if code_field not in lulc.columns:
        raise ValueError(f"Campo de código em falta: {code_field}")

    lookup_product = lookup.loc[
        lookup["product"] == product
    ].copy()

    if lookup_product.empty:
        raise ValueError(
            f"O produto '{product}' não existe na tabela de harmonização."
        )

    lookup_product["code_original"] = _normalise_code(
        lookup_product["code_original"]
    )

    duplicates = lookup_product.loc[
        lookup_product.duplicated("code_original", keep=False)
    ]

    if not duplicates.empty:
        raise ValueError(
            f"Existem códigos repetidos para o produto '{product}'."
        )

    lulc = lulc.copy()
    lulc[code_field] = _normalise_code(lulc[code_field])

    columns = [code_field]

    if label_field and label_field in lulc.columns:
        columns.append(label_field)

    classes = (
        lulc[columns]
        .drop_duplicates()
        .rename(columns={code_field: "code_original"})
    )

    if label_field and label_field in classes.columns:
        classes = classes.rename(
            columns={label_field: "label_original"}
        )
    else:
        classes["label_original"] = None

    classes_missing = classes.merge(
        lookup_product[["code_original"]],
        on="code_original",
        how="left",
        indicator=True,
    )

    classes_missing = classes_missing.loc[
        classes_missing["_merge"] == "left_only",
        ["code_original", "label_original"],
    ].sort_values(["code_original", "label_original"])

    harmonized = lulc.merge(
        lookup_product,
        left_on=code_field,
        right_on="code_original",
        how="inner",
        validate="many_to_one",
    )

    n_matched = len(harmonized)
    n_excluded = int(
        (harmonized["incluir_modelo"] != 1).sum()
    )

    harmonized = harmonized.loc[
        harmonized["incluir_modelo"] == 1
    ].copy()

    harmonized = harmonized.rename(
        columns={"label_harmonizada": "classe_harm"}
    )

    keep_columns = list(lulc.columns) + [
        "id_harm",
        "classe_harm",
        "incluir_modelo",
        "obs",
    ]

    keep_columns = list(dict.fromkeys(keep_columns))
    harmonized = harmonized[keep_columns].copy()

    summary = {
        "feicoes_recortadas": len(lulc),
        "feicoes_com_correspondencia": n_matched,
        "feicoes_excluidas_modelo": n_excluded,
        "feicoes_harmonizadas": len(harmonized),
        "classes_sem_correspondencia": len(classes_missing),
    }

    return harmonized, classes_missing, summary

"""Criação de harminização"""
def split_codes(value):
    if pd.isna(value):
        return []

    value = str(value).strip()

    if value in ["", "-", "—", "nan", "None"]:
        return []

    return [code.strip() for code in value.split(";") if code.strip()]


def harm_id(value):
    value = str(value).strip().replace("H", "") 
    return int(value)


def build_lookup(df, source_col, product):
    rows = []

    for _, row in df.iterrows():
        for code in split_codes(row[source_col]):
            rows.append({
                "product": product,
                "code_original": code,
                "id_harm": harm_id(row["cod_harm"]),
                "label_harmonizada": row["classe_harmonizada"],
                "incluir_modelo": int(str(row["uso_modelo"]).strip() == "Incluir"),
                "obs": row["uso_modelo"],
            })

    return pd.DataFrame(rows)


