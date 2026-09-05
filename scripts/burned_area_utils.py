"""Preparação das áreas ardidas do ICNF."""

import os
import shutil
from glob import glob
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio



SHAPE_EXTENSIONS = [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix"]


def _remove_shape(path):
    base = Path(path).with_suffix("")

    for ext in SHAPE_EXTENSIONS:
        Path(str(base) + ext).unlink(missing_ok=True)


def _clean(gdf):
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.make_valid()
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)

    return gdf[
        gdf.geometry.notna()
        & ~gdf.geometry.is_empty
        & gdf.geom_type.isin(["Polygon", "MultiPolygon"])
    ].copy()


def prepare_icnf(
    raw_shps,
    aoi,
    reference,
    yearly_folder,
    clipped_folder,
    train_folder,
    valid_folder,
    validation_year=2025
):
    """Cria os vetores anuais de áreas ardidas para treino e validação."""

    folders = [yearly_folder, clipped_folder, train_folder, valid_folder]

    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)

        for shp in Path(folder).glob("aa_*.shp"):
            _remove_shape(shp)

    study_area = gpd.read_file(aoi)[["geometry"]].dissolve()
    boundary = study_area.geometry.iloc[0].boundary

    with rio.open(reference) as src:
        cell_area = abs(src.transform.a * src.transform.e)

    blocks = []

    for shp in raw_shps:
        gdf = gpd.read_file(shp)
        gdf = gdf[
            ((gdf["Ano"] <= 1983) & (gdf["AreaHaSIG"] > 30))
            | ((gdf["Ano"] > 1983) & (gdf["AreaHaSIG"] > 5))
        ][["Ano", "AreaHaSIG", "geometry"]]
        blocks.append(gdf)

    fires = gpd.GeoDataFrame(
        pd.concat(blocks, ignore_index=True),
        geometry="geometry",
        crs=blocks[0].crs
    )

    summary = []

    for year, annual in fires.groupby("Ano"):
        year = int(year)
        yearly = os.path.join(yearly_folder, f"aa_{year}.shp")
        clipped = os.path.join(clipped_folder, f"aa_{year}.shp")

        _remove_shape(yearly)
        annual.to_file(yearly)

        if annual.crs != study_area.crs:
            annual = annual.to_crs(study_area.crs)

        annual_aoi = _clean(gpd.clip(_clean(annual), study_area))
        area = annual_aoi.geometry.area
        boundary_part = annual_aoi.geometry.boundary.intersects(boundary)

        annual_aoi = annual_aoi[
            (area > cell_area)
            & ~(boundary_part & (area <= 5 * cell_area))
        ]

        _remove_shape(clipped)
        annual_aoi.to_file(clipped)

        folder = valid_folder if year == validation_year else train_folder
        destination = os.path.join(folder, f"aa_{year}.shp")
        _remove_shape(destination)
        annual_aoi.to_file(destination)

        summary.append([
            year,
            len(annual),
            len(annual_aoi),
            "validação" if year == validation_year else "treino"
        ])

    return pd.DataFrame(
        summary,
        columns=["ano", "perimetros", "perimetros_aoi", "conjunto"]
    )


def create_count_raster(years, source_folder, temp_folder, reference, output):
    """Cria um raster de contagem com a função original do glass."""

    from glass.rst.stats import count_region_in_shape

    shutil.rmtree(temp_folder, ignore_errors=True)
    Path(temp_folder).mkdir(parents=True)
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    for year in years:
        shp = os.path.join(source_folder, f"aa_{year}.shp")

        if not os.path.exists(shp):
            raise FileNotFoundError(f"Shapefile em falta: {shp}")

        for file in glob(shp.replace(".shp", ".*")):
            shutil.copy(file, temp_folder)

    count_region_in_shape(temp_folder, reference, output)

    with rio.open(output) as src:
        data = src.read(1).astype("int16")
        profile = src.profile.copy()

    data[data == 0] = -1
    profile.update(dtype="int16", nodata=-1)

    with rio.open(output, "w", **profile) as dst:
        dst.write(data, 1)

    shutil.rmtree(temp_folder)

    return output


def create_binary_raster(source, output):
    """Codificação: 1 nas áreas ardidas e -1 nas restantes."""

    with rio.open(source) as src:
        data = src.read(1)
        profile = src.profile.copy()

    binary = np.where(data > 0, 1, -1).astype("int16")
    profile.update(dtype="int16", nodata=-1)

    with rio.open(output, "w", **profile) as dst:
        dst.write(binary, 1)

    return output


def align_outputs(rasters, reference, aoi):
    """Alinha os rasters"""

    from aoi_utils import align_raster

    for raster in rasters:
        temp = raster.replace(".tif", "_tmp.tif")
        align_raster(raster, reference, aoi, temp, "near", -1)
        os.replace(temp, raster)

    return rasters

"""Preparação das áreas ardidas do EFFIS."""
from rasterio.enums import Resampling
from rasterio.shutil import copy as copy_raster
from rasterio.vrt import WarpedVRT
from shapely import make_valid
from shapely.ops import unary_union

def _polygon_only_effis(geom):
    """Repara a geometria e mantém apenas polígonos."""

    if geom is None or geom.is_empty:
        return None

    geom = make_valid(geom)

    if geom.is_empty:
        return None

    if geom.geom_type in ["Polygon", "MultiPolygon"]:
        return geom

    if geom.geom_type == "GeometryCollection":
        polygons = []

        for part in geom.geoms:
            if part.geom_type == "Polygon":
                polygons.append(part)
            elif part.geom_type == "MultiPolygon":
                polygons.extend(part.geoms)

        return unary_union(polygons) if polygons else None

    return None


def _clean_effis(gdf, explode=False):
    """Remove geometrias vazias e não poligonais."""

    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.apply(_polygon_only_effis)

    gdf = gdf[
        gdf.geometry.notna()
        & ~gdf.geometry.is_empty
        & gdf.geom_type.isin(["Polygon", "MultiPolygon"])
    ].copy()

    if explode:
        gdf = gdf.explode(
            index_parts=False
        ).reset_index(drop=True)

    return gdf


def prepare_effis(
    raw_dir,
    aoi,
    reference,
    yearly_folder,
    clipped_folder,
    train_folder,
    valid_folder,
    field_date="initialdat",
    field_area="area_ha",
    start_year=2008,
    end_year=2025,
    validation_year=2025,
    min_area_ha=5
):
    """Prepara os vetores anuais EFFIS."""

    folders = [
        yearly_folder,
        clipped_folder,
        train_folder,
        valid_folder
    ]

    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)

        for shp in Path(folder).glob("aa_*.shp"):
            _remove_shape(shp)

    raw_files = sorted(
        glob(
            os.path.join(raw_dir, "**", "*.shp"),
            recursive=True
        )
    )

    if len(raw_files) != 1:
        raise ValueError(
            f"Era esperado um Shapefile EFFIS, "
            f"mas foram encontrados {len(raw_files)}."
        )

    study_area = gpd.read_file(aoi).dissolve()
    boundary = study_area.geometry.iloc[0].boundary

    with rio.open(reference) as src:
        cell_area = abs(
            src.transform.a * src.transform.e
        )

    fires = gpd.read_file(raw_files[0])

    required = {field_date, field_area, "geometry"}
    missing = required - set(fires.columns)

    if missing:
        raise ValueError(
            f"Campos EFFIS em falta: {sorted(missing)}"
        )

    if fires.crs != study_area.crs:
        fires = fires.to_crs(study_area.crs)

    fires = _clean_effis(fires)

    fires["Ano"] = pd.to_datetime(
        fires[field_date],
        format="%Y-%m-%d",
        errors="coerce"
    ).dt.year

    fires["AreaHaSIG"] = pd.to_numeric(
        fires[field_area],
        errors="coerce"
    )

    fires = fires[
        fires["Ano"].between(start_year, end_year)
        & (fires["AreaHaSIG"] > min_area_ha)
        & fires.geometry.intersects(
            study_area.geometry.iloc[0]
        )
    ][
        ["Ano", field_date, "AreaHaSIG", "geometry"]
    ].copy()

    fires["Ano"] = fires["Ano"].astype(int)

    summary = []

    for year, annual in fires.groupby("Ano"):
        year = int(year)

        yearly = os.path.join(
            yearly_folder,
            f"aa_{year}.shp"
        )

        clipped = os.path.join(
            clipped_folder,
            f"aa_{year}.shp"
        )

        _remove_shape(yearly)
        annual.to_file(yearly)

       
        annual = gpd.read_file(yearly)
        annual = _clean_effis(annual)

        annual_aoi = gpd.clip(
            annual,
            study_area
        )

        annual_aoi = _clean_effis(
            annual_aoi,
            explode=True
        )

        area = annual_aoi.geometry.area

        boundary_part = (
            annual_aoi.geometry.boundary
            .intersects(boundary)
        )

        annual_aoi = annual_aoi[
            (area > cell_area)
            & ~(
                boundary_part
                & (area <= 5 * cell_area)
            )
        ].copy()

        _remove_shape(clipped)
        annual_aoi.to_file(clipped)

        destination_folder = (
            valid_folder
            if year == validation_year
            else train_folder
        )

        destination = os.path.join(
            destination_folder,
            f"aa_{year}.shp"
        )

        _remove_shape(destination)

      
        gpd.read_file(clipped).to_file(destination)

        summary.append({
            "ano": year,
            "perimetros": len(annual),
            "perimetros_aoi": len(annual_aoi),
            "conjunto": (
                "validação"
                if year == validation_year
                else "treino"
            )
        })

    return pd.DataFrame(summary)

def align_outputs_effis(rasters, reference):
    """Alinha os rasters pela grelha de referência."""

    with rio.open(reference) as ref:
        grid = {
            "crs": ref.crs,
            "transform": ref.transform,
            "width": ref.width,
            "height": ref.height,
            "resampling": Resampling.nearest
        }

        for raster in rasters:
            temporary = raster.replace(
                ".tif",
                "_aligned_tmp.tif"
            )

            Path(temporary).unlink(missing_ok=True)

            with rio.open(raster) as src:
                options = {
                    **grid,
                    "nodata": src.nodata,
                    "dtype": src.dtypes[0]
                }

                if src.nodata is not None:
                    options["src_nodata"] = src.nodata

                with WarpedVRT(src, **options) as vrt:
                    copy_raster(
                        vrt,
                        temporary,
                        driver="GTiff",
                        compress="lzw",
                        tiled=True,
                        blockxsize=512,
                        blockysize=512,
                        BIGTIFF="YES"
                    )

            os.replace(temporary, raster)

    return rasters