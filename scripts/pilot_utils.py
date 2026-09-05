"""
Funções simples para preparar limites municipais e recortar rasters.
"""

from pathlib import Path
import unicodedata


def _normalize_text(value):
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(
        char for char in text
        if not unicodedata.combining(char)
    )

    return text.strip().casefold()


def _read_vector(source, layer=None):
    """Lê um GeoPackage, Shapefile ou pasta com um Shapefile."""

    import geopandas as gpd

    source = Path(source)

    if not source.exists():
        raise FileNotFoundError(f"Limite não encontrado: {source}")

    if source.is_dir():
        files = sorted(
            path for path in source.rglob("*.shp")
            if not path.name.startswith("._")
        )

        if not files:
            raise FileNotFoundError(
                f"Não existe Shapefile em: {source}"
            )

        source = files[0]

    return (
        gpd.read_file(source, layer=layer)
        if layer
        else gpd.read_file(source)
    )


def select_municipality(
    source,
    field,
    value,
    output,
    layer=None,
    expected_area_km2=None
):
    """Seleciona e dissolve um município, guardando-o num GeoPackage."""

    import geopandas as gpd

    output = Path(output)
    gdf = _read_vector(source, layer=layer)

    if gdf.crs is None:
        raise ValueError("O limite municipal não tem CRS definido.")

    if field not in gdf.columns:
        raise KeyError(
            f"O campo '{field}' não existe. "
            f"Campos: {list(gdf.columns)}"
        )

    target = _normalize_text(value)
    names = gdf[field].map(_normalize_text)
    municipality = gdf.loc[names == target, [field, "geometry"]].copy()

    if municipality.empty:
        raise ValueError(
            f"O município '{value}' não foi encontrado em '{field}'."
        )

    municipality = municipality[
        municipality.geometry.notna()
        & ~municipality.geometry.is_empty
    ].dissolve()

    if municipality.empty:
        raise ValueError("O limite municipal não contém geometria válida.")

    area_gdf = municipality

    if not municipality.crs.is_projected:
        projected_crs = municipality.estimate_utm_crs()

        if projected_crs is None:
            raise ValueError(
                "Não foi possível determinar um CRS projectado "
                "para calcular a área municipal."
            )

        area_gdf = municipality.to_crs(projected_crs)

    area_km2 = float(
        area_gdf.geometry.area.sum() / 1_000_000
    )

    if expected_area_km2 and area_km2 is not None:
        min_area, max_area = expected_area_km2

        if not min_area <= area_km2 <= max_area:
            raise ValueError(
                f"Área municipal inesperada: {area_km2:.3f} km². "
                f"Intervalo esperado: {min_area}-{max_area} km²."
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    municipality.to_file(output, layer="municipality", driver="GPKG")

    print("Município:", value)
    print("Output:", output)

    if area_km2 is not None:
        print("Área:", round(area_km2, 3), "km²")

    return str(output)


def clip_raster(raster, vector, output, layer="municipality"):
    """Recorta um raster pela geometria de um município."""

    import geopandas as gpd
    import numpy as np
    import rasterio as rio
    from rasterio.mask import mask

    raster = Path(raster)
    vector = Path(vector)
    output = Path(output)

    for path in [raster, vector]:
        if not path.exists():
            raise FileNotFoundError(f"Ficheiro não encontrado: {path}")

    municipality = gpd.read_file(vector, layer=layer)

    with rio.open(raster) as src:
        municipality = municipality.to_crs(src.crs)

        nodata = src.nodata

        if nodata is None:
            nodata = (
                0
                if np.issubdtype(np.dtype(src.dtypes[0]), np.integer)
                else -9999.0
            )

        data, transform = mask(
            src,
            municipality.geometry,
            crop=True,
            filled=True,
            nodata=nodata
        )

        profile = src.profile.copy()
        profile.update(
            height=data.shape[1],
            width=data.shape[2],
            transform=transform,
            nodata=nodata,
            compress="lzw",
            BIGTIFF="IF_SAFER"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)

    with rio.open(output, "w", **profile) as dst:
        dst.write(data)

        try:
            with rio.open(raster) as src:
                colormap = src.colormap(1)
            dst.write_colormap(1, colormap)
        except ValueError:
            pass

    return str(output)
