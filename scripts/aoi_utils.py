"""
Funções auxiliares para preparar a área de estudo.
"""


def select_aoi(source, field, value, epsg, output, layer=None):
    """Seleciona, reprojeta e guarda a área de estudo."""

    import geopandas as gpd

    gdf = gpd.read_file(source, layer=layer) if layer else gpd.read_file(source)
    aoi = gdf.loc[gdf[field] == value, [field, "geometry"]].copy()

    if len(aoi) != 1:
        raise ValueError(f"Foram encontradas {len(aoi)} entidades para {value}.")

    aoi = aoi.rename(columns={field: "area"}).to_crs(epsg)
    aoi.to_file(output)

    return aoi


def create_reference(aoi, grid, mask, cellsize, epsg):
    """Cria a grelha de referência e a máscara da área de estudo."""

    from glass.dtt.rst.torst import shpext_to_rst, shp_to_rst

    shpext_to_rst(aoi, grid, cellsize=cellsize, epsg=epsg)

    shp_to_rst(
        aoi,
        1,
        None,
        0,
        mask,
        rst_template=grid,
        api="gdal",
        dtype="Byte",
        rtype=int
    )

    return grid, mask


def align_raster(raster, reference, clip, output, method="bilinear", nodata=-9999):
    """Alinha e recorta um raster pela grelha de referência."""

    from shlex import quote

    from glass.prop.prj import rst_epsg
    from glass.prop.rst import rst_fullprop
    from glass.pys import execmd

    extent, _, shape = rst_fullprop(reference)
    left, right, bottom, top = extent
    rows, cols = shape
    epsg = rst_epsg(reference)

    cmd = (
        f"gdalwarp -overwrite -t_srs EPSG:{epsg} "
        f"-te {left} {bottom} {right} {top} -ts {cols} {rows} "
        f"-r {method} -dstnodata {nodata} -cutline {quote(str(clip))} "
        f"-of GTiff -co COMPRESS=LZW -co TILED=YES -co BIGTIFF=IF_NEEDED "
        f"{quote(str(raster))} {quote(str(output))}"
    )

    execmd(cmd)

    return output


def check_alignment(rasters, reference):
    """Confirma o alinhamento de um ou mais rasters."""

    from glass.prop.prj import rst_epsg
    from glass.prop.rst import rst_fullprop
    from glass.pys import obj_to_lst

    ref_prop = rst_fullprop(reference)
    ref_epsg = rst_epsg(reference)

    for raster in obj_to_lst(rasters):
        if rst_fullprop(raster) != ref_prop or rst_epsg(raster) != ref_epsg:
            raise ValueError(f"O raster não está alinhado: {raster}")

        print("Raster alinhado:", raster)

    return True
