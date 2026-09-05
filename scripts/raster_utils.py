"""
Funções auxiliares para preparação e alinhamento de rasters.
"""

from pathlib import Path

import numpy as np
import rasterio as rio

from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window


def check_raster_alignment(raster, template):
    """
    Confirma que um raster está alinhado com um raster de referência.
    """

    with rio.open(template) as ref:
        with rio.open(raster) as src:
            differences = []

            if src.crs != ref.crs:
                differences.append("CRS")

            if src.transform != ref.transform:
                differences.append("transformação")

            if src.shape != ref.shape:
                differences.append("dimensão")

            if differences:
                raise ValueError(
                    f"O raster {raster} não está alinhado com {template}. "
                    f"Diferenças: {', '.join(differences)}."
                )

            print("Raster alinhado:", raster)
            print("CRS:", src.crs)
            print("Resolução:", src.res)
            print("Dimensão:", src.shape)
            print("NoData:", src.nodata)

    return True


def align_raster_to_template(
    source_raster,
    template_raster,
    mask_raster,
    output_raster,
    resampling="bilinear",
    nodata=-9999.0,
    mask_value=1,
    block_size=1024
):
    """
    Reprojeta e alinha um raster a uma grelha de referência.

    O raster é processado por blocos e limitado pela máscara indicada.
    """

    source_raster = Path(source_raster)
    template_raster = Path(template_raster)
    mask_raster = Path(mask_raster)
    output_raster = Path(output_raster)

    for path in [source_raster, template_raster, mask_raster]:
        if not path.exists():
            raise FileNotFoundError(f"Ficheiro não encontrado: {path}")

    if not hasattr(Resampling, resampling):
        raise ValueError(f"Método de reamostragem inválido: {resampling}")

    resampling_method = getattr(Resampling, resampling)

    check_raster_alignment(
        raster=str(mask_raster),
        template=str(template_raster)
    )

    output_raster.parent.mkdir(parents=True, exist_ok=True)
    output_raster.unlink(missing_ok=True)

    with rio.open(source_raster) as src:
        with rio.open(template_raster) as ref:
            with rio.open(mask_raster) as mask_src:
                profile = ref.profile.copy()

                profile.update(
                    driver="GTiff",
                    dtype="float32",
                    count=1,
                    nodata=nodata,
                    compress="lzw",
                    predictor=3,
                    tiled=True,
                    blockxsize=512,
                    blockysize=512,
                    BIGTIFF="YES"
                )

                vrt_options = {
                    "crs": ref.crs,
                    "transform": ref.transform,
                    "width": ref.width,
                    "height": ref.height,
                    "resampling": resampling_method,
                    "nodata": nodata,
                    "dtype": "float32",
                    "warp_mem_limit": 128
                }

                if src.nodata is not None:
                    vrt_options["src_nodata"] = src.nodata

                n_block_rows = (
                    ref.height + block_size - 1
                ) // block_size

                n_block_cols = (
                    ref.width + block_size - 1
                ) // block_size

                total_blocks = n_block_rows * n_block_cols
                block_number = 0

                with WarpedVRT(src, **vrt_options) as vrt:
                    with rio.open(output_raster, "w", **profile) as dst:
                        for row_off in range(0, ref.height, block_size):
                            height = min(
                                block_size,
                                ref.height - row_off
                            )

                            for col_off in range(0, ref.width, block_size):
                                width = min(
                                    block_size,
                                    ref.width - col_off
                                )

                                window = Window(
                                    col_off=col_off,
                                    row_off=row_off,
                                    width=width,
                                    height=height
                                )

                                raster_block = vrt.read(
                                    1,
                                    window=window,
                                    out_dtype="float32"
                                )

                                mask_block = mask_src.read(
                                    1,
                                    window=window
                                )

                                invalid = (
                                    (mask_block != mask_value)
                                    | ~np.isfinite(raster_block)
                                )

                                raster_block[invalid] = nodata

                                dst.write(
                                    raster_block,
                                    1,
                                    window=window
                                )

                                block_number += 1

                                if (
                                    block_number % 50 == 0
                                    or block_number == total_blocks
                                ):
                                    print(
                                        "Blocos:",
                                        f"{block_number}/{total_blocks}"
                                    )

    check_raster_alignment(
        raster=str(output_raster),
        template=str(template_raster)
    )

    return str(output_raster)


""" rasterização """
def raster_ids(raster, nodata=0):
    """Obtém os identificadores presentes sem carregar todo o raster."""

    ids = set()

    with rio.open(raster) as src:
        for _, window in src.block_windows(1):
            values = np.unique(
                src.read(1, window=window)
            )
            ids.update(values.tolist())

    return sorted(
        value
        for value in ids
        if value != nodata
    )