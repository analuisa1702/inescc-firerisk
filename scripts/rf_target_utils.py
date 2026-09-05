"""
Funções auxiliares para criar os rasters alvo do Random Forest.
"""

import os

import numpy as np
import rasterio as rio


def _valid_mask(array, nodata):
    """Identifica as células válidas do raster de referência."""

    valid = np.isfinite(array)

    if nodata is not None:
        try:
            nodata_is_nan = np.isnan(nodata)
        except TypeError:
            nodata_is_nan = False

        if not nodata_is_nan:
            valid = valid & (array != nodata)

    return valid


def _burned_mask(array, nodata):
    """Identifica as células com uma ou mais ocorrências de incêndio."""

    burned = np.isfinite(array) & (array > 0)

    if nodata is not None:
        try:
            nodata_is_nan = np.isnan(nodata)
        except TypeError:
            nodata_is_nan = False

        if not nodata_is_nan:
            burned = burned & (array != nodata)

    return burned


def count_to_rf_target_blocks(
    count_rst,
    ref_rst,
    out_rst,
    nodata=-1,
    block_size=512
):
    """
    Converte um raster de contagem num raster alvo para Random Forest.

    O processamento é realizado por blocos.

    Valores de saída:
    -1 = NoData ou área não elegível
     0 = área elegível não ardida
     1 = área elegível ardida
    """

    os.makedirs(os.path.dirname(out_rst), exist_ok=True)

    if os.path.exists(out_rst):
        os.remove(out_rst)

    counts = {}

    with rio.open(ref_rst) as ref_src, rio.open(count_rst) as count_src:
        if (
            ref_src.width != count_src.width
            or ref_src.height != count_src.height
        ):
            raise ValueError("Os rasters não têm a mesma dimensão.")

        if not ref_src.transform.almost_equals(
            count_src.transform,
            precision=1e-8
        ):
            raise ValueError("Os rasters não têm a mesma geotransformação.")

        if ref_src.crs != count_src.crs:
            raise ValueError(
                "Os rasters não têm o mesmo sistema de coordenadas."
            )

        profile = ref_src.profile.copy()
        profile.update(
            driver="GTiff",
            dtype="int16",
            nodata=nodata,
            count=1,
            compress="lzw",
            tiled=True,
            blockxsize=block_size,
            blockysize=block_size,
            BIGTIFF="YES"
        )

        ref_nodata = ref_src.nodata
        count_nodata = count_src.nodata

        print("Raster de referência:", ref_rst)
        print("Raster de contagem:", count_rst)

        with rio.open(out_rst, "w", **profile) as dst:
            for row_off in range(0, ref_src.height, block_size):
                win_height = min(
                    block_size,
                    ref_src.height - row_off
                )

                for col_off in range(0, ref_src.width, block_size):
                    win_width = min(
                        block_size,
                        ref_src.width - col_off
                    )

                    window = rio.windows.Window(
                        col_off=col_off,
                        row_off=row_off,
                        width=win_width,
                        height=win_height
                    )

                    ref_array = ref_src.read(
                        1,
                        window=window
                    )
                    count_array = count_src.read(
                        1,
                        window=window
                    )

                    valid = _valid_mask(
                        ref_array,
                        ref_nodata
                    )
                    burned = _burned_mask(
                        count_array,
                        count_nodata
                    )

                    target = np.full(
                        ref_array.shape,
                        nodata,
                        dtype=np.int16
                    )

                    target[valid] = 0
                    target[valid & burned] = 1

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

    print("Raster alvo criado:", out_rst)
    print("Valores escritos:", counts)

    return out_rst


def check_target_blocks(raster):
    """Verifica os valores do raster alvo sem o carregar integralmente."""

    counts = {}

    with rio.open(raster) as src:
        for _, window in src.block_windows(1):
            array = src.read(
                1,
                window=window
            )

            values, frequencies = np.unique(
                array,
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

        print("Raster:", raster)
        print("NoData:", src.nodata)
        print("Dimensão:", src.width, src.height)
        print("CRS:", src.crs)
        print("Valores:", counts)

    unexpected = set(counts) - {-1, 0, 1}

    if unexpected:
        print(
            "ATENÇÃO: existem valores inesperados:",
            unexpected
        )
    else:
        print(
            "OK: o raster só tem valores -1, 0 e 1."
        )

    return counts
