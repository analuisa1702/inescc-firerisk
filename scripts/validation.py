def _add_freq(freq_dct, vals, freq):
    """
    Add frequencies to dictionary.
    """

    for i in range(len(vals)):
        k = int(vals[i])
        v = int(freq[i])

        freq_dct[k] = freq_dct.get(k, 0) + v


def _rst_windows(src, block_size=1024):
    """
    Generate raster windows.
    """

    from rasterio.windows import Window

    for row in range(0, src.height, block_size):
        h = min(block_size, src.height - row)

        for col in range(0, src.width, block_size):
            w = min(block_size, src.width - col)

            yield Window(col, row, w, h)


def pseudo_roc_blocks(ref, perigo_rst, posval, otbl=None, block_size=1024):
    """
    Oliveira et al. 2021 validation metrics implementation.

    Block-based version to avoid loading full rasters into memory.
    """

    import numpy as np
    import pandas as pd
    import rasterio as rio

    from sklearn import metrics
    from glass.wt import obj_to_tbl

    area_freq = {}
    fire_freq = {}

    with rio.open(ref) as ref_src, rio.open(perigo_rst) as prob_src:

        if ref_src.shape != prob_src.shape:
            raise ValueError("Reference and hazard rasters have different shapes")

        if ref_src.transform != prob_src.transform:
            raise ValueError("Reference and hazard rasters have different transforms")

        if ref_src.crs != prob_src.crs:
            raise ValueError("Reference and hazard rasters have different CRS")

        prob_nd = prob_src.nodata

        for window in _rst_windows(prob_src, block_size=block_size):

            ref_array = ref_src.read(1, window=window)
            prob_array = prob_src.read(1, window=window)

            valid_prob = np.isfinite(prob_array)

            if prob_nd is not None and np.isfinite(float(prob_nd)):
                valid_prob = valid_prob & (prob_array != prob_nd)

            if not valid_prob.any():
                continue

            vals_area = np.rint(
                prob_array[valid_prob] * 10000
            ).astype(np.int32)

            vals, freq = np.unique(vals_area, return_counts=True)
            _add_freq(area_freq, vals, freq)

            valid_fire = valid_prob & (ref_array == posval)

            if not valid_fire.any():
                continue

            vals_fire = np.rint(
                prob_array[valid_fire] * 10000
            ).astype(np.int32)

            refvals, reffreq = np.unique(vals_fire, return_counts=True)
            _add_freq(fire_freq, refvals, reffreq)

    freq_a = pd.DataFrame({
        "vals"    : list(area_freq.keys()),
        "areafreq": list(area_freq.values())
    })

    freq_b = pd.DataFrame({
        "fvals"   : list(fire_freq.keys()),
        "firefreq": list(fire_freq.values())
    })

    ftbl = freq_a.merge(
        freq_b,
        how='left',
        left_on='vals',
        right_on='fvals'
    )

    ftbl.drop('fvals', axis=1, inplace=True)

    ftbl['firefreq'] = np.where(
        ftbl.firefreq.isna(), 0,
        ftbl.firefreq
    )

    ftbl.sort_values(by='vals', ascending=False, inplace=True)
    ftbl.reset_index(inplace=True)

    ftbl["cumarea"] = ftbl.areafreq.cumsum()
    ftbl["cumfire"] = ftbl.firefreq.cumsum()

    areat = ftbl.areafreq.sum()
    firet = ftbl.firefreq.sum()

    if firet == 0:
        raise ValueError("No burned pixels found for validation")

    ftbl["tarearatio"] = ftbl.cumarea / float(areat)
    ftbl["tfireration"] = ftbl.cumfire / float(firet)

    auxtbl = ftbl.copy(deep=True)

    dc = [
        c for c in ftbl.columns.values
        if c != 'tarearatio' and c != 'tfireration'
    ]

    auxtbl.drop(dc, axis=1, inplace=True)
    auxtbl.rename(
        columns={
            'tarearatio' : 'aratio',
            'tfireration': 'fratio'
        },
        inplace=True
    )

    ftbl['lidx'] = ftbl.index
    auxtbl['ridx'] = auxtbl.index + 1

    ftbl = ftbl.merge(
        auxtbl,
        how='left',
        left_on='lidx',
        right_on='ridx'
    )

    ftbl['aratio'] = np.where(
        ftbl.aratio.isna(), 0,
        ftbl.aratio
    )

    ftbl['fratio'] = np.where(
        ~ftbl.fratio.isna(),
        ftbl.fratio, 0
    )

    ftbl["auc_a"] = ftbl['tarearatio'] - ftbl.aratio
    ftbl["auc_b"] = (ftbl['tfireration'] + ftbl.fratio) / 2.0
    ftbl["auc_c"] = ftbl.auc_b * ftbl.auc_a

    auc = ftbl.auc_c.sum()

    realauc = metrics.auc(
        ftbl["tarearatio"],
        ftbl["tfireration"]
    )

    if otbl:
        obj_to_tbl(ftbl, otbl)

        return otbl, realauc, auc

    return ftbl, realauc, auc