def split_rst_radomly_byblocks(
    inrst, proportion, random_rst, other_rst=None,
    min_sample=None, absbycls=None, block_size=1024
):
    """
    Extract some cells of one raster and save them into a new raster.

    The cells not selected for extraction may be exported to other raster.
    This version reads the raster by blocks.
    """

    import numpy as np
    from osgeo import gdal

    from glass.prop.df  import drv_name
    from glass.prop.rst import compress_option

    img = gdal.Open(inrst, gdal.GA_ReadOnly)

    if img is None:
        raise ValueError(f'Could not open raster: {inrst}')

    bnd = img.GetRasterBand(1)
    nd = bnd.GetNoDataValue()

    rows = img.RasterYSize
    cols = img.RasterXSize
    dtype = bnd.DataType

    # Count cells by value
    val_count = {}

    for yoff in range(0, rows, block_size):
        nrows = min(block_size, rows - yoff)

        for xoff in range(0, cols, block_size):
            ncols = min(block_size, cols - xoff)

            arr = bnd.ReadAsArray(xoff, yoff, ncols, nrows)

            if nd is None:
                vals, cnts = np.unique(arr, return_counts=True)
            else:
                vals, cnts = np.unique(arr[arr != nd], return_counts=True)

            for i in range(vals.shape[0]):
                val_count[vals[i].item()] = \
                    val_count.get(vals[i].item(), 0) + int(cnts[i])

    if min_sample:
        val_count = {
            k: v for k, v in val_count.items() if v > min_sample
        }

    if not val_count:
        raise ValueError('No cells available for sampling.')

    val = np.array(sorted(val_count.keys()))

    # Number of cells to select by class
    if not absbycls:
        ncells_byval = {
            v: int(round(val_count[v] * proportion / 100.0, 0))
            for v in val
        }

    else:
        ncells_byval = {
            v: int(proportion[v])
            for v in val if v in proportion
        }

    for v in ncells_byval:
        if ncells_byval[v] > val_count[v]:
            raise ValueError(
                f'Class {v} has {val_count[v]} cells, '
                f'but {ncells_byval[v]} were requested.'
            )

    val = np.array(sorted(ncells_byval.keys()))

    # Random positions inside each class
    rnd_pos = {
        v: np.sort(np.random.choice(
            val_count[v],
            size=ncells_byval[v],
            replace=False
        ))
        for v in val
    }

    def create_rst(out_rst):
        drv = drv_name(out_rst)
        driver = gdal.GetDriverByName(drv)

        c_opt = compress_option(drv)
        options = [c_opt] if c_opt else []

        if drv == 'GTiff':
            options.append('BIGTIFF=YES')

        out = driver.Create(
            out_rst, cols, rows, 1, dtype, options=options
        )

        out.SetGeoTransform(img.GetGeoTransform())
        out.SetProjection(img.GetProjection())

        obnd = out.GetRasterBand(1)

        if nd is not None:
            obnd.SetNoDataValue(nd)
            obnd.Fill(nd)

        return out, obnd

    rnd_img, rnd_bnd = create_rst(random_rst)

    if other_rst:
        oth_img, oth_bnd = create_rst(other_rst)
    else:
        oth_img, oth_bnd = None, None

    # Number of cells already seen by class
    seen = {v: 0 for v in val}

    for yoff in range(0, rows, block_size):
        nrows = min(block_size, rows - yoff)

        for xoff in range(0, cols, block_size):
            ncols = min(block_size, cols - xoff)

            arr = bnd.ReadAsArray(xoff, yoff, ncols, nrows)
            flat = arr.ravel()

            res = np.full(arr.shape, nd, dtype=arr.dtype)
            res_flat = res.ravel()

            if other_rst:
                nres = np.full(arr.shape, nd, dtype=arr.dtype)
                nres_flat = nres.ravel()

            selected = np.zeros(flat.shape, dtype=bool)

            for v in val:
                idx = np.flatnonzero(flat == v)
                nidx = idx.size

                if not nidx:
                    continue

                start = seen[v]
                end = seen[v] + nidx

                p = rnd_pos[v]
                lo = np.searchsorted(p, start, side='left')
                hi = np.searchsorted(p, end, side='left')

                if hi > lo:
                    local_pos = idx[p[lo:hi] - start]
                    res_flat[local_pos] = v
                    selected[local_pos] = True

                seen[v] = end

            if other_rst:
                if nd is None:
                    valid = np.ones(flat.shape, dtype=bool)
                else:
                    valid = flat != nd

                nres_flat[valid & ~selected] = flat[valid & ~selected]

            rnd_bnd.WriteArray(res, xoff, yoff)

            if other_rst:
                oth_bnd.WriteArray(nres, xoff, yoff)

    rnd_bnd.FlushCache()
    rnd_img.FlushCache()

    rnd_bnd = None
    rnd_img = None

    if other_rst:
        oth_bnd.FlushCache()
        oth_img.FlushCache()

        oth_bnd = None
        oth_img = None

    img = None

    return random_rst, other_rst



def rf_mean_prob_byblocks(
    mdl_list, imgvar, outrst, class_val=1,
    block_size=2048, predict_batch_size=250000,
    nodata=101, n_jobs=-1
):
    """
    Apply several Random Forest models by blocks and write the mean
    probability of one class.

    This avoids writing individual classifications/probability rasters
    for each model.
    """

    import os
    import datetime as dt
    import numpy as np

    from osgeo import gdal
    from joblib import load

    from glass.prop.df import drv_name
    from glass.prop.rst import compress_option

    srcs = [gdal.Open(r, gdal.GA_ReadOnly) for r in imgvar]

    if any(s is None for s in srcs):
        raise ValueError('At least one feature raster could not be opened.')

    ref = srcs[0]

    rows = ref.RasterYSize
    cols = ref.RasterXSize
    gtrans = ref.GetGeoTransform()
    proj = ref.GetProjection()

    for src in srcs[1:]:
        if src.RasterYSize != rows or src.RasterXSize != cols:
            raise ValueError('Feature rasters have different shapes.')

        if src.GetGeoTransform() != gtrans:
            raise ValueError('Feature rasters have different transforms.')

        if src.GetProjection() != proj:
            raise ValueError('Feature rasters have different projections.')

    nds = [
        src.GetRasterBand(1).GetNoDataValue()
        for src in srcs
    ]

    models = []
    cls_idx = []

    for mdl in mdl_list:
        rf = load(mdl)

        if hasattr(rf, 'set_params'):
            rf.set_params(n_jobs=n_jobs)

        classes = list(rf.classes_)

        if class_val not in classes:
            raise ValueError(
                f'Class {class_val} not found in model classes: {classes}'
            )

        models.append(rf)
        cls_idx.append(classes.index(class_val))

    os.makedirs(os.path.dirname(outrst), exist_ok=True)

    drv = drv_name(outrst)
    driver = gdal.GetDriverByName(drv)

    if os.path.exists(outrst):
        driver.Delete(outrst)

    c_opt = compress_option(drv)
    options = [c_opt] if c_opt else []

    if drv == 'GTiff':
        options.append('BIGTIFF=YES')

    out = driver.Create(
        outrst,
        cols,
        rows,
        1,
        gdal.GDT_Float32,
        options=options
    )

    out.SetGeoTransform(gtrans)
    out.SetProjection(proj)

    out_bnd = out.GetRasterBand(1)
    out_bnd.SetNoDataValue(nodata)
    out_bnd.Fill(nodata)

    total_blocks = 0
    done_blocks = 0

    for yoff in range(0, rows, block_size):
        for xoff in range(0, cols, block_size):
            total_blocks += 1

    time_a = dt.datetime.now().replace(microsecond=0)

    for yoff in range(0, rows, block_size):
        nrows = min(block_size, rows - yoff)

        for xoff in range(0, cols, block_size):
            ncols = min(block_size, cols - xoff)

            arrays = []
            valid = None

            for src, nd in zip(srcs, nds):
                arr = src.GetRasterBand(1).ReadAsArray(
                    xoff, yoff, ncols, nrows
                ).astype(np.float32)

                v = np.isfinite(arr)

                if nd is not None:
                    v = v & (arr != nd)

                valid = v if valid is None else valid & v
                arrays.append(arr)

            out_arr = np.full(
                (nrows, ncols),
                nodata,
                dtype=np.float32
            )

            if valid.any():
                x = np.stack(
                    [a[valid] for a in arrays],
                    axis=1
                ).astype(np.float32)

                mean_prob = np.zeros(
                    x.shape[0],
                    dtype=np.float32
                )

                for ini in range(0, x.shape[0], predict_batch_size):
                    end = min(ini + predict_batch_size, x.shape[0])
                    xb = x[ini:end]

                    prob_sum = np.zeros(
                        xb.shape[0],
                        dtype=np.float32
                    )

                    for rf, cidx in zip(models, cls_idx):
                        prob_sum += rf.predict_proba(xb)[:, cidx].astype(np.float32)

                    mean_prob[ini:end] = prob_sum / float(len(models))

                out_arr[valid] = mean_prob

            out_bnd.WriteArray(out_arr, xoff, yoff)

            done_blocks += 1

            time_b = dt.datetime.now().replace(microsecond=0)

            print(
                f'Bloco {done_blocks}/{total_blocks} | '
                f'linha {yoff + nrows}/{rows} | '
                f'tempo {time_b - time_a}'
            )

    out_bnd.FlushCache()
    out.FlushCache()

    out_bnd = None
    out = None

    for src in srcs:
        src = None

    return outrst