import os
import re
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject
from rasterio.enums import Resampling
from dask import delayed, compute
from dask.diagnostics import ProgressBar
from sklearn.metrics import confusion_matrix, cohen_kappa_score

# ============================================================
# GLOBALS (set by Streamlit)
# ============================================================
LEGACY_DIR = None
NEW_DIR    = None
OUT_DIR    = None
SEASON     = None
PIXEL_AREA = 0.01  # hectares

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("CS_VALIDATION")

# ============================================================
# FILENAME PARSER  (matches CSCG040, CSCO040, CSPA040, CSPU040)
# ============================================================
PATTERN = re.compile(r"\d+_(\d{15})_CS([A-Z]{2})\d+_(K\d{3}|R\d{3})\.tif")

def parse(fname):
    m = PATTERN.match(fname)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None

# ============================================================
# LEGACY REFERENCE
# ============================================================
def get_legacy_reference_info(rid):
    for f in LEGACY_DIR.iterdir():
        p = parse(f.name)
        if p and p[0] == rid and p[2] == SEASON:
            with rasterio.open(f) as src:
                return {
                    "crs": src.crs,
                    "transform": src.transform,
                    "shape": src.shape
                }
    return None

# ============================================================
# READ + REPROJECT (pipeline-equivalent)
# ============================================================
def read_and_reproject(path, ref):
    with rasterio.open(path) as src:
        raw = src.read(1)
        nodata = src.nodata if src.nodata is not None else 0

        data = (raw > 0).astype(np.uint8)
        mask = raw != nodata

        if src.crs == ref["crs"] and src.transform == ref["transform"]:
            return data, mask

        dst = np.zeros(ref["shape"], dtype=np.uint8)
        dst_mask = np.zeros(ref["shape"], dtype=np.uint8)

        reproject(data, dst,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=ref["transform"], dst_crs=ref["crs"],
                  resampling=Resampling.nearest)

        reproject(mask.astype(np.uint8), dst_mask,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=ref["transform"], dst_crs=ref["crs"],
                  resampling=Resampling.nearest)

        return dst, dst_mask > 0

# ============================================================
# BUILD STACK
# ============================================================
def build_stack(folder, rid, ref):
    stack = {}
    footprint = None

    for f in folder.iterdir():
        p = parse(f.name)
        if not p:
            continue
        frid, fcrop, fseason = p
        if frid != rid or fseason != SEASON:
            continue

        arr, mask = read_and_reproject(f, ref)
        stack[fcrop] = arr

        if footprint is None:
            footprint = mask
        else:
            footprint &= mask

    return stack, footprint

# ============================================================
# SINGLE RID PROCESSOR
# ============================================================
@delayed
def process_rid(rid):

    ref = get_legacy_reference_info(rid)
    if not ref:
        return None, None

    legacy, fp1 = build_stack(LEGACY_DIR, rid, ref)
    new,    fp2 = build_stack(NEW_DIR,    rid, ref)

    if fp1 is None:
        fp1 = np.ones_like(next(iter(legacy.values())), dtype=bool)
    if fp2 is None:
        fp2 = np.ones_like(next(iter(new.values())), dtype=bool)

    valid = fp1 & fp2

    crops = sorted(set(legacy) & set(new))

    summary_rows = []
    switch = {c:{d:0 for d in crops} for c in crops}

    for crop in crops:
        l = legacy[crop]
        n = new[crop]

        lbin = (l > 0)
        nbin = (n > 0)

        # ---- crop switch masking
        legacy_other = np.zeros_like(lbin, dtype=bool)
        new_other    = np.zeros_like(nbin, dtype=bool)

        for c, arr in legacy.items():
            if c != crop:
                legacy_other |= (arr > 0)

        for c, arr in new.items():
            if c != crop:
                new_other |= (arr > 0)

        crop_switch   = (lbin == 1) & new_other
        crop_replaced = (nbin == 1) & legacy_other

        valid2 = valid & (~crop_switch) & (~crop_replaced)
        if valid2.sum() == 0:
            valid2 = valid

        lvec = lbin[valid2].ravel().astype(np.uint8)
        nvec = nbin[valid2].ravel().astype(np.uint8)

        if len(lvec) == 0:
            continue

        ul = np.unique(lvec)
        un = np.unique(nvec)

        if len(ul)==1 and len(un)==1:
            if ul[0]==un[0]:
                if ul[0]==1:
                    tp=len(lvec); fp=fn=tn=0
                else:
                    tn=len(lvec); tp=fp=fn=0
            else:
                if ul[0]==1:
                    fn=len(lvec); tp=fp=tn=0
                else:
                    fp=len(lvec); tp=fn=tn=0
        else:
            tn, fp, fn, tp = confusion_matrix(lvec, nvec, labels=[0,1]).ravel()

        precision = tp/(tp+fp) if tp+fp else 0
        recall    = tp/(tp+fn) if tp+fn else 0
        f1        = 2*precision*recall/(precision+recall) if precision+recall else 0
        acc       = (tp+tn)/len(lvec)

        try:
            kappa = cohen_kappa_score(lvec, nvec)
        except:
            kappa = 0

        px_l = int((lbin & valid).sum())
        px_n = int((nbin & valid).sum())
        delta = (px_n - px_l)/max(px_l,1)*100

        summary_rows.append({
            "RID": rid, "Crop": crop, "Season": SEASON,
            "TP": tp, "FP": fp, "FN": fn,
            "Precision": precision, "Recall": recall, "F1": f1,
            "Accuracy": acc, "Kappa": kappa,
            "Legacy_px": px_l, "New_px": px_n,
            "Legacy_ha": round(px_l*PIXEL_AREA,2),
            "New_ha": round(px_n*PIXEL_AREA,2),
            "Delta_pct": round(delta,2)
        })

    for c1 in crops:
        for c2 in crops:
            switch[c1][c2] = int(((legacy[c1]>0) & (new[c2]>0) & valid).sum())

    mat = []
    for c in crops:
        tot = sum(switch[c].values())
        row = {"Legacy_Crop":c, **switch[c], "Legacy_Total":tot}
        for cc in crops:
            row[f"{cc}_%"] = round(100*switch[c][cc]/tot,2) if tot else 0
        mat.append(row)

    return summary_rows, mat

# ============================================================
# MAIN
# ============================================================
def main():
    rids = sorted({parse(f.name)[0] for f in LEGACY_DIR.iterdir() if parse(f.name)})
    tasks = [process_rid(r) for r in rids]

    with ProgressBar():
        res = compute(*tasks, scheduler="threads")  # Streamlit-safe

    all_rows = []
    all_mat  = []

    for rows, mat in res:
        if rows:
            all_rows.extend(rows)
        if mat:
            all_mat.extend(mat)

    pd.DataFrame(all_rows).to_csv(OUT_DIR/"RID_Validation_Summary.csv", index=False)
    pd.DataFrame(all_mat).to_csv(OUT_DIR/"RID_CropSwitch_Matrix.csv", index=False)

# ============================================================
# STREAMLIT ENTRY
# ============================================================
def run_validation(legacy_dir, new_dir, output_dir, season):
    global LEGACY_DIR, NEW_DIR, OUT_DIR, SEASON
    LEGACY_DIR = Path(legacy_dir)
    NEW_DIR    = Path(new_dir)
    OUT_DIR    = Path(output_dir)
    SEASON     = season
    OUT_DIR.mkdir(exist_ok=True)
    main()
