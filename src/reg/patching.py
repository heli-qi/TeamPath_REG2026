"""Array-based WSI patching — port of wsi_patching.py (segment_tissue + grid), reading
from the tifffile level-0 array instead of openslide (REG2026 tiffs are 20x base; many
also carry a downsample pyramid, which we exploit for the segmentation thumbnail).

Speedups vs the original streaming reader (both give identical patch pixels):
  - thumbnail: read a pyramid overview level when present (else downsample level-0 strip-by-
    strip). Avoids decompressing the whole multi-GB level-0 just to build a 2048px thumbnail.
  - patch read: read each patch's own tile region (z0[y:y+fp, x:x+fp]) instead of a full-width
    row strip — for a sparse capped patch set this decodes far fewer tiles.
  - downsample: read a (patch_size*downsample) field of view and area-resize to patch_size,
    giving 10x patches (downsample=2) from the 20x level-0 for multi-scale models.
"""
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2

cv2.setNumThreads(1)


def _decode_threads(threads):
    if threads and threads > 0:
        return threads
    return max(1, min(8, (os.cpu_count() or 1)))


def segment_tissue(thumb_rgb, sat_thresh=0, close_k=7, min_frac=5e-4):
    """uint8 binary tissue mask (255=tissue) at thumbnail scale — identical to wsi_patching."""
    hsv = cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2HSV)
    sat = cv2.medianBlur(hsv[:, :, 1], 7)
    if sat_thresh and sat_thresh > 0:
        _, mask = cv2.threshold(sat, sat_thresh, 255, cv2.THRESH_BINARY)
    else:
        _, mask = cv2.threshold(sat, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    k = np.ones((close_k, close_k), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 1:
        min_area = min_frac * mask.shape[0] * mask.shape[1]
        keep = np.zeros_like(mask)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                keep[lab == i] = 255
        mask = keep
    return mask


def patch_wsi_array(arr, patch_size=256, tissue_thresh=0.25, seg_thumb=2048, sat_thresh=0):
    """arr: [H,W,3|4] uint8 level-0 (==20x). Returns imgs [N,256,256,3] uint8, coords [N,2] int32.
    Same tissue-fraction>=thresh grid as wsi_patching at base magnification (step == patch_size)."""
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, 2)
    arr = np.ascontiguousarray(arr[:, :, :3])
    H, W = arr.shape[:2]

    # tissue mask on a downsampled thumbnail (longest side ~ seg_thumb)
    sc = max(W, H) / seg_thumb
    th_w, th_h = max(1, int(W / sc)), max(1, int(H / sc))
    thumb = cv2.resize(arr, (th_w, th_h), interpolation=cv2.INTER_AREA)
    sx, sy = W / th_w, H / th_h
    mask = segment_tissue(thumb, sat_thresh)

    fp = patch_size
    coords = []
    for y in range(0, max(1, H - fp + 1), fp):
        ty0, ty1 = int(y / sy), int((y + fp) / sy) + 1
        for x in range(0, max(1, W - fp + 1), fp):
            tx0, tx1 = int(x / sx), int((x + fp) / sx) + 1
            sub = mask[ty0:ty1, tx0:tx1]
            if sub.size and (sub > 0).mean() >= tissue_thresh:
                coords.append((x, y))
    coords = np.array(coords, np.int32) if coords else np.zeros((0, 2), np.int32)

    N = len(coords)
    imgs = np.zeros((N, fp, fp, 3), np.uint8)
    for i, (x, y) in enumerate(coords):
        patch = arr[int(y):int(y) + fp, int(x):int(x) + fp]
        if patch.shape[:2] != (fp, fp):  # edge tile: pad
            p = np.zeros((fp, fp, 3), np.uint8); p[:patch.shape[0], :patch.shape[1]] = patch
            patch = p
        imgs[i] = patch
    return imgs, coords, mask


def _open_levels(path):
    """Return the pyramid levels of a tiled tiff as a list of zarr arrays, biggest first.
    Single-level slides return a 1-element list ([level-0])."""
    import tifffile, zarr
    grp = zarr.open(tifffile.imread(path, aszarr=True), mode="r")
    if hasattr(grp, "shape"):        # single level -> plain zarr array
        return [grp]
    return [grp[i] for i in range(len(grp))]   # multi-level -> zarr Group


def _open_level0(path):
    """Backward-compatible: lazy level-0 zarr view of a (possibly multi-GB) tiled tiff."""
    return _open_levels(path)[0]


def _build_thumb(levels, H, W, th_w, th_h, nthr, max_strip_bytes):
    """Segmentation thumbnail (th_h, th_w, 3). Reads a pyramid overview level when one is
    >= the thumbnail target; otherwise downsamples level-0 strip-by-strip (parallel)."""
    z0 = levels[0]
    seg_long = max(th_w, th_h)
    pyr = None
    for z in levels:                              # levels big->small; keep smallest still >= target
        if max(int(z.shape[0]), int(z.shape[1])) >= seg_long:
            pyr = z
    if pyr is not None and pyr is not z0:
        arr = np.ascontiguousarray(np.asarray(pyr[:, :, :3]))
        return cv2.resize(arr, (th_w, th_h), interpolation=cv2.INTER_AREA)
    # fallback: no usable pyramid -> stream level-0
    thumb = np.zeros((th_h, th_w, 3), np.uint8)
    strip_rows = max(1, max_strip_bytes // (max(1, W) * 3))

    def _thumb_strip(y0):
        y1 = min(H, y0 + strip_rows)
        oy0, oy1 = int(round(y0 * th_h / H)), int(round(y1 * th_h / H))
        if oy1 <= oy0:
            return None
        sub = np.ascontiguousarray(np.asarray(z0[y0:y1, :, :3]))
        return oy0, oy1, cv2.resize(sub, (th_w, oy1 - oy0), interpolation=cv2.INTER_AREA)

    with ThreadPoolExecutor(max_workers=nthr) as ex:
        for r in ex.map(_thumb_strip, range(0, H, strip_rows)):
            if r is not None:
                oy0, oy1, t = r
                thumb[oy0:oy1] = t
    return thumb


def patch_wsi_path(path, patch_size=256, tissue_thresh=0.25, seg_thumb=2048,
                   sat_thresh=0, max_strip_bytes=512 * 1024 * 1024,
                   cap=None, cap_seed=0, threads=None, downsample=1):
    """Memory-bounded patching straight from a tiled tiff. Never materializes the full
    level-0 array. `downsample` reads a (patch_size*downsample) field of view and area-resizes
    to patch_size (downsample=1 -> 20x, downsample=2 -> 10x). If cap is given and N>cap, coords
    are sub-sampled with np.sort(RandomState(cap_seed).choice(N, cap)) *before* pixels are read.
    Returns imgs [N,patch_size,patch_size,3] uint8, coords [N,2] int32 (level-0 px), mask."""
    levels = _open_levels(path)
    z0 = levels[0]
    H, W = int(z0.shape[0]), int(z0.shape[1])
    fp = patch_size * downsample                  # field of view in level-0 pixels
    nthr = _decode_threads(threads)

    # thumbnail dims fixed by seg_thumb so the grid mapping is identical regardless of source
    sc = max(W, H) / seg_thumb
    th_w, th_h = max(1, int(W / sc)), max(1, int(H / sc))
    thumb = _build_thumb(levels, H, W, th_w, th_h, nthr, max_strip_bytes)
    sx, sy = W / th_w, H / th_h
    mask = segment_tissue(thumb, sat_thresh)

    # grid coords — identical loop to patch_wsi_array (step == fp)
    coords = []
    for y in range(0, max(1, H - fp + 1), fp):
        ty0, ty1 = int(y / sy), int((y + fp) / sy) + 1
        for x in range(0, max(1, W - fp + 1), fp):
            tx0, tx1 = int(x / sx), int((x + fp) / sx) + 1
            sub = mask[ty0:ty1, tx0:tx1]
            if sub.size and (sub > 0).mean() >= tissue_thresh:
                coords.append((x, y))
    coords = np.array(coords, np.int32) if coords else np.zeros((0, 2), np.int32)

    # cap #patches before reading pixels (random subset, fixed seed -> reproducible)
    if cap and len(coords) > cap:
        sel = np.sort(np.random.RandomState(cap_seed).choice(len(coords), cap, replace=False))
        coords = coords[sel]

    # per-patch tile reads: each patch decodes only the tiles it overlaps (parallel)
    N = len(coords)
    imgs = np.zeros((N, patch_size, patch_size, 3), np.uint8)

    def _one(i):
        x, y = int(coords[i][0]), int(coords[i][1])
        patch = np.asarray(z0[y:min(y + fp, H), x:min(x + fp, W), :3])
        if patch.shape[:2] != (fp, fp):           # edge tile: pad to full FoV
            q = np.zeros((fp, fp, 3), np.uint8); q[:patch.shape[0], :patch.shape[1]] = patch
            patch = q
        if downsample > 1:
            patch = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_AREA)
        return i, patch

    with ThreadPoolExecutor(max_workers=nthr) as ex:
        for i, patch in ex.map(_one, range(N)):
            imgs[i] = patch
    return imgs, coords, mask
