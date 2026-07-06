#!/usr/bin/env python3
"""
Stitch a series of overlapping photos of a flat document into one image.

Usage:
    python stitch_document.py page1.jpg page2.jpg page3.jpg -o result.jpg
    python stitch_document.py photos/*.jpg -o result.jpg

Requires: pip install -r requirements.txt
"""

import argparse
import sys
import cv2
import numpy as np
import piexif
from scipy.optimize import least_squares


def load_images(paths, max_dim=3000):
    """Load images, downscaling very large ones to keep stitching fast/stable."""
    images = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            sys.exit(f"Error: could not read image '{p}'")
        h, w = img.shape[:2]
        scale = max_dim / max(h, w)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
            print(f"  {p}: downscaled to {img.shape[1]}x{img.shape[0]}")
        else:
            print(f"  {p}: {w}x{h}")
        images.append(img)
    return images


def _match_all_pairs(images, conf_thresh=1.0):
    """
    Feature-match every pair of images with SIFT, via OpenCV's `cv2.detail`
    matcher (which scores each pair with the same inlier-based confidence
    formula cv2.Stitcher uses internally to decide what's trustworthy).

    ORB, which cv2.Stitcher's high-level API is hardcoded to, finds too few
    reliable matches on flat, low-texture documents (large blank areas with
    small, sparse photos/text). SIFT finds far more robust matches on this
    kind of content.

    Returns (features, matches, pairs) where pairs maps (i, j) -> (H,
    src_pts, dst_pts) for every pair confident enough to trust, H mapping
    points from image i into image j's coordinate frame.
    """
    finder = cv2.SIFT_create(nfeatures=8000)
    features = [cv2.detail.computeImageFeatures2(finder, img) for img in images]

    matcher = cv2.detail.BestOf2NearestMatcher_create(False, 0.3)
    matches = matcher.apply2(features)
    matcher.collectGarbage()

    pairs = {}
    for m in matches:
        i, j = m.src_img_idx, m.dst_img_idx
        if i < 0 or j < 0 or i >= j or m.confidence < conf_thresh:
            continue
        inliers = m.inliers_mask.astype(bool)
        dmatches = [d for d, keep in zip(m.matches, inliers) if keep]
        src = np.float32([features[i].keypoints[d.queryIdx].pt for d in dmatches])
        dst = np.float32([features[j].keypoints[d.trainIdx].pt for d in dmatches])
        pairs[(i, j)] = (m.H, src, dst)

    return features, matches, pairs


def _chain_homographies(n, pairs, ref):
    """
    Breadth-first chain of homographies mapping every connected image into
    `ref`'s coordinate frame, following the strongest-matching path from
    ref. Used as the initial guess for bundle adjustment.
    """
    adjacency = {i: {} for i in range(n)}
    for (i, j), (H, _, _) in pairs.items():
        adjacency[i][j] = H
        adjacency[j][i] = np.linalg.inv(H)

    to_ref = {ref: np.eye(3)}
    frontier = [ref]
    while frontier:
        cur = frontier.pop()
        for nxt, H_nxt_to_cur in adjacency[cur].items():
            if nxt not in to_ref:
                to_ref[nxt] = to_ref[cur] @ H_nxt_to_cur
                frontier.append(nxt)
    return to_ref


def _detect_grid_lines(images, min_len=100, sample_step=40, tol_deg=15):
    """
    Detect the long near-horizontal and near-vertical line segments in each
    image — the family-tree connector rails, page edges, table borders, etc.
    A flat document is full of these, and because they are truly horizontal
    or vertical on the page, they're what lets the alignment lock the whole
    grid straight (see `_bundle_adjust`). Each segment is returned as a list
    of sampled points along it.
    """
    lsd = cv2.createLineSegmentDetector()
    h_segs, v_segs = [], []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected = lsd.detect(gray)[0]
        hs, vs = [], []
        if detected is not None:
            for x1, y1, x2, y2 in detected.reshape(-1, 4):
                length = np.hypot(x2 - x1, y2 - y1)
                if length < min_len:
                    continue
                angle = ((np.degrees(np.arctan2(y2 - y1, x2 - x1)) + 90) % 180) - 90
                t = np.linspace(0, 1, max(2, int(length / sample_step)))
                pts = np.stack([x1 + (x2 - x1) * t, y1 + (y2 - y1) * t], 1)
                if abs(angle) < tol_deg:
                    hs.append(pts)
                elif abs(angle) > 90 - tol_deg:
                    vs.append(pts)
        h_segs.append(hs)
        v_segs.append(vs)
    return h_segs, v_segs


def _rectify_reference(h_segs, v_segs, shape, min_lines=8):
    """
    Build a homography that makes the reference panel fronto-parallel, by
    sending its horizontal and vertical line families to the image axes
    (i.e. their vanishing points to infinity). Holding this panel fixed
    during bundle adjustment both removes its own perspective from the
    mosaic and pins the scale, so the line constraints can't collapse the
    result. Returns None if the panel lacks the line structure to do this
    reliably, in which case stitching falls back to a feature-only fit.
    """
    if len(h_segs) < min_lines or len(v_segs) < min_lines:
        return None

    def vanishing_point(segs):
        ends = [(s[0], s[-1]) for s in segs]
        lengths = np.array([np.hypot(*(b - a)) for a, b in ends])
        coords = np.array([np.cross([a[0], a[1], 1.0], [b[0], b[1], 1.0]) for a, b in ends])
        coords /= np.linalg.norm(coords[:, :2], axis=1, keepdims=True)
        weights = lengths.copy()
        for _ in range(5):
            _, _, vt = np.linalg.svd((coords * weights[:, None]).T @ coords)
            vp = vt[-1]
            resid = np.abs(coords @ vp)
            weights = lengths / (1 + (resid / (np.median(resid) + 1e-9)) ** 2)
        return vp

    vp_h = vanishing_point(h_segs)
    vp_v = vanishing_point(v_segs)

    # Send the vanishing line (horizon through both vanishing points) to
    # infinity, which restores parallelism.
    horizon = np.cross(vp_h, vp_v)
    if abs(horizon[2]) < 1e-12:
        return None
    horizon = horizon / horizon[2]
    proj = np.array([[1, 0, 0], [0, 1, 0], [horizon[0], horizon[1], 1.0]])

    # Then an affine correction that maps the (now finite) line directions
    # onto the x and y axes, so horizontal lines run horizontal and vertical
    # lines run vertical with a right angle between them.
    d_h = proj @ vp_h
    d_h = d_h[:2] / np.linalg.norm(d_h[:2])
    d_v = proj @ vp_v
    d_v = d_v[:2] / np.linalg.norm(d_v[:2])
    if d_h[0] < 0:
        d_h = -d_h  # keep the horizontal axis pointing right
    if d_v[1] < 0:
        d_v = -d_v  # keep the vertical axis pointing down
    affine = np.linalg.inv(np.column_stack([d_h, d_v]))
    rect = np.array([[affine[0, 0], affine[0, 1], 0],
                     [affine[1, 0], affine[1, 1], 0],
                     [0, 0, 1.0]]) @ proj

    # Normalize so the panel keeps its resolution (unit area scale at its
    # center), otherwise rectification can silently shrink the mosaic.
    h, w = shape[:2]
    center = rect @ [w / 2, h / 2, 1]
    center /= center[2]
    jac_x = rect @ [w / 2 + 1, h / 2, 1]
    jac_x /= jac_x[2]
    jac_y = rect @ [w / 2, h / 2 + 1, 1]
    jac_y /= jac_y[2]
    area = abs((jac_x[0] - center[0]) * (jac_y[1] - center[1]) -
               (jac_x[1] - center[1]) * (jac_y[0] - center[0]))
    return np.diag([1 / np.sqrt(area), 1 / np.sqrt(area), 1.0]) @ rect


def _bundle_adjust(n, pairs, base_homographies, ref, rectify, h_segs, v_segs,
                   line_weight=0.7, anchor_weight=0.01):
    """
    Jointly refine every panel's homography so that it both (a) matches
    features across the overlaps and (b) keeps each panel's detected
    horizontal lines level and vertical lines plumb.

    Feature matches alone leave the mosaic wavy: the matched points sit in a
    narrow horizontal band (the photos), so the rows are free to drift up
    and down from panel to panel. The line terms remove exactly that freedom
    and straighten the grid across seams.

    When a fronto-parallel `rectify` for the reference panel is available it
    is held fixed; features then tie the other panels to its scale while the
    line terms flatten everything. Without it (too few lines) this degrades
    to a feature-only fit with the reference held at identity.
    """
    use_lines = rectify is not None
    ref_H = rectify if use_lines else np.eye(3)
    others = [i for i in range(n) if i != ref]
    index = {i: k for k, i in enumerate(others)}

    def pack(Hs):
        return np.concatenate([(Hs[i] / Hs[i][2, 2]).flatten()[:8] for i in others])

    def unpack(p):
        Hs = [None] * n
        Hs[ref] = ref_H
        for i in others:
            Hs[i] = np.array([*p[index[i] * 8:index[i] * 8 + 8], 1.0]).reshape(3, 3)
        return Hs

    def apply_h(H, pts):
        proj = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
        return proj[:, :2] / proj[:, 2:3]

    p_init = pack(base_homographies)

    def residuals(p):
        Hs = unpack(p)
        res = [(apply_h(Hs[i], src) - apply_h(Hs[j], dst)).ravel()
               for (i, j), (_, src, dst) in pairs.items()]
        if use_lines:
            for k in range(n):
                for seg in h_segs[k]:
                    warped = apply_h(Hs[k], seg)
                    res.append(line_weight * (warped[1:, 1] - warped[0, 1]))
                for seg in v_segs[k]:
                    warped = apply_h(Hs[k], seg)
                    res.append(line_weight * (warped[1:, 0] - warped[0, 0]))
            # Soft anchor to the initial guess pins the leftover translation
            # gauge and keeps the optimization stable.
            res.append(anchor_weight * (p - p_init))
        return np.concatenate(res)

    result = least_squares(residuals, p_init, method="lm", max_nfev=100)
    return unpack(result.x)


def _warp_and_blend(images, homographies, seam_scale=0.25, num_bands=5):
    """
    Warp each image into the shared frame, then composite with content-aware
    seams instead of averaging the overlaps.

    Feather (distance-weighted) blending mixes every overlapping pixel from
    all covering photos, so any residual misalignment a homography can't
    model — chiefly lens distortion, worst near frame edges — shows up as
    ghosting/blur on content in the overlaps. Instead a seam finder routes
    the boundary between photos through low-detail regions (the blank page),
    so each headshot is taken whole from a single photo; only a narrow
    multi-band transition is blended to hide the seam. Per-image exposure
    gains even out brightness so the seams don't show.
    """
    # Canvas bounds from all warped corners.
    all_corners = []
    for img, H in zip(images, homographies):
        h, w = img.shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        proj = np.hstack([corners, np.ones((4, 1))]) @ H.T
        all_corners.append(proj[:, :2] / proj[:, 2:3])
    stacked = np.vstack(all_corners)
    min_xy = np.floor(stacked.min(axis=0)).astype(int)
    max_xy = np.ceil(stacked.max(axis=0)).astype(int)
    canvas_w, canvas_h = int(max_xy[0] - min_xy[0]), int(max_xy[1] - min_xy[1])
    translate = np.array([[1, 0, -min_xy[0]], [0, 1, -min_xy[1]], [0, 0, 1.0]])

    # Warp each image (and its coverage mask) into its own tight sub-rect of
    # the canvas, to keep the seam finder and blender memory-light.
    corners, warped, masks = [], [], []
    for img, H in zip(images, homographies):
        h, w = img.shape[:2]
        H = translate @ H
        proj = np.hstack([np.float32([[0, 0], [w, 0], [w, h], [0, h]]),
                          np.ones((4, 1))]) @ H.T
        proj = proj[:, :2] / proj[:, 2:3]
        lo = np.floor(proj.min(axis=0)).astype(int)
        hi = np.ceil(proj.max(axis=0)).astype(int)
        x0, y0 = max(0, lo[0]), max(0, lo[1])
        x1, y1 = min(canvas_w, hi[0]), min(canvas_h, hi[1])
        offset = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1.0]]) @ H
        warped.append(cv2.warpPerspective(img, offset, (x1 - x0, y1 - y0),
                                          flags=cv2.INTER_LINEAR))
        masks.append(cv2.warpPerspective(np.full((h, w), 255, np.uint8), offset,
                                         (x1 - x0, y1 - y0), flags=cv2.INTER_NEAREST))
        corners.append((int(x0), int(y0)))

    # Even out per-image brightness so the seams are invisible.
    compensator = cv2.detail.ExposureCompensator_createDefault(
        cv2.detail.ExposureCompensator_GAIN)
    compensator.feed(corners, warped, masks)
    for i in range(len(images)):
        compensator.apply(i, corners[i], warped[i], masks[i])

    # Find seams on downscaled copies (fast; the boundary is smooth anyway).
    small_imgs = [cv2.resize(w_, (max(1, int(w_.shape[1] * seam_scale)),
                                  max(1, int(w_.shape[0] * seam_scale)))).astype(np.float32)
                  for w_ in warped]
    small_masks = [cv2.resize(m, (max(1, int(m.shape[1] * seam_scale)),
                                  max(1, int(m.shape[0] * seam_scale))),
                              interpolation=cv2.INTER_NEAREST) for m in masks]
    small_corners = [(int(x * seam_scale), int(y * seam_scale)) for x, y in corners]
    seam_masks = cv2.detail_DpSeamFinder("COLOR_GRAD").find(
        small_imgs, small_corners, small_masks)
    seam_masks = [m.get() if hasattr(m, "get") else m for m in seam_masks]

    # Multi-band blend across the narrow seam transitions at full resolution.
    blender = cv2.detail_MultiBandBlender(0, num_bands)
    blender.prepare((0, 0, canvas_w, canvas_h))
    for i in range(len(images)):
        seam = cv2.resize(seam_masks[i], (masks[i].shape[1], masks[i].shape[0]),
                          interpolation=cv2.INTER_LINEAR)
        seam = cv2.bitwise_and(seam, masks[i])
        blender.feed(warped[i].astype(np.int16), seam, corners[i])
    result, _ = blender.blend(None, None)
    return cv2.convertScaleAbs(result)


def stitch(images, paths):
    """Stitch a set of overlapping document photos into one flat mosaic."""
    n = len(images)
    features, matches, pairs = _match_all_pairs(images)

    kept = [int(i) for i in cv2.detail.leaveBiggestComponent(features, matches, 1.0)]
    if len(kept) < n:
        dropped = [paths[i] for i in range(n) if i not in kept]
        sys.exit(
            "Stitching failed: these photos don't share enough matching "
            f"features with the rest to place reliably: {', '.join(dropped)}. "
            "Make sure adjacent photos overlap by 30-40%."
        )

    h_segs, v_segs = _detect_grid_lines(images)
    # Anchor on the panel with the strongest grid in both directions, so
    # rectifying it to fronto-parallel is well-conditioned.
    ref = max(kept, key=lambda i: min(len(h_segs[i]), len(v_segs[i])))
    rectify = _rectify_reference(h_segs[ref], v_segs[ref], images[ref].shape)

    base = rectify if rectify is not None else np.eye(3)
    to_ref = _chain_homographies(n, pairs, ref)
    base_homographies = [base @ to_ref[i] for i in range(n)]
    homographies = _bundle_adjust(n, pairs, base_homographies, ref, rectify,
                                  h_segs, v_segs)
    return _warp_and_blend(images, homographies)


def _level(img, threshold=10):
    """Rotate so the document's top/bottom edges are horizontal, undoing
    the tilt inherited from whichever photo was used as the reference
    frame during stitching."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    def edge_slope(rows_are_top):
        ys = np.where(mask.any(axis=0), mask.argmax(axis=0) if rows_are_top
                      else mask.shape[0] - 1 - mask[::-1].argmax(axis=0), -1)
        xs = np.where(ys >= 0)[0]
        ys = ys[xs]
        coeffs = np.polyfit(xs, ys, 1)
        for _ in range(5):
            resid = ys - np.polyval(coeffs, xs)
            keep = np.abs(resid) < 2 * resid.std() if resid.std() > 0 else slice(None)
            coeffs = np.polyfit(xs[keep], ys[keep], 1)
        return coeffs[0]

    slope = (edge_slope(True) + edge_slope(False)) / 2
    angle = np.degrees(np.arctan(slope))
    if abs(angle) > 10:
        return img  # something's off; don't risk distorting further

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def fill_borders(img, threshold=10):
    """
    Level the document and fill in the remaining black margins (parts of
    the canvas no input photo covered) with plausible background, so the
    output has no black border without cropping away real content.
    """
    img = _level(img, threshold)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, missing = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return cv2.inpaint(img, missing, 5, cv2.INPAINT_TELEA)


def copy_metadata(src_path, dst_path):
    """Copy EXIF metadata (camera info, geolocation, timestamp, ...) from
    a source photo onto the stitched output, since cv2.imwrite drops it."""
    try:
        exif_dict = piexif.load(src_path)
        piexif.insert(piexif.dump(exif_dict), dst_path)
    except Exception as e:
        print(f"Warning: could not copy metadata from '{src_path}': {e}")


def main():
    parser = argparse.ArgumentParser(description="Stitch document photos into one image.")
    parser.add_argument("images", nargs="+",
                        help="Input JPEGs, in order (top to bottom of the document)")
    parser.add_argument("-o", "--output", default="stitched.jpg",
                        help="Output JPEG path (default: stitched.jpg)")
    parser.add_argument("--quality", type=int, default=95,
                        help="JPEG quality 1-100 (default: 95)")
    parser.add_argument("--no-fill", action="store_true",
                        help="Skip leveling and background fill (leave raw perspective and black borders)")
    args = parser.parse_args()

    if len(args.images) < 2:
        sys.exit("Need at least 2 images to stitch.")

    print(f"Loading {len(args.images)} images...")
    images = load_images(args.images)

    print("Stitching (this can take a minute)...")
    result = stitch(images, args.images)

    if not args.no_fill:
        result = fill_borders(result)

    cv2.imwrite(args.output, result,
                [cv2.IMWRITE_JPEG_QUALITY, args.quality])
    copy_metadata(args.images[0], args.output)
    h, w = result.shape[:2]
    print(f"Done: {args.output} ({w}x{h})")


if __name__ == "__main__":
    main()
