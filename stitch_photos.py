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


def _bundle_adjust(n, pairs, to_ref, ref):
    """
    Jointly refine every image's homography-to-reference by minimizing
    reprojection error across all confident pairwise matches at once
    (not just the chain used for the initial guess), which spreads out
    the drift that would otherwise accumulate frame-to-frame.
    """
    others = [i for i in range(n) if i in to_ref and i != ref]
    index = {i: k for k, i in enumerate(others)}

    def pack(Hs):
        return np.concatenate([(Hs[i] / Hs[i][2, 2]).flatten()[:8] for i in others])

    def unpack(p):
        Hs = {ref: np.eye(3)}
        for i in others:
            h = p[index[i] * 8:index[i] * 8 + 8]
            Hs[i] = np.array([*h[:8], 1.0]).reshape(3, 3)
        return Hs

    def apply_h(H, pts):
        proj = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
        return proj[:, :2] / proj[:, 2:3]

    def residuals(p):
        Hs = unpack(p)
        res = [apply_h(Hs[i], src) - apply_h(Hs[j], dst)
               for (i, j), (_, src, dst) in pairs.items()
               if i in Hs and j in Hs]
        return np.concatenate([r.ravel() for r in res])

    p0 = pack(to_ref)
    result = least_squares(residuals, p0, method="lm", max_nfev=200)
    return unpack(result.x)


def _warp_and_blend(images, homographies):
    """Warp each image into the shared reference frame and feather-blend
    the overlaps by distance to each image's own edge."""
    all_corners = []
    for img, H in homographies.items():
        h, w = images[img].shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        proj = np.hstack([corners, np.ones((4, 1))]) @ H.T
        all_corners.append(proj[:, :2] / proj[:, 2:3])
    all_corners = np.vstack(all_corners)
    min_xy = np.floor(all_corners.min(axis=0)).astype(int)
    max_xy = np.ceil(all_corners.max(axis=0)).astype(int)
    canvas_w, canvas_h = (max_xy - min_xy)
    translate = np.array([[1, 0, -min_xy[0]], [0, 1, -min_xy[1]], [0, 0, 1]])

    acc = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    weight = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    for img_idx, H in homographies.items():
        img = images[img_idx]
        H = translate @ H
        warped = cv2.warpPerspective(img, H, (canvas_w, canvas_h),
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        mask = np.full(img.shape[:2], 255, np.uint8)
        warped_mask = cv2.warpPerspective(mask, H, (canvas_w, canvas_h),
                                          flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        dist = cv2.distanceTransform(warped_mask, cv2.DIST_L2, 5)
        acc += warped.astype(np.float32) * dist[..., None]
        weight += dist

    weight[weight == 0] = 1
    return np.clip(acc / weight[..., None], 0, 255).astype(np.uint8)


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

    ref = kept[len(kept) // 2]
    to_ref = _chain_homographies(n, pairs, ref)
    homographies = _bundle_adjust(n, pairs, to_ref, ref)
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
