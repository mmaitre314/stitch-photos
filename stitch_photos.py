#!/usr/bin/env python3
"""
Stitch a series of overlapping photos of a flat document into one image.

Usage:
    python stitch_document.py page1.jpg page2.jpg page3.jpg -o result.jpg
    python stitch_document.py photos/*.jpg -o result.jpg

Requires: pip install opencv-python
"""

import argparse
import sys
import cv2


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


def stitch(images):
    """
    Stitch using SCANS mode — designed for flat documents/scans.
    It uses an affine transform model instead of the rotational
    (panorama) model, so it won't warp the page like a 360 pano.
    """
    stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
    status, result = stitcher.stitch(images)

    if status == cv2.Stitcher_OK:
        return result

    errors = {
        cv2.Stitcher_ERR_NEED_MORE_IMGS:
            "Not enough matching features between images. "
            "Make sure adjacent photos overlap by 30-40%.",
        cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL:
            "Could not align the images. Try photos with more overlap, "
            "sharper focus, or more consistent distance from the page.",
        cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL:
            "Alignment adjustment failed. Try reducing perspective tilt "
            "between shots.",
    }
    sys.exit(f"Stitching failed: {errors.get(status, f'unknown error {status}')}")


def crop_black_borders(img, threshold=10):
    """Trim the black border left around the stitched result."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    return img[y:y + h, x:x + w]


def main():
    parser = argparse.ArgumentParser(description="Stitch document photos into one image.")
    parser.add_argument("images", nargs="+",
                        help="Input JPEGs, in order (top to bottom of the document)")
    parser.add_argument("-o", "--output", default="stitched.jpg",
                        help="Output JPEG path (default: stitched.jpg)")
    parser.add_argument("--quality", type=int, default=95,
                        help="JPEG quality 1-100 (default: 95)")
    parser.add_argument("--no-crop", action="store_true",
                        help="Skip auto-cropping of black borders")
    args = parser.parse_args()

    if len(args.images) < 2:
        sys.exit("Need at least 2 images to stitch.")

    print(f"Loading {len(args.images)} images...")
    images = load_images(args.images)

    print("Stitching (this can take a minute)...")
    result = stitch(images)

    if not args.no_crop:
        result = crop_black_borders(result)

    cv2.imwrite(args.output, result,
                [cv2.IMWRITE_JPEG_QUALITY, args.quality])
    h, w = result.shape[:2]
    print(f"Done: {args.output} ({w}x{h})")


if __name__ == "__main__":
    main()
