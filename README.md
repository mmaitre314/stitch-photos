# stitch-photos

Stitch a series of overlapping photos of a flat document into one image.

## Quick start

1. Open this repo in a GitHub Codespace.
2. Upload your photos (in order, top to bottom of the document) into the
   `data/` folder — drag and drop them onto it in the Explorer, or
   right-click `data/` and choose **Upload...**.
3. Run the stitch:
   ```bash
   source .venv/bin/activate
   python stitch_photos.py data/*.jpg -o data/result.jpg
   ```
4. Download the output — right-click `data/result.jpg` in the Explorer and
   choose **Download...**.

Adjacent photos should overlap by 30-40% for reliable alignment.

### Options

| Flag | Description |
| --- | --- |
| `-o`, `--output` | Output JPEG path (default: `stitched.jpg`) |
| `--quality` | JPEG quality 1-100 (default: `95`) |
| `--no-fill` | Skip leveling and background fill (leave raw perspective and black borders) |

## Manual setup

If you're not using the devcontainer/Codespace, set up the virtualenv
yourself:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This installs the pinned dependencies in `requirements.txt`: `opencv-python-headless`
(the headless build has no GUI/display dependencies, so no `libGL` system
package is needed), `scipy` (for aligning the photos), and `piexif` (for
copying EXIF metadata onto the stitched output).

`data/` is ignored by git, so it's a convenient place to keep input photos
without tracking them.
