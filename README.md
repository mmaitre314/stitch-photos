# stitch-photos

Stitch a series of overlapping photos of a flat document into one image.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This creates a `.venv` and installs `opencv-python-headless` as pinned in
`requirements.txt`. The headless build has no GUI/display dependencies, so
no `libGL` system package is needed.

## Usage

Activate the virtualenv, then run `stitch_photos.py`, passing the photos in
order (top to bottom of the document) and an output path:

```bash
source .venv/bin/activate
python stitch_photos.py page1.jpg page2.jpg page3.jpg -o result.jpg
python stitch_photos.py photos/*.jpg -o result.jpg
```

### Options

| Flag | Description |
| --- | --- |
| `-o`, `--output` | Output JPEG path (default: `stitched.jpg`) |
| `--quality` | JPEG quality 1-100 (default: `95`) |
| `--no-crop` | Skip auto-cropping of black borders left by stitching |

Adjacent photos should overlap by 30-40% for reliable alignment.

Place input photos in `data/` (ignored by git) if you don't want to track them
elsewhere.
