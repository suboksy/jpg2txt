#!/usr/bin/env python3
"""
jpg2txt.py — High-quality image-to-ASCII renderer.
"""

import sys
import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Tuneable parameters ───────────────────────────────────────────────────────

# Character cell dimensions (pixels).
# Width:height ≈ 1:2 matches the aspect ratio of standard terminal fonts so
# the rendered text preserves the original image proportions.
CELL_W: int = 9
CELL_H: int = 16

# Radius of each sampling circle, as a fraction of CELL_H.
CIRCLE_RADIUS: float = 0.32

# Number of sample points per side of the square grid used to estimate each
# circle's coverage (total samples ≈ SAMPLE_SIDE² × π/4 within the circle).
SAMPLE_SIDE: int = 7   # 7×7 grid  →  ~38 points inside each circle

# Power exponent applied during contrast enhancement.  Higher values produce
# sharper, more graphic edges at the cost of some tonal subtlety.
CONTRAST_EXPONENT: float = 3.5

# Full set of printable ASCII characters (space through ~).
CHARS: str = (
    " !\"#$%&'()*+,-./0123456789:;<=>?@"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
    "abcdefghijklmnopqrstuvwxyz{|}~"
)

# ── Sampling circle layout ────────────────────────────────────────────────────
# Six internal circles in a staggered 2-column × 3-row grid.
# Coordinates are (x, y) normalised to [0, 1] within the cell.
# Left column (x≈0.25) is shifted down slightly; right (x≈0.75) shifted up,
# so the circles tile the cell more uniformly.
INTERNAL_XY = [
    (0.25, 0.20),   # 0: left-top
    (0.75, 0.17),   # 1: right-top
    (0.25, 0.50),   # 2: left-mid
    (0.75, 0.50),   # 3: right-mid
    (0.25, 0.80),   # 4: left-bot
    (0.75, 0.83),   # 5: right-bot
]

# Ten external circles placed just outside the cell boundary.
# They sample lightness from neighbouring cells to detect edges.
EXTERNAL_XY = [
    (0.25, -0.40),  #  0: above-left
    (0.75, -0.40),  #  1: above-right
    (-0.42, 0.20),  #  2: left-upper
    (1.42, 0.20),   #  3: right-upper
    (-0.42, 0.50),  #  4: left-mid
    (1.42, 0.50),   #  5: right-mid
    (-0.42, 0.80),  #  6: left-lower
    (1.42, 0.80),   #  7: right-lower
    (0.25, 1.40),   #  8: below-left
    (0.75, 1.40),   #  9: below-right
]

# Which external circles influence each internal circle's contrast enhancement.
# A light external neighbour "pushes down" the corresponding internal component,
# sharpening the perceived boundary.
AFFECTING_EXT = [
    [0, 1, 2, 4],    # internal 0 (left-top)
    [0, 1, 3, 5],    # internal 1 (right-top)
    [2, 4, 6],       # internal 2 (left-mid)
    [3, 5, 7],       # internal 3 (right-mid)
    [4, 6, 8, 9],    # internal 4 (left-bot)
    [5, 7, 8, 9],    # internal 5 (right-bot)
]

# ── Font helpers ──────────────────────────────────────────────────────────────

def _load_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Attempt to load a TrueType monospace font; fall back to the bitmap default."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
        "/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
        # macOS
        "/System/Library/Fonts/Menlo.ttc",
        "/Library/Fonts/Courier New.ttf",
        # Windows
        "C:/Windows/Fonts/cour.ttf",
        "C:/Windows/Fonts/consola.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, CELL_H)
            except Exception:
                continue
    return ImageFont.load_default()


def _render_char(ch: str, font) -> np.ndarray:
    """
    Render *ch* centred in a CELL_W × CELL_H black image.
    Returns a float32 array in [0, 1] (0 = black, 1 = white).
    """
    img = Image.new("L", (CELL_W * 2, CELL_H * 2), 0)
    draw = ImageDraw.Draw(img)

    if hasattr(font, "getbbox"):
        try:
            bb = font.getbbox(ch)
            ch_w = bb[2] - bb[0]
            ch_h = bb[3] - bb[1]
            x = (CELL_W - ch_w) // 2 - bb[0]
            y = (CELL_H - ch_h) // 2 - bb[1]
        except Exception:
            x, y = 0, 0
    else:
        x, y = 0, 0

    draw.text((x + CELL_W // 2, y + CELL_H // 2), ch, fill=255, font=font)
    # Crop back to the centred cell region
    cropped = img.crop((CELL_W // 2, CELL_H // 2,
                         CELL_W // 2 + CELL_W, CELL_H // 2 + CELL_H))
    return np.array(cropped, dtype=np.float32) / 255.0


# ── Circle sampling helpers ───────────────────────────────────────────────────

def _make_circle_offsets(radius_px: float, n_side: int) -> np.ndarray:
    """
    Pre-compute (dy, dx) pixel offsets for sample points within a circle.
    Returns an (N, 2) float32 array of offsets relative to the circle centre.
    """
    offsets = []
    r = radius_px
    step = 2 * r / max(n_side - 1, 1)
    for i in range(n_side):
        for j in range(n_side):
            dx = -r + step * j
            dy = -r + step * i
            if dx * dx + dy * dy <= r * r:
                offsets.append((dy, dx))
    return np.array(offsets, dtype=np.float32) if offsets else np.array([[0.0, 0.0]])


def _sample_circles_in_array(arr: np.ndarray,
                               centres: list[tuple[float, float]],
                               offsets_list: list[np.ndarray]) -> np.ndarray:
    """
    Sample *arr* at each circle described by *centres* + *offsets_list*.
    Returns a float32 vector of average lightness values.

    arr       : 2-D float32 image (H × W)
    centres   : list of (cx, cy) in pixel coordinates
    offsets   : per-circle (N, 2) array of (dy, dx) offsets
    """
    h, w = arr.shape
    result = np.zeros(len(centres), dtype=np.float32)
    for idx, (cx, cy) in enumerate(centres):
        offs = offsets_list[idx]
        xs = np.clip(np.round(cx + offs[:, 1]).astype(np.int32), 0, w - 1)
        ys = np.clip(np.round(cy + offs[:, 0]).astype(np.int32), 0, h - 1)
        result[idx] = arr[ys, xs].mean()
    return result


# ── Shape vector computation ─────────────────────────────────────────────────

def _build_character_table(font) -> tuple[list[str], np.ndarray]:
    """
    Compute a normalised 6-D shape vector for every character in CHARS.

    Returns (chars_list, shape_matrix) where shape_matrix is (N_chars, 6).
    """
    radius_px = CIRCLE_RADIUS * CELL_H
    offsets = [_make_circle_offsets(radius_px, SAMPLE_SIDE)] * 6   # same grid for all

    # Internal circle centres inside the character cell (in pixels)
    int_centres_px = [(cx * CELL_W, cy * CELL_H) for cx, cy in INTERNAL_XY]

    chars_list: list[str] = []
    vecs: list[np.ndarray] = []
    for ch in CHARS:
        arr = _render_char(ch, font)
        vec = _sample_circles_in_array(arr, int_centres_px, offsets)
        chars_list.append(ch)
        vecs.append(vec)

    shape_matrix = np.stack(vecs, axis=0)  # (N, 6)

    # Normalise: divide each dimension by its maximum across all characters
    dim_max = shape_matrix.max(axis=0)
    dim_max = np.where(dim_max == 0, 1.0, dim_max)
    shape_matrix /= dim_max

    return chars_list, shape_matrix, dim_max


# ── Contrast enhancement ──────────────────────────────────────────────────────

def _apply_contrast(internal: np.ndarray,
                    external: np.ndarray,
                    exponent: float) -> np.ndarray:
    """
    Apply directional then global contrast enhancement to *internal*.

    internal : (6,)  normalised sampling vector for this cell
    external : (10,) normalised external sampling vector
    exponent : power applied to normalised component values
    """
    result = internal.copy()

    # ── Directional contrast enhancement ─────────────────────────────────────
    # For each component, find the maximum lightness among the external circles
    # that "reach in" from the corresponding direction.  If a neighbour is
    # brighter, the component is suppressed to sharpen the edge boundary.
    for i, ext_indices in enumerate(AFFECTING_EXT):
        ext_vals = external[ext_indices]
        max_val = max(float(result[i]), float(ext_vals.max()))
        if max_val > 0:
            v = result[i] / max_val
            v = v ** exponent
            result[i] = v * max_val

    # ── Global contrast enhancement ───────────────────────────────────────────
    # Normalise by the sampling vector's own maximum so that uniform regions
    # are preserved while transitions between light and dark are exaggerated.
    max_val = float(result.max())
    if max_val > 0:
        result = result / max_val
        result = result ** exponent
        result = result * max_val

    return result


# ── Main rendering pipeline ───────────────────────────────────────────────────

def image_to_ascii(image_path: str) -> str:
    # ── Load & convert to luminance ───────────────────────────────────────────
    img = Image.open(image_path).convert("RGB")
    rgb = np.array(img, dtype=np.float32) / 255.0
    # Relative luminance (ITU-R BT.709)
    gray = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]

    img_h, img_w = gray.shape
    n_rows = img_h // CELL_H
    n_cols = img_w // CELL_W
    print(f"  Source   : {img_w}×{img_h} px")
    print(f"  Cell     : {CELL_W}×{CELL_H} px")
    print(f"  Grid     : {n_cols} cols × {n_rows} rows")

    # ── Build character shape table ───────────────────────────────────────────
    print("  Building character shape vectors …")
    font = _load_font()
    chars_list, shape_matrix, dim_max = _build_character_table(font)
    # shape_matrix : (N_chars, 6) — normalised

    # Pre-compute sampling offsets (same radius for every circle)
    radius_px = CIRCLE_RADIUS * CELL_H
    circle_offsets = [_make_circle_offsets(radius_px, SAMPLE_SIDE)
                      for _ in range(len(INTERNAL_XY) + len(EXTERNAL_XY))]

    # ── Render each cell ──────────────────────────────────────────────────────
    print("  Rendering …")
    output_lines: list[str] = []

    for row in range(n_rows):
        line: list[str] = []
        oy = row * CELL_H
        for col in range(n_cols):
            ox = col * CELL_W

            # Build pixel-space centre lists for internal + external circles
            int_centres = [(ox + cx * CELL_W, oy + cy * CELL_H)
                           for cx, cy in INTERNAL_XY]
            ext_centres = [(ox + cx * CELL_W, oy + cy * CELL_H)
                           for cx, cy in EXTERNAL_XY]

            int_raw = _sample_circles_in_array(gray, int_centres,
                                                circle_offsets[:6])
            ext_raw = _sample_circles_in_array(gray, ext_centres,
                                                circle_offsets[6:])

            # Contrast enhancement operates entirely in raw [0, 1] space so
            # that internal and external vectors are directly comparable.
            enhanced_raw = _apply_contrast(int_raw, ext_raw, CONTRAST_EXPONENT)

            # Normalise into character-shape space only for the lookup.
            enhanced = enhanced_raw / dim_max

            # Nearest-neighbour character lookup (squared Euclidean distance)
            diffs = shape_matrix - enhanced          # (N_chars, 6)
            dists = (diffs * diffs).sum(axis=1)      # (N_chars,)
            best_idx = int(dists.argmin())
            line.append(chars_list[best_idx])

        output_lines.append("".join(line))

        if (row + 1) % 20 == 0 or (row + 1) == n_rows:
            pct = (row + 1) / n_rows * 100
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            print(f"  [{bar}] {pct:5.1f}%", end="\r")

    print()
    return "\n".join(output_lines)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python text-render.py <image.jpg>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: file not found: {image_path!r}")
        sys.exit(1)

    base, _ = os.path.splitext(image_path)
    output_path = base + ".txt"

    print(f"\nASCII Renderer  (shape-vector method)")
    print(f"{'─' * 45}")
    ascii_art = image_to_ascii(image_path)

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(ascii_art)

    lines = ascii_art.count("\n") + 1
    cols = len(ascii_art.splitlines()[0]) if ascii_art else 0
    print(f"  Output   : {output_path}  ({cols} × {lines} chars)")
    print(f"{'─' * 45}\n")


if __name__ == "__main__":
    main()
