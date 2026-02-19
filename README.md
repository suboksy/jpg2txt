# jpg2txt

Render JPG Images As High Quality ASCII Text

Python program accepts jpg image filename as cli input, outputs ASCII rendered text file with same basename.

**Dependencies:** `Pillow` and `numpy` (`pip install pillow numpy`)

**Usage:**
```bash
python jpg2txt.py photo.jpg
```

---

## Notes

The renderer follows the three-layer algorithm:

1. **6-D shape vectors** — each ASCII character is rendered at cell size and described by six lightness-coverage values sampled from a staggered 2×3 grid of overlapping circles. These capture top/bottom, left/right, and middle distinctions (e.g. `^` vs `_`, `p` vs `q`) that a single-luminance approach misses entirely.

2. **Per-dimension normalisation** — the maximum value in each of the six dimensions across all characters is computed, then every character vector is divided by those maxima. This spreads the characters across the full 6-D space so the nearest-neighbour lookup has good discrimination everywhere.

3. **Two-pass contrast enhancement** — before the lookup, each cell's sampling vector goes through:
   - *Directional enhancement*: ten external circles reach into the four neighbouring cells; a lighter neighbour suppresses the matching internal component, sharpening edges.
   - *Global enhancement*: the vector is normalised by its own maximum, raised to a configurable power, then denormalised — exaggerating the shape of boundary-crossing cells.

The result is that edge characters follow contours rather than just approximating luminance, giving noticeably sharper edges compared to standard supersampling-only renderers.

## License

This utility is provided as-is for educational and research purposes (MIT License).

