import re
import numpy as np

_HEADER_RE = re.compile(r"/\*\s*===\s*(.+?)\s*===\s*\*/")
_BW_RE = re.compile(r"_bw\[EMOJI_ROWS\]\s*=\s*\{(.*?)\};", re.DOTALL)
_COLOR_RE = re.compile(
    r"_color\[EMOJI_ROWS\]\[EMOJI_COLS\]\s*=\s*\{(.*?)\n\};", re.DOTALL
)
_DEF_RE = re.compile(r"#define\s+EMOJI_(ROWS|COLS)\s+(\d+)")
_HEX_RE = re.compile(r"0x[0-9a-fA-F]+")


def _read_dims(text):
    dims = {m.group(1): int(m.group(2)) for m in _DEF_RE.finditer(text)}
    return dims["ROWS"], dims["COLS"]


def _bw_to_bitmap(row_masks, rows, cols):
    """Unpack row bitmasks (MSB = leftmost column) into a (rows, cols) array."""
    bitmap = np.zeros((rows, cols), dtype=np.float32)
    for r, mask in enumerate(row_masks):
        for c in range(cols):
            bitmap[r, c] = (mask >> (cols - 1 - c)) & 1
    return bitmap


def _color_to_bitmap(rgb_vals, rows, cols):
    """Turn a flat list of 0xRRGGBB ints into a (rows, cols, 3) float array."""
    arr = np.array(rgb_vals, dtype=np.uint32).reshape(rows, cols)
    r = (arr >> 16) & 0xFF
    g = (arr >> 8) & 0xFF
    b = arr & 0xFF
    return np.stack([r, g, b], axis=-1).astype(np.float32) / 255.0


def load_emojis(path):
    """Load emojis.h.

    Returns
    -------
    X_color : ndarray (n, ROWS*COLS*3) float32 in [0, 1]
        Flattened RGB images, ready to train a color autoencoder.
    X_bw : ndarray (n, ROWS*COLS) float32 in {0, 1}
        Flattened binary silhouettes (same spirit as font.h).
    bitmaps_color : ndarray (n, ROWS, COLS, 3) float32 in [0, 1]
    bitmaps_bw : ndarray (n, ROWS, COLS) float32 in {0, 1}
    labels : list of str
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    rows, cols = _read_dims(text)

    labels = _HEADER_RE.findall(text)
    bw_blocks = _BW_RE.findall(text)
    color_blocks = _COLOR_RE.findall(text)

    if not (len(labels) == len(bw_blocks) == len(color_blocks)):
        raise ValueError(
            "Parse mismatch: %d labels, %d bw, %d color blocks"
            % (len(labels), len(bw_blocks), len(color_blocks))
        )

    bitmaps_bw = []
    bitmaps_color = []
    for bw_body, color_body in zip(bw_blocks, color_blocks):
        bw_vals = [int(h, 16) for h in _HEX_RE.findall(bw_body)]
        color_vals = [int(h, 16) for h in _HEX_RE.findall(color_body)]
        bitmaps_bw.append(_bw_to_bitmap(bw_vals, rows, cols))
        bitmaps_color.append(_color_to_bitmap(color_vals, rows, cols))

    bitmaps_bw = np.array(bitmaps_bw, dtype=np.float32)
    bitmaps_color = np.array(bitmaps_color, dtype=np.float32)

    X_bw = bitmaps_bw.reshape(len(bitmaps_bw), -1)
    X_color = bitmaps_color.reshape(len(bitmaps_color), -1)

    return X_color, X_bw, bitmaps_color, bitmaps_bw, labels


def print_char_bw(bitmap, label=""):
    """Print a binary emoji with block characters (like font_loader)."""
    if label:
        print(f"[ {label} ]")
    for row in bitmap:
        print("".join("██" if p else "  " for p in row))
    print()


def print_char_color(bitmap_rgb, label=""):
    """Print a color emoji using ANSI 24-bit truecolor background blocks."""
    if label:
        print(f"[ {label} ]")
    for row in bitmap_rgb:
        line = []
        for px in row:
            r, g, b = (int(round(v * 255)) for v in px)
            line.append(f"\x1b[48;2;{r};{g};{b}m  \x1b[0m")
        print("".join(line))
    print()


def print_all(bitmaps_color, bitmaps_bw, labels):
    for i, label in enumerate(labels):
        print_char_color(bitmaps_color[i], label=f"{label}  (idx {i})  [color]")
        print_char_bw(bitmaps_bw[i], label=f"{label}  (idx {i})  [b&w]")
