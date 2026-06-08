import re
import numpy as np

ROWS = 7
COLS = 5


def _parse_font_h(path):
    """Parse header file and return (raw_data, labels)
    """
    raw_data = []
    labels = []

    with open(path, "r") as f:
        for line in f:
            # Only process lines that have a { ... } block with hex values
            brace_match = re.search(r'\{([^}]+)\}', line)
            if not brace_match:
                continue
            hex_vals = [int(h, 16) for h in re.findall(r'0x([0-9a-fA-F]+)', brace_match.group(1))]
            if len(hex_vals) != ROWS:
                continue  # skip the outer array declaration line

            label_match = re.search(r'//\s*0x[0-9a-f]+,\s*(.+)', line)
            label = label_match.group(1).strip() if label_match else '?'

            raw_data.append(hex_vals)
            labels.append(label)

    return raw_data, labels


def _to_bitmap(hex_row):
    bitmap = np.zeros((ROWS, COLS), dtype=np.float32)
    for r, val in enumerate(hex_row):
        for c in range(COLS):
            bitmap[r, COLS - 1 - c] = (val >> c) & 1
    return bitmap


def load_font(path):
    """Load font.h and return the training matrix and metadata.

    Parameters
    ----------
    path : str
        Path to the font.h file.

    Returns
    -------
    X : ndarray, shape (n_chars, ROWS*COLS) = (32, 35)
        Flattened binary bitmaps, float32 in {0, 1}. Ready to pass to train().
    bitmaps : ndarray, shape (n_chars, ROWS, COLS) = (32, 7, 5)
        2D grids, useful for visualisation.
    labels : list of str
        Character label for each row of X.
    """
    raw_data, labels = _parse_font_h(path)
    bitmaps = np.array([_to_bitmap(row) for row in raw_data], dtype=np.float32)
    X = bitmaps.reshape(len(bitmaps), -1)
    return X, bitmaps, labels



def print_char(flat_or_grid, label=""):
    """Pretty-print a character to the console using block characters."""
    grid = flat_or_grid.reshape(ROWS, COLS)
    if label:
        print(f"[ {label} ]")
    for row in grid:
        print(" ".join("█" if p else "·" for p in row))
    print()


def print_all(X, labels):
    """Print every character in the dataset."""
    for i, (x, label) in enumerate(zip(X, labels)):
        print_char(x, label=f"{label}  (idx {i})")

