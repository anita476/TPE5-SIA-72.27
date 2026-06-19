import os
import sys
import unicodedata

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Configuration ────────────────────────────────────────────────────────────
SIZE = 20            # output grid is SIZE x SIZE
RENDER = 160         # glyph size; rendered large, then downscaled
MARGIN = 0.08        # blank border around the emoji, as a fraction of SIZE
ALPHA_THRESHOLD = 0.5  # alpha >= this -> pixel belongs to the emoji
LUMA_THRESHOLD = 0.6  # luminance < this -> pixel is "on" (black ink)

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\seguiemj.ttf",          # Windows: Segoe UI Emoji
    "/System/Library/Fonts/Apple Color Emoji.ttc",  # macOS
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",  # Linux (Noto)
]

EMOJIS = [
    "\U0001F600",  # grinning face
    "\U0001F603",  # grinning face with big eyes
    "\U0001F604",  # grinning face with smiling eyes
    "\U0001F601",  # beaming face with smiling eyes
    "\U0001F606",  # grinning squinting face
    "\U0001F605",  # grinning face with sweat
    "\U0001F602",  # face with tears of joy
    "\U0001F642",  # slightly smiling face
    "\U0001F609",  # winking face
    "\U0001F60A",  # smiling face with smiling eyes
    "\U0001F60D",  # smiling face with heart-eyes
    "\U0001F60E",  # smiling face with sunglasses
    "\U0001F62E",  # face with open mouth
    "\U0001F622",  # crying face
    "\U0001F62D",  # loudly crying face
    "\U0001F620",  # angry face
    "\U0001F631",  # face screaming in fear
    "\U0001F634",  # sleeping face
    "\U0001F61C",  # winking face with tongue
    "\U0001F914",  # thinking face,
    "\U0001F923",  # rolling on the floor laughing
    "\U0001F607",  # smiling face with halo
    "\U0001F970",  # smiling face with hearts
    "\U0001F618",  # face blowing a kiss
    "\U0001F617",  # kissing face
    "\U0001F928",  # face with raised eyebrow
    "\U0001F610",  # neutral face
    "\U0001F611",  # expressionless face
    "\U0001F615",  # confused face
    "\U0001F644",  # face with rolling eyes
    "\U0001F623",  # persevering face
    "\U0001F625",  # sad but relieved face
    "\U0001F643",  # upside-down face
    "\U0001F973",  # partying face
    "\U0001F929",  # star-struck
    "\U0001F92A",  # zany face
    "\U0001F60B",  # face savoring food
    "\U0001F61B",  # face with tongue
    "\U0001F61D",  # squinting face with tongue
    "\U0001F612",  # unamused face
    "\U0001F613",  # downcast face with sweat
    "\U0001F614",  # pensive face
    "\U0001F616",  # confounded face
    "\U0001F61F",  # worried face
    "\U0001F621",  # pouting face
    "\U0001F624",  # face with steam from nose
    "\U0001F62C",  # grimacing face
    "\U0001F633",  # flushed face
]


def _load_font():
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, RENDER), path
            except OSError:
                continue
    raise SystemExit(
        "No color emoji font found. Tried:\n  " + "\n  ".join(FONT_CANDIDATES)
    )


def _identifier(emoji):
    """Build a valid C identifier from the emoji's Unicode name."""
    try:
        name = unicodedata.name(emoji[0])
    except ValueError:
        name = "emoji_%x" % ord(emoji[0])
    ident = name.lower().replace(" ", "_").replace("-", "_")
    return "".join(ch for ch in ident if ch.isalnum() or ch == "_")


def render_emoji(emoji, font):
    """Return (rgb, alpha) arrays of shape (SIZE, SIZE).

    rgb   : uint8, the emoji composited over white.
    alpha : float32 in [0, 1], opacity of the emoji.
    """
    canvas = int(RENDER * 1.6)
    img = Image.new("RGBA", (canvas, canvas), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    draw.text((canvas // 2, canvas // 2), emoji, font=font,
              anchor="mm", embedded_color=True)

    bbox = img.getchannel("A").getbbox()
    if bbox is not None:
        img = img.crop(bbox)
    side = int(max(img.size) / (1.0 - 2.0 * MARGIN))
    square = Image.new("RGBA", (side, side), (255, 255, 255, 0))
    square.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
    img = square.resize((SIZE, SIZE), Image.LANCZOS)

    arr = np.array(img, dtype=np.float32)          # (SIZE, SIZE, 4)
    alpha = arr[..., 3] / 255.0
    rgb = arr[..., :3] * alpha[..., None] + 255.0 * (1.0 - alpha[..., None])
    return rgb.round().astype(np.uint8), alpha


def to_bw_rows(rgb, alpha):
    """Pack the binary emoji into SIZE row bitmasks (MSB = leftmost col).

    A pixel is on (1) where the emoji is opaque and its luminance is dark,
    i.e. the expressive features (eyes, mouth, brows, tears) become black ink
    on a white background.
    """
    luma = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1]
            + 0.0722 * rgb[..., 2]) / 255.0
    ink = (luma < LUMA_THRESHOLD) & (alpha >= ALPHA_THRESHOLD)
    rows = []
    for r in range(SIZE):
        bits = 0
        for c in range(SIZE):
            bits <<= 1
            if ink[r, c]:
                bits |= 1
        rows.append(bits)
    return rows


def emit(emojis, out_path):
    font, font_path = _load_font()
    hex_width = (SIZE + 3) // 4

    lines = [
        "/* Auto-generated emoji dataset. DO NOT EDIT BY HAND. */",
        "/* Regenerate with: python src/utils/emoji_to_h.py */",
        "/* Source font: %s */" % font_path,
        "",
        "#define EMOJI_ROWS %d" % SIZE,
        "#define EMOJI_COLS %d" % SIZE,
        "",
    ]

    for emoji in emojis:
        ident = _identifier(emoji)
        codepoints = "+".join("U+%04X" % ord(ch) for ch in emoji)
        try:
            label = unicodedata.name(emoji[0]).lower()
        except ValueError:
            label = ident
        rgb, alpha = render_emoji(emoji, font)
        bw_rows = to_bw_rows(rgb, alpha)

        lines.append("/* === %s  %s === */" % (codepoints, label))

        lines.append("static const unsigned int %s_bw[EMOJI_ROWS] = {" % ident)
        body = ", ".join("0x%0*X" % (hex_width, v) for v in bw_rows)
        lines.append("    %s" % body)
        lines.append("};")

        lines.append(
            "static const unsigned int %s_color[EMOJI_ROWS][EMOJI_COLS] = {"
            % ident
        )
        for r in range(SIZE):
            row_vals = []
            for c in range(SIZE):
                rr, gg, bb = rgb[r, c]
                row_vals.append("0x%02X%02X%02X" % (rr, gg, bb))
            lines.append("    {%s}," % ", ".join(row_vals))
        lines.append("};")
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Wrote %d emojis to %s" % (len(emojis), out_path))


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "emojis.h"
    )
    out_path = os.path.normpath(out_path)
    emit(EMOJIS, out_path)


if __name__ == "__main__":
    main()
