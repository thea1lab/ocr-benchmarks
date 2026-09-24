"""Draw the benchmark pages and write the transcripts they are scored against."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent
PAGES = ROOT / "pages"
SANS = "/usr/share/fonts/noto/NotoSans-Regular.ttf"
SANS_BOLD = "/usr/share/fonts/noto/NotoSans-Bold.ttf"
MONO = "/usr/share/fonts/noto/NotoSansMono-Regular.ttf"

LETTER = """Northwind Supply
14 March 2026

Dear Mina,

Please confirm the delivery for order 1842. The crates should arrive at Dock 3 before 16:00. Two of the crates are marked fragile.

Thank you,
Helena Costa
Accounts"""

RECEIPT = """NORTHWIND MARKET
14 March 2026
12:41

Oat milk            3.40
Sourdough           4.50
Coffee beans       10.50

Subtotal           18.40
Tax                 0.00
Total              18.40

Card payment
Thank you"""

LEFT = """Harbor Notes

The crane arrived on Tuesday. Order 1842 is still open. Dock 3 closes at 16:00.

Inspectors counted the crates twice. The left column ends here."""

RIGHT = """Spare parts

Crate A holds bolts. Crate B holds oil. Paint is stored in crate C.

The right column ends here."""

NOTE = "Note 1. The tide turns at 18:40, so the barge must leave before then."

COLUMNS = f"{LEFT}\n\n{RIGHT}\n\n{NOTE}"

MATH = """Load table

Item
Weight kg
Count
Bolts
2.5
40
Oil
12.0
3

w = 2.5 * 40 + 12.0 * 3 = 136.0 kg"""


def font(path, size):
    return ImageFont.truetype(path, size)


def wrap(draw, text, face, width):
    lines = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue
        current = ""
        for word in paragraph.split():
            trial = word if not current else f"{current} {word}"
            if not current or draw.textlength(trial, font=face) <= width:
                current = trial
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
    return lines


def draw_wrapped(path, text, face, size, width=1000, margin=72):
    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    lines = wrap(probe, text, face, width - margin * 2)
    gap = int(size * 0.55)
    line_h = size + gap
    image = Image.new("RGB", (width, margin * 2 + line_h * len(lines)), "white")
    draw = ImageDraw.Draw(image)
    y = margin
    for line in lines:
        draw.text((margin, y), line, font=face, fill=(20, 20, 20))
        y += line_h
    image.save(path)
    return image


def write_letter():
    draw_wrapped(PAGES / "letter.png", LETTER, font(SANS, 32), 32)
    (PAGES / "letter.txt").write_text(LETTER + "\n", encoding="utf-8")


def write_columns():
    width, height = 1100, 1500
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    body = font(SANS, 28)
    left_lines = wrap(draw, LEFT, body, 460)
    right_lines = wrap(draw, RIGHT, body, 460)
    y = 70
    for line in left_lines:
        draw.text((50, y), line, font=body, fill=(20, 20, 20))
        y += 44
    y = 70
    for line in right_lines:
        draw.text((590, y), line, font=body, fill=(20, 20, 20))
        y += 44
    draw.line((50, 1180, 1050, 1180), fill=(20, 20, 20), width=2)
    note_lines = wrap(draw, NOTE, body, 1000)
    y = 1210
    for line in note_lines:
        draw.text((50, y), line, font=body, fill=(20, 20, 20))
        y += 44
    image.save(PAGES / "columns.png")
    (PAGES / "columns.txt").write_text(COLUMNS + "\n", encoding="utf-8")
    return image


def write_math():
    title = font(SANS_BOLD, 36)
    cell = font(SANS, 28)
    eq = font(MONO, 28)
    headers = ["Item", "Weight kg", "Count"]
    rows = [["Bolts", "2.5", "40"], ["Oil", "12.0", "3"]]
    image = Image.new("RGB", (1100, 900), "white")
    draw = ImageDraw.Draw(image)
    draw.text((48, 40), "Load table", font=title, fill=(20, 20, 20))
    col_w, row_h = 280, 64
    top, left = 120, 48
    for r, row in enumerate([headers, *rows]):
        y = top + r * row_h
        for c, value in enumerate(row):
            x = left + c * col_w
            draw.rectangle((x, y, x + col_w, y + row_h), outline=(20, 20, 20), width=2)
            draw.text((x + 16, y + 16), value, font=cell, fill=(20, 20, 20))
    draw.text((48, 420), "w = 2.5 * 40 + 12.0 * 3 = 136.0 kg", font=eq, fill=(20, 20, 20))
    image.save(PAGES / "math.png")
    (PAGES / "math.txt").write_text(MATH + "\n", encoding="utf-8")
    return image


def write_receipt_image():
    face = font(MONO, 28)
    lines = RECEIPT.split("\n")
    width, height = 780, 760
    image = Image.new("RGB", (width, height), (252, 248, 240))
    draw = ImageDraw.Draw(image)
    draw.rectangle((18, 18, width - 18, height - 18), outline=(40, 40, 40), width=2)
    y = 48
    for line in lines:
        draw.text((48, y), line, font=face, fill=(20, 20, 20))
        y += 42
    return image


def write_skewed(receipt):
    tilted = receipt.rotate(6, expand=True, fillcolor=(214, 210, 200))
    faded = ImageEnhance.Contrast(tilted).enhance(0.82)
    faded = faded.filter(ImageFilter.GaussianBlur(0.5))
    faded.save(PAGES / "skewed-receipt.jpg", quality=68)
    (PAGES / "skewed-receipt.txt").write_text(RECEIPT + "\n", encoding="utf-8")


def main():
    PAGES.mkdir(parents=True, exist_ok=True)
    write_letter()
    columns = write_columns()
    math = write_math()
    write_skewed(write_receipt_image())
    columns.save(ROOT / "paper.pdf", "PDF", resolution=150, save_all=True, append_images=[math])
    for stale in ("clean-letter.png", "clean-letter.txt", "receipt.png", "receipt.txt", "table.png", "table.txt"):
        path = PAGES / stale
        if path.exists():
            path.unlink()
    print(f"Wrote {PAGES} and {ROOT / 'paper.pdf'}")


if __name__ == "__main__":
    main()
