# Corpus

Six pages. The two hard paper pages are also `paper.pdf`.

| Page | File | What a miss looks like |
| --- | --- | --- |
| Clean letter | `pages/letter.png` | Easy control. A model that fails here is not usable. |
| Two columns and a footnote | `pages/columns.png` | Reading across the columns, or dropping the note under the line. |
| Table and an equation | `pages/math.png` | A wrong cell, or the total `136.0` missing. |
| Tilted receipt | `pages/skewed-receipt.jpg` | Prices and the total on a faded, rotated photo of a receipt. |
| Invoice table | `pages/table.png` | A real NF-e tax and product block. A miss is the note total `3.254,07`, the product code, or a retained-tax amount. |
| Package photo | `../images/chocolate.jpeg` | A real phone photo. Scored on the nutrition lines you can read, not the tiny print. |

`pages/*.txt` is the transcript. `manifest.json` lists the fields that must show up in the output.
Regenerate the drawn pages with `python corpus/make_pages.py`.
