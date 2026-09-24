"""OvisOCR2 runner. 0.8B end-to-end page parser."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.vlm_runner import main

MODEL_ID = "ATH-MaaS/OvisOCR2"
PROMPT = (
    "Extract all readable content from the image in natural human reading order "
    "and output the result as a single Markdown document. For charts or images, "
    'represent them using an HTML image tag: <img src="images/bbox_{left}_{top}_{right}_{bottom}.jpg" />, '
    "where left, top, right, bottom are bounding box coordinates scaled to [0, 1000). "
    "Format formulas as LaTeX. Format tables as HTML: <table>...</table>. "
    "Transcribe all other text as standard Markdown. Preserve the original text "
    "without translation or paraphrasing."
)

if __name__ == "__main__":
    main(
        MODEL_ID,
        PROMPT,
        template_kwargs={"enable_thinking": False},
    )
