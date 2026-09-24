"""GLM-OCR runner. 0.9B model with the official text-recognition prompt.

The published page-level scores use Z.ai's SDK, which adds a layout model
in front of this checkpoint. This runner calls the checkpoint directly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.vlm_runner import main

if __name__ == "__main__":
    main("zai-org/GLM-OCR", "Text Recognition:")
