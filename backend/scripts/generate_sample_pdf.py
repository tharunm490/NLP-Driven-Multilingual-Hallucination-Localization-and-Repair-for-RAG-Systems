"""Generate a small, original sample PDF for manual Phase 2 testing.

Usage (from the backend directory, with the virtual environment active):

    python scripts/generate_sample_pdf.py

The PDF is written next to the TXT sample in backend/samples/.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "samples"
OUTPUT_PATH = SAMPLE_DIR / "sample_english.pdf"

PAGES = [
    "Sample Document: Geography and Science\n\n"
    "The Eiffel Tower is located in Paris, France. It was completed in 1889 "
    "and is 330 metres tall. The tower was designed by the engineering company "
    "of Gustave Eiffel.",
    "Sample Document: Cities\n\n"
    "Kyoto was the capital of Japan for more than one thousand years. Today it "
    "is known for its temples and traditional gardens.\n\n"
    "New Delhi is the capital of India and lies beside the Yamuna river.",
    "Sample Document: Measurements\n\n"
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.\n"
    "The Great Wall of China is more than 21,000 kilometres long.\n"
    "The Indian Constitution came into effect on 26 January 1950.",
]


def main() -> None:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open()
    for content in PAGES:
        page = document.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 545, 780), content, fontsize=12)
    document.save(str(OUTPUT_PATH))
    document.close()
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
