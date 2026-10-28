r"""Export the local NLI model to ONNX Runtime format.

Examples:
    .\.venv\Scripts\python.exe scripts\export_nli_onnx.py
    .\.venv\Scripts\python.exe scripts\export_nli_onnx.py --quantize-int8
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from evaluation.onnx_export import main  # noqa: E402

if __name__ == "__main__":
    main()
