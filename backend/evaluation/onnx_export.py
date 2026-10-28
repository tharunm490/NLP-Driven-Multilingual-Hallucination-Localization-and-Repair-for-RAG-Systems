"""Export the local multilingual NLI model to ONNX, with optional INT8 weights.

This is a local, offline conversion step. It does not call a hosted inference
service. The exported file is intentionally kept outside source control because
the mDeBERTa checkpoint is large.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
DEFAULT_OUTPUT = Path("data/onnx/mdeberta-v3-base-mnli-xnli")


def export_nli_onnx(
    model_name: str = DEFAULT_MODEL,
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    opset_version: int = 17,
    quantize_int8: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "torch and transformers are required to export the NLI model."
        ) from exc

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = target_dir / "model.onnx"
    manifest_path = target_dir / "manifest.json"
    if onnx_path.exists() and not force:
        manifest = _read_manifest(manifest_path)
        if manifest:
            quantized_path = target_dir / "model_int8.onnx"
            if quantize_int8 and not quantized_path.exists():
                _quantize(onnx_path, quantized_path)
            if quantized_path.exists():
                manifest["int8_exported"] = True
                manifest["int8_bytes"] = quantized_path.stat().st_size
                manifest_path.write_text(
                    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
                )
            return manifest

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, dtype=torch.float32
    )
    model.eval()
    try:
        model.config._attn_implementation = "eager"
    except Exception:
        pass

    encoded = tokenizer(
        ["Information security preserves confidentiality and integrity."],
        ["The ISMS preserves confidentiality and integrity."],
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    if "token_type_ids" not in encoded:
        encoded["token_type_ids"] = torch.zeros_like(encoded["input_ids"])

    class _LogitsWrapper(torch.nn.Module):
        def __init__(self, inner: Any) -> None:
            super().__init__()
            self.inner = inner

        def forward(
            self,
            input_ids: Any,
            attention_mask: Any,
            token_type_ids: Any,
        ) -> Any:
            return self.inner(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            ).logits

    wrapper = _LogitsWrapper(model)
    wrapper.eval()
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (
                encoded["input_ids"],
                encoded["attention_mask"],
                encoded["token_type_ids"],
            ),
            str(onnx_path),
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "token_type_ids": {0: "batch", 1: "sequence"},
                "logits": {0: "batch"},
            },
            opset_version=opset_version,
            do_constant_folding=True,
            dynamo=False,
        )

    manifest = {
        "model_name": model_name,
        "format": "onnx",
        "opset_version": opset_version,
        "id2label": {
            str(index): str(label)
            for index, label in (model.config.id2label or {}).items()
        },
        "max_position_embeddings": getattr(
            model.config, "max_position_embeddings", None
        ),
        "tokenizer_max_length": getattr(tokenizer, "model_max_length", None),
        "export_seconds": round(time.perf_counter() - started, 3),
        "onnx_bytes": onnx_path.stat().st_size,
        "int8_exported": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    if quantize_int8:
        quantized_path = target_dir / "model_int8.onnx"
        _quantize(onnx_path, quantized_path)
        manifest["int8_exported"] = True
        manifest["int8_bytes"] = quantized_path.stat().st_size
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    return manifest


def _quantize(source: Path, target: Path) -> None:
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "onnxruntime quantization tools are required for INT8 export."
        ) from exc
    quantize_dynamic(
        str(source),
        str(target),
        weight_type=QuantType.QInt8,
    )


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--quantize-int8", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = export_nli_onnx(
        model_name=args.model_name,
        output_dir=args.output_dir,
        opset_version=args.opset,
        quantize_int8=args.quantize_int8,
        force=args.force,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
