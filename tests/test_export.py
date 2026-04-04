"""Tests for src/models/export.py."""

import json
from pathlib import Path

import pytest

from src.models.export import (
    ExportConfig,
    _parse_export_config,
    export_to_onnx,
    load_model_metadata,
)


class TestExportConfig:
    """Tests for ExportConfig defaults and parsing."""

    def test_defaults(self) -> None:
        cfg = ExportConfig()
        assert cfg.simplify is True
        assert cfg.half is False
        assert cfg.int8 is False
        assert cfg.opset == 17
        assert cfg.image_size == 640

    def test_parse_from_dict(self) -> None:
        config = {"export": {"simplify": False, "half": True, "opset": 13}}
        cfg = _parse_export_config(config)
        assert cfg.simplify is False
        assert cfg.half is True
        assert cfg.opset == 13

    def test_parse_none(self) -> None:
        cfg = _parse_export_config(None)
        assert cfg == ExportConfig()


class TestLoadModelMetadata:
    """Tests for loading class names from sidecar JSON."""

    def test_load_valid(self, tmp_path: Path) -> None:
        # Create a fake metadata file
        onnx_path = tmp_path / "model.onnx"
        onnx_path.touch()
        meta_path = tmp_path / "model_meta.json"
        meta_path.write_text(
            json.dumps(
                {
                    "class_names": {"0": "person", "1": "car"},
                    "num_classes": 2,
                    "source_weights": "model",
                }
            )
        )

        names = load_model_metadata(onnx_path)
        assert names == {0: "person", 1: "car"}

    def test_missing_file(self, tmp_path: Path) -> None:
        onnx_path = tmp_path / "model.onnx"
        with pytest.raises(FileNotFoundError, match="metadata"):
            load_model_metadata(onnx_path)


@pytest.mark.slow
class TestExportToOnnx:
    """Integration test for the full export pipeline."""

    def test_export_produces_onnx(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "export"
        onnx_path = export_to_onnx(
            weights_path="yolov8n.pt",
            output_dir=output_dir,
        )
        assert onnx_path.exists()
        assert onnx_path.suffix == ".onnx"

        # Check metadata sidecar was created
        meta_path = onnx_path.with_name(onnx_path.stem + "_meta.json")
        assert meta_path.exists()

        names = load_model_metadata(onnx_path)
        assert len(names) == 80  # COCO classes
        assert names[0] == "person"
