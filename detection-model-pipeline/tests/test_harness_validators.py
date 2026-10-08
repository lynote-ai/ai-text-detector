"""Harness 校验器测试（轻量：stdlib + PyYAML，CI 可跑）。"""

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from harness.validate_demo_data import (  # noqa: E402
    validate_demo_release,
    validate_jsonl,
)
from harness.validate_docs import (  # noqa: E402
    validate_documentation,
    validate_english_text,
    validate_mermaid_blocks,
)
from harness.validate_manifest import validate_manifest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = REPO_ROOT / "data" / "demos"
DEMO_MANIFEST = DEMO_ROOT / "manifest.yaml"

MANIFEST_BASE = {
    "schema_version": 1,
    "version": {
        "id": "open_source",
        "status": "prepared",
        "created_on": "2026-09-29",
        "objective": "Ship a two-level AI-text detector with bounded public demos.",
    },
    "planned_training": {"stages": [{"name": "prepare-dataset", "objective": "x", "status": "implemented"}]},
    "inputs": {
        "data_revisions": [],
        "input_sources": [{
            "id": "detection-log-export",
            "status": "external-not-distributed",
            "path": "supplied-by-user",
            "format": "detection-service-api-log-csv",
            "file_pattern": "*.csv",
            "intended_use": "two-level-soft-label-distillation",
        }],
    },
    "build_parameters": {"random_seed": 42},
    "artifacts": {"processed_data": [], "checkpoints": [], "evaluations": []},
    "notes": [],
}


def _write_manifest(tmp_path: Path, overrides: dict | None = None) -> Path:
    manifest = json.loads(json.dumps(MANIFEST_BASE))
    if overrides:
        for key, value in overrides.items():
            manifest[key] = value
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False),
                    encoding="utf-8")
    return path


def _demo_record(scan_id: str = "a" * 32, doc_class: str = "human",
                 source: str = "synthetic") -> dict:
    return {
        "source": source,
        "scan_id": scan_id,
        "text": "First sentence. Second sentence.",
        "doc_label": [0.90, 0.06, 0.04],
        "sentences": [
            {"text": "First sentence.", "y_origin": [0.93, 0.04, 0.03],
             "y_human_ai": 0.93},
            {"text": "Second sentence.", "y_origin": [0.90, 0.06, 0.04],
             "y_human_ai": 0.90},
        ],
        "gptzero_doc_class": doc_class,
        "gptzero_language": "en",
    }


class TestValidateManifest:
    def test_project_manifest_passes(self) -> None:
        version_id = validate_manifest(REPO_ROOT / "manifest.yaml")
        assert version_id == "detection-model-pipeline"

    def test_wrong_schema_version_is_rejected(self, tmp_path: Path) -> None:
        path = _write_manifest(tmp_path, overrides={"schema_version": 2})
        with pytest.raises(ValueError, match="schema_version"):
            validate_manifest(path)

    def test_id_must_match_directory_name(self, tmp_path: Path) -> None:
        path = _write_manifest(tmp_path)
        with pytest.raises(ValueError, match="directory"):
            validate_manifest(path)

    def test_input_source_requires_all_fields(self, tmp_path: Path) -> None:
        version_dir = tmp_path / "open_source"
        version_dir.mkdir()
        path = version_dir / "manifest.yaml"
        manifest = json.loads(json.dumps(MANIFEST_BASE))
        del manifest["inputs"]["input_sources"][0]["file_pattern"]
        path.write_text(yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False),
                        encoding="utf-8")
        with pytest.raises(ValueError, match="file_pattern"):
            validate_manifest(path)


class TestValidateDemoData:
    def test_current_demos_pass(self) -> None:
        counts = validate_demo_release(DEMO_ROOT, DEMO_MANIFEST)
        assert counts == {"01_prepare": 25, "02_calibrate": 1, "03_serve": 1}

    def test_real_source_label_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "demo.jsonl"
        path.write_text(json.dumps(_demo_record(source="crawl")) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="synthetic"):
            validate_jsonl(path, 20, want_val_split=False)

    def test_scan_id_must_match_split_rule(self, tmp_path: Path) -> None:
        path = tmp_path / "demo.jsonl"
        path.write_text(json.dumps(_demo_record(scan_id="f" * 32)) + "\n",
                        encoding="utf-8")
        with pytest.raises(ValueError, match="split rule"):
            validate_jsonl(path, 20, want_val_split=True)

    def test_sentence_must_be_substring_of_text(self, tmp_path: Path) -> None:
        record = _demo_record()
        record["sentences"][0]["text"] = "Not in the text."
        path = tmp_path / "demo.jsonl"
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="substring"):
            validate_jsonl(path, 20, want_val_split=False)

    def test_doc_label_must_sum_to_one(self, tmp_path: Path) -> None:
        record = _demo_record()
        record["doc_label"] = [0.5, 0.1, 0.1]
        path = tmp_path / "demo.jsonl"
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="sum to 1"):
            validate_jsonl(path, 20, want_val_split=False)

    def test_over_limit_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "demo.jsonl"
        path.write_text(json.dumps(_demo_record()) + "\n" + json.dumps(_demo_record()) + "\n",
                        encoding="utf-8")
        with pytest.raises(ValueError, match="exceeds"):
            validate_jsonl(path, 1, want_val_split=False)


class TestValidateDocs:
    def test_current_docs_pass(self) -> None:
        counts = validate_documentation(REPO_ROOT)
        assert counts["README.md"] >= 1
        assert counts["docs/ARCHITECTURE.md"] >= 2
        assert counts["docs/DATA_CONSTRUCTION.md"] >= 3
        assert counts["docs/MODEL.md"] >= 2
        assert counts["docs/API.md"] >= 1

    def test_chinese_in_english_document_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "README.md"
        path.write_text("# Title\n\n中文混入。\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Chinese text"):
            validate_english_text(path, path.read_text(encoding="utf-8"))

    def test_missing_relative_link_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "README.md"
        path.write_text("# Title\n\n[gone](docs/missing.md)\n", encoding="utf-8")
        from harness.validate_docs import validate_relative_links  # noqa: E402
        with pytest.raises(ValueError, match="broken relative link"):
            validate_relative_links(path, path.read_text(encoding="utf-8"))

    def test_mermaid_minimum_is_enforced(self, tmp_path: Path) -> None:
        path = tmp_path / "README.md"
        path.write_text("# Title\n\nno diagram here\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Mermaid blocks"):
            validate_mermaid_blocks(path, path.read_text(encoding="utf-8"), 1)
