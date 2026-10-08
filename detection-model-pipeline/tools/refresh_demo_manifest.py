"""Regenerate data/demos/manifest.yaml with pinned SHA-256 digests.

Maintainer-only: run after editing any file under data/demos/. The generated
YAML is emitted deterministically (fixed key order, single-quoted scalars)
so that make demo-validate stays diff-clean, and it is re-parsed for real by
PyYAML inside make demo-validate, which closes the loop on this hand-rolled
emitter.
"""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

RECORD_LIMIT_PER_STAGE = 20
SELECTION_POLICY = "synthetic-records-authored-for-publication"
DEMO_STAGES = [
    (
        "01_prepare",
        "Deduplicated two-level soft-label records with the deterministic train/validation split",
        [
            ("data/demos/01_prepare/train_hier_demo.jsonl", "jsonl"),
            ("data/demos/01_prepare/val_hier_demo.jsonl", "jsonl"),
        ],
    ),
    (
        "02_calibrate",
        "Document-head temperature calibration artifact",
        [
            ("data/demos/02_calibrate/calibration_demo.json", "json"),
        ],
    ),
    (
        "03_serve",
        "Full HTTP envelope for one detect request",
        [
            ("data/demos/03_serve/detect_response_demo.json", "json"),
        ],
    ),
]


def parse_args() -> argparse.Namespace:
    """Parse the demo root and manifest paths."""
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-root", type=Path,
                        default=project_root / "data" / "demos")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="defaults to <demo-root>/manifest.yaml")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for one demo file."""
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_records(path: Path, kind: str) -> int:
    """Count JSONL lines or JSON artifacts."""
    if kind == "json":
        with path.open("r", encoding="utf-8") as input_file:
            json.load(input_file)
        return 1
    count = 0
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                raise ValueError(f"blank line at {path}")
            json.loads(line)
            count += 1
            if count > RECORD_LIMIT_PER_STAGE:
                raise ValueError(
                    f"{path} exceeds the {RECORD_LIMIT_PER_STAGE}-record public limit"
                )
    return count


def yaml_scalar(value: object) -> str:
    """Emit one scalar with single quotes, doubling embedded quotes."""
    return "'" + str(value).replace("'", "''") + "'"


def render_manifest(stages: list[tuple[str, str, list[tuple[str, int, str]]]]) -> str:
    """Render the demo manifest deterministically."""
    lines = [
        "schema_version: 1",
        f"record_limit_per_stage: {RECORD_LIMIT_PER_STAGE}",
        f"selection_policy: {yaml_scalar(SELECTION_POLICY)}",
        "stages:",
    ]
    for stage, description, files in stages:
        lines.append(f"- stage: {yaml_scalar(stage)}")
        lines.append(f"  description: {yaml_scalar(description)}")
        lines.append("  files:")
        for path, count, digest in files:
            lines.append(f"  - path: {yaml_scalar(path)}")
            lines.append(f"    record_count: {count}")
            lines.append(f"    sha256: {digest}")
    return "\n".join(lines) + "\n"


def main() -> int:
    """Rebuild the demo manifest and report status."""
    args = parse_args()
    demo_root = args.demo_root.resolve()
    manifest_path = (args.manifest or demo_root / "manifest.yaml").resolve()
    print("target: demo-refresh")
    print(f"demo_root: {demo_root}")
    print(f"manifest: {manifest_path}")
    print(f"started_at: {datetime.now().astimezone().isoformat()}")
    try:
        registered_files: list[Path] = []
        stages = []
        for stage, description, file_specs in DEMO_STAGES:
            files = []
            for relative_path, kind in file_specs:
                path = demo_root / Path(relative_path).relative_to("data/demos")
                if not path.is_file():
                    raise ValueError(f"registered demo file is missing: {path}")
                count = count_records(path, kind)
                if kind == "jsonl" and count == 0:
                    raise ValueError(f"JSONL demo must not be empty: {path}")
                files.append((relative_path, count, sha256_file(path)))
                registered_files.append(path.resolve())
            stages.append((stage, description, files))
        discovered = {path.resolve() for path in demo_root.rglob("*") if path.is_file()}
        # The manifest itself may not exist yet on a first run; it is written below.
        discovered.discard(manifest_path)
        expected = set(registered_files)
        if discovered != expected:
            extra = sorted(str(p) for p in discovered - expected)
            missing = sorted(str(p) for p in expected - discovered)
            raise ValueError(f"unexpected demo file set; extra={extra}, missing={missing}")
        manifest_path.write_text(render_manifest(stages), encoding="utf-8")
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print("status: failed")
        print(f"error: {error}")
        return 2
    print(f"demo_stage_count: {len(stages)}")
    print("status: success")
    print(f"finished_at: {datetime.now().astimezone().isoformat()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
