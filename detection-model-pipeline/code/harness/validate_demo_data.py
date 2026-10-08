"""Validate bounded synthetic demo data and their release manifest.

Every record under data/demos must carry ``source: synthetic``. This is the
machine-checkable guarantee that real user text can never be committed here:
the production intake does not write a ``source`` field at all.
"""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import yaml


DEMO_RECORD_LIMIT = 20
DEMO_SOURCE_LABEL = "synthetic"
SPLIT_RULE_MOD = 10
SPLIT_RULE_HOLDOUT = 1

EXPECTED_STAGES = {
    "01_prepare": ("train_hier_demo.jsonl", "val_hier_demo.jsonl"),
    "02_calibrate": ("calibration_demo.json",),
    "03_serve": ("detect_response_demo.json",),
}
JSONL_FILES = {"train_hier_demo.jsonl", "val_hier_demo.jsonl"}
PINNED_COUNTS = {"train_hier_demo.jsonl": 20, "val_hier_demo.jsonl": 5}

RECORD_KEYS = {
    "source", "scan_id", "text", "doc_label", "sentences",
    "gptzero_doc_class", "gptzero_language",
}
SENTENCE_KEYS = {"text", "y_origin", "y_human_ai"}
DOC_CLASSES = ["human", "ai", "mixed"]
SENT_CLASSES = ["human", "ai", "paraphrased"]
DOC_CLASS_ENUM = {"human": "HUMAN_ONLY", "ai": "AI_ONLY", "mixed": "MIXED"}

DOCUMENT_KEYS = {
    "paragraphs", "sentences", "classProbabilities", "confidenceThresholdsRaw",
    "confidenceScoresRaw", "subclass", "pageNumber", "language", "inputText",
    "documentId", "predictedClass", "confidenceScore", "confidenceCategory",
    "documentClassification", "resultMessage", "completelyGeneratedProb",
    "averageGeneratedProb", "overallBurstiness", "writingStats", "version",
    "neatVersion",
}
THRESHOLD_LEVELS = {"reject": 0.33, "low": 0.6, "medium": 0.8}
RESULT_MESSAGES = {
    ("human", "high"): "High confidence that the text was written entirely by a human.",
    ("human", "medium"): "Moderate confidence that the text was written entirely by a human.",
    ("ai", "high"): "High confidence that the text was written by AI.",
    ("ai", "medium"): "Moderate confidence that the text was written by AI.",
    ("mixed", "high"): "High confidence that the text mixes human-written and AI-written parts.",
    ("mixed", "medium"): "Moderate confidence that the text mixes human-written and AI-written parts.",
}
LOW_MESSAGE = (
    "Low confidence: the signal is not strong enough to call this text "
    "human-written or AI-written."
)


def parse_args() -> argparse.Namespace:
    """Parse the demo root and manifest supplied by the Make target."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for one demo file."""
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_val(scan_id: str) -> bool:
    """Mirror code/i_prepare_hier_dataset.is_val: md5(scan_id)[:8] % 10 == 1."""
    digest = hashlib.md5(scan_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % SPLIT_RULE_MOD == SPLIT_RULE_HOLDOUT


def _require_probs(values: list[float], name: str, path: Path, line: int) -> None:
    """Require a length-3 probability vector that sums to one."""
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError(f"{name} must be a length-3 list at {path}:{line}")
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        raise ValueError(f"{name} must be numeric at {path}:{line}")
    if abs(sum(values) - 1.0) > 1e-6:
        raise ValueError(f"{name} must sum to 1 at {path}:{line}")


def validate_jsonl(path: Path, record_limit: int, want_val_split: bool) -> int:
    """Validate one JSONL demo against the production schema plus provenance."""
    count = 0
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                raise ValueError(f"blank line at {path}:{line_number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"record must be a mapping at {path}:{line_number}")
            here = f"{path}:{line_number}"

            if record.get("source") != DEMO_SOURCE_LABEL:
                raise ValueError(
                    f"record source must be {DEMO_SOURCE_LABEL!r} at {here}"
                )
            if not RECORD_KEYS <= set(record):
                missing = sorted(RECORD_KEYS - set(record))
                raise ValueError(f"missing keys {missing} at {here}")
            scan_id = record["scan_id"]
            if not isinstance(scan_id, str) or not len(scan_id) == 32 or \
                    not all(c in "0123456789abcdef" for c in scan_id):
                raise ValueError(f"scan_id must be 32 lowercase hex chars at {here}")
            if is_val(scan_id) != want_val_split:
                split_name = "val" if want_val_split else "train"
                raise ValueError(
                    f"scan_id does not satisfy the {split_name} split rule at {here}"
                )
            if not isinstance(record["text"], str) or not record["text"].strip():
                raise ValueError(f"text must be a non-empty string at {here}")

            _require_probs(record["doc_label"], "doc_label", path, line_number)
            if record["gptzero_doc_class"] not in DOC_CLASSES:
                raise ValueError(f"unknown gptzero_doc_class at {here}")
            if record["gptzero_doc_class"] != DOC_CLASSES[
                record["doc_label"].index(max(record["doc_label"]))
            ]:
                raise ValueError(f"gptzero_doc_class must match doc_label argmax at {here}")
            if not isinstance(record["gptzero_language"], str) or \
                    not record["gptzero_language"].strip():
                raise ValueError(f"gptzero_language must be a non-empty string at {here}")

            sentences = record["sentences"]
            if not isinstance(sentences, list) or not sentences:
                raise ValueError(f"sentences must be a non-empty list at {here}")
            if len(sentences) > 256:
                raise ValueError(f"too many sentences at {here}")
            for s_index, sentence in enumerate(sentences):
                s_here = f"{here} sentence[{s_index}]"
                if not isinstance(sentence, dict) or not SENTENCE_KEYS <= set(sentence):
                    raise ValueError(f"sentence keys missing at {s_here}")
                if not isinstance(sentence["text"], str) or \
                        sentence["text"] not in record["text"]:
                    raise ValueError(f"sentence text must be a substring of text at {s_here}")
                _require_probs(sentence["y_origin"], "y_origin", path, line_number)
                y_human_ai = sentence["y_human_ai"]
                if not isinstance(y_human_ai, (int, float)) or \
                        abs(y_human_ai - sentence["y_origin"][0]) > 1e-6:
                    raise ValueError(f"y_human_ai must equal y_origin[0] at {s_here}")
            count += 1
            if count > record_limit:
                raise ValueError(f"{path} exceeds the {record_limit}-record public limit")
    return count


def validate_calibration(path: Path) -> int:
    """Validate the calibration demo against k_calibrate_hier_model output."""
    with path.open("r", encoding="utf-8") as input_file:
        record = json.load(input_file)
    if not isinstance(record, dict):
        raise ValueError(f"{path} must contain one JSON mapping")
    temperature = record.get("temperature")
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) \
            or not temperature > 0:
        raise ValueError(f"{path} temperature must be a positive number")
    if record.get("doc_class_names") != DOC_CLASSES:
        raise ValueError(f"{path} doc_class_names must equal {DOC_CLASSES}")
    if not isinstance(record.get("fitted_samples"), int) or \
            isinstance(record.get("fitted_samples"), bool) or \
            record.get("fitted_samples") <= 0:
        raise ValueError(f"{path} fitted_samples must be a positive integer")
    return 1


def _validate_result_message(message: str, path: Path, here: str) -> None:
    """Require one of the service's own neutral result messages."""
    allowed = list(RESULT_MESSAGES.values()) + [LOW_MESSAGE]
    if not any(message == text or message.startswith(text + " (Input truncated")
               for text in allowed):
        raise ValueError(f"unknown resultMessage at {here}")


def validate_response(path: Path) -> int:
    """Validate the detect-response demo envelope for internal consistency."""
    with path.open("r", encoding="utf-8") as input_file:
        envelope = json.load(input_file)
    if not isinstance(envelope, dict):
        raise ValueError(f"{path} must contain one JSON mapping")
    here = str(path)
    if envelope.get("code") != 0 or envelope.get("msg") != "success":
        raise ValueError(f"envelope code/msg mismatch at {here}")
    data = envelope.get("data")
    if not isinstance(data, dict) or data.get("version") != "V2-hier-1" or \
            data.get("neatVersion") != "V2h":
        raise ValueError(f"data version mismatch at {here}")
    scan_id = data.get("scanId", "")
    if not isinstance(scan_id, str) or len(scan_id) != 32 or \
            not all(c in "0123456789abcdef" for c in scan_id):
        raise ValueError(f"scanId must be 32 lowercase hex chars at {here}")
    documents = data.get("documents")
    if not isinstance(documents, list) or len(documents) != 1:
        raise ValueError(f"documents must contain exactly one document at {here}")
    document = documents[0]
    if not isinstance(document, dict) or not DOCUMENT_KEYS <= set(document):
        missing = sorted(DOCUMENT_KEYS - set(document))
        raise ValueError(f"document missing keys {missing} at {here}")

    class_probs = document["classProbabilities"]
    if set(class_probs) != set(DOC_CLASSES) or abs(sum(class_probs.values()) - 1.0) > 1e-6:
        raise ValueError(f"classProbabilities must sum to 1 over {DOC_CLASSES} at {here}")
    predicted = document["predictedClass"]
    if predicted != max(class_probs, key=class_probs.get):
        raise ValueError(f"predictedClass must be the classProbabilities argmax at {here}")
    if abs(document["confidenceScore"] - max(class_probs.values())) > 1e-6:
        raise ValueError(f"confidenceScore must equal max(classProbabilities) at {here}")
    score = document["confidenceScore"]
    expected_category = (
        "high" if score >= 0.8 else
        "medium" if score >= 0.6 else
        "low" if score >= 0.33 else "reject"
    )
    if document["confidenceCategory"] != expected_category:
        raise ValueError(f"confidenceCategory must match the 0.33/0.6/0.8 bands at {here}")
    if document["documentClassification"] != DOC_CLASS_ENUM[predicted]:
        raise ValueError(f"documentClassification mismatch at {here}")
    if abs(document["completelyGeneratedProb"] - class_probs["ai"]) > 1e-6:
        raise ValueError(f"completelyGeneratedProb must equal P(ai) at {here}")
    if document["overallBurstiness"] != 0 or document["writingStats"] != {}:
        raise ValueError(f"overallBurstiness/writingStats must keep reserved values at {here}")
    if document["language"] != "en":
        raise ValueError(f"demo response language must be en at {here}")
    if document["version"] != "V2-hier-1" or document["neatVersion"] != "V2h":
        raise ValueError(f"document version fields mismatch at {here}")
    _validate_result_message(document["resultMessage"], path, here)

    expected_thresholds = {"identity": {c: dict(THRESHOLD_LEVELS) for c in DOC_CLASSES}}
    if document["confidenceThresholdsRaw"] != expected_thresholds:
        raise ValueError(f"confidenceThresholdsRaw mismatch at {here}")
    if document["confidenceScoresRaw"] != {"identity": dict(class_probs)}:
        raise ValueError(f"confidenceScoresRaw mismatch at {here}")

    sentences = document["sentences"]
    if not isinstance(sentences, list) or not sentences:
        raise ValueError(f"sentences must be a non-empty list at {here}")
    generated = []
    for s_index, sentence in enumerate(sentences):
        s_here = f"{here} sentence[{s_index}]"
        if not isinstance(sentence, dict):
            raise ValueError(f"sentence must be a mapping at {s_here}")
        s_class_probs = sentence.get("classProbabilities")
        if not isinstance(s_class_probs, dict) or set(s_class_probs) != set(SENT_CLASSES) \
                or abs(sum(s_class_probs.values()) - 1.0) > 1e-6:
            raise ValueError(f"sentence classProbabilities invalid at {s_here}")
        expected_generated = s_class_probs["ai"] + s_class_probs["paraphrased"]
        if abs(sentence["generatedProb"] - expected_generated) > 1e-6:
            raise ValueError(f"generatedProb must equal P(ai)+P(paraphrased) at {s_here}")
        if sentence["perplexity"] != 0:
            raise ValueError(f"perplexity must stay 0 at {s_here}")
        if sentence["highlightSentenceForAi"] != (sentence["generatedProb"] >= 0.5):
            raise ValueError(f"highlightSentenceForAi mismatch at {s_here}")
        argmax_class = max(s_class_probs, key=s_class_probs.get)
        expected_special = "polished" if argmax_class == "paraphrased" else None
        if sentence["specialHighlightType"] != expected_special:
            raise ValueError(f"specialHighlightType mismatch at {s_here}")
        if sentence["sentence"] not in document["inputText"]:
            raise ValueError(f"sentence text must be a substring of inputText at {s_here}")
        generated.append(sentence["generatedProb"])

    if abs(document["averageGeneratedProb"] - sum(generated) / len(generated)) > 1e-6:
        raise ValueError(f"averageGeneratedProb mismatch at {here}")
    paragraphs = document["paragraphs"]
    if not isinstance(paragraphs, list):
        raise ValueError(f"paragraphs must be a list at {here}")
    cursor = 0
    for p_index, paragraph in enumerate(paragraphs):
        p_here = f"{here} paragraph[{p_index}]"
        if paragraph["startSentenceIndex"] != cursor:
            raise ValueError(f"paragraph startSentenceIndex mismatch at {p_here}")
        num = paragraph["numSentences"]
        if num <= 0 or cursor + num > len(generated):
            raise ValueError(f"paragraph numSentences out of range at {p_here}")
        segment = generated[cursor:cursor + num]
        if abs(paragraph["completelyGeneratedProb"] - sum(segment) / len(segment)) > 1e-6:
            raise ValueError(f"paragraph completelyGeneratedProb mismatch at {p_here}")
        cursor += num
    if cursor != len(generated):
        raise ValueError(f"paragraphs must cover every sentence at {here}")
    return 1


def _validate_file(path: Path, filename: str, entry: dict, here: str) -> int:
    """Validate one registered demo file and return its record count."""
    if filename in JSONL_FILES:
        want_val_split = filename == "val_hier_demo.jsonl"
        count = validate_jsonl(path, DEMO_RECORD_LIMIT, want_val_split)
        pinned = PINNED_COUNTS.get(filename)
        if pinned is not None and count != pinned:
            raise ValueError(f"{here} has {count} records; expected {pinned}")
    elif filename == "calibration_demo.json":
        count = validate_calibration(path)
    elif filename == "detect_response_demo.json":
        count = validate_response(path)
    else:  # pragma: no cover - EXPECTED_STAGES is the only source of filenames
        raise ValueError(f"unregistered demo file kind: {filename}")
    if entry.get("record_count") != count:
        raise ValueError(f"manifest record count mismatch at {here}")
    if entry.get("sha256") != sha256_file(path):
        raise ValueError(f"manifest SHA-256 mismatch at {here}")
    return count


def validate_demo_release(demo_root: Path, manifest_path: Path) -> dict[str, int]:
    """Validate stage coverage, records, hashes, and the no-extra-files rule."""
    with manifest_path.open("r", encoding="utf-8") as input_file:
        manifest = yaml.safe_load(input_file)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("demo manifest schema_version must equal 1")
    record_limit = manifest.get("record_limit_per_stage")
    if record_limit != DEMO_RECORD_LIMIT:
        raise ValueError(f"demo record_limit_per_stage must equal {DEMO_RECORD_LIMIT}")
    stages = manifest.get("stages")
    if not isinstance(stages, list) or len(stages) != len(EXPECTED_STAGES):
        raise ValueError("demo manifest must register every expected stage exactly once")

    registered: dict[str, dict] = {}
    for entry in stages:
        if not isinstance(entry, dict) or not isinstance(entry.get("stage"), str):
            raise ValueError("each demo stage entry must be a mapping with a stage name")
        stage = str(entry["stage"])
        if stage in registered:
            raise ValueError(f"duplicate demo stage registration: {stage}")
        registered[stage] = entry
    if set(registered) != set(EXPECTED_STAGES):
        raise ValueError("registered demo stages do not match the expected pipeline")

    expected_files: set[Path] = set()
    counts: dict[str, int] = {}
    project_root = demo_root.parent.parent
    for stage, filenames in EXPECTED_STAGES.items():
        entry = registered[stage]
        files = entry.get("files")
        if not isinstance(files, list) or len(files) != len(filenames):
            raise ValueError(f"demo stage {stage} must register its files exactly once")
        stage_counts = []
        for file_entry, filename in zip(files, filenames):
            if not isinstance(file_entry, dict) or file_entry.get("path") != \
                    f"data/demos/{stage}/{filename}":
                raise ValueError(f"manifest path mismatch for {stage}/{filename}")
            path = demo_root / stage / filename
            expected_files.add(path.resolve())
            stage_counts.append(_validate_file(path, filename, file_entry,
                                               f"{stage}/{filename}"))
        counts[stage] = sum(stage_counts)

    discovered = {path.resolve() for path in demo_root.rglob("*") if path.is_file()}
    expected = expected_files | {(demo_root / "manifest.yaml").resolve()}
    if discovered != expected:
        extra = sorted(str(p) for p in discovered - expected)
        missing = sorted(str(p) for p in expected - discovered)
        raise ValueError(f"unexpected demo file set; extra={extra}, missing={missing}")
    return counts


def main() -> int:
    """Validate all public demos and return a reliable process exit code."""
    args = parse_args()
    print("target: demo-validate")
    print(f"demo_root: {args.demo_root.resolve()}")
    print(f"manifest: {args.manifest.resolve()}")
    print(f"started_at: {datetime.now().astimezone().isoformat()}")
    try:
        counts = validate_demo_release(args.demo_root.resolve(), args.manifest.resolve())
    except (json.JSONDecodeError, OSError, TypeError, ValueError, yaml.YAMLError) as error:
        print("status: failed")
        print(f"error: {error}")
        return 2
    for stage, count in counts.items():
        print(f"{stage}_record_count: {count}")
    print("status: success")
    print(f"finished_at: {datetime.now().astimezone().isoformat()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
