# Best AI Text Detector

<p align="center">
  <img src="assets/hero.svg" alt="AI Detector Skill hero" width="100%" />
</p>

<p align="center">
  <a href="https://github.com/lynote-ai/ai-detector-skill/actions/workflows/ci.yml"><img src="https://github.com/lynote-ai/ai-detector-skill/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <img src="https://img.shields.io/badge/python-3.9%2B-3776ab" alt="Python 3.9+" />
  <img src="https://img.shields.io/badge/license-MIT-16a34a" alt="MIT License" />
  <img src="https://img.shields.io/badge/network-none-f59e0b" alt="No Network" />
</p>

<p align="center">
  An explainable, cautious AI-generated text risk analyzer for coding agents and local workflows.
</p>

<p align="center">
  <a href="./README.zh-CN.md">简体中文</a>
  ·
  <a href="#install">Install</a>
  ·
  <a href="#evaluation">Evaluation</a>
  ·
  <a href="#use-cases">Use Cases</a>
</p>

**Other Quality Projects**</br>
AI Humanize Text:https://github.com/lynote-ai/humanize-text</br>
AI Image Detector:https://github.com/lynote-ai/ai-image-detector</br>

## Why This Exists

Most AI text detectors are either overconfident, opaque, or awkward to embed inside agent workflows.

`ai-detector-skill` takes the opposite approach:

- explainable weighted signals instead of hidden model claims
- a local CLI and Python API that are easy to script
- a short-text guardrail that refuses to overstate weak evidence
- skill-ready packaging for Codex, Claude Code, and other repo-aware agents
- reproducible benchmark and dataset evaluation scripts

If you want a **triage tool** that stays cautious and leaves room for human review, this repo is built for that.

## Install

```bash
pip install -e .
ai-detect examples/sample_ai_like.txt --json
```

Or bootstrap a local environment:

```bash
bash scripts/setup.sh
```

Example output:

```text
Conclusion: AI-like signals are present, but this medium-confidence score is a risk estimate rather than proof.
Score: 84/100
Confidence: medium
Verdict: high_ai_likelihood
Words analyzed: 256
```

## What You Get

- `score`: 0-100 AI-like writing risk estimate
- `verdict`: `insufficient_text`, `low_ai_likelihood`, `mixed_or_uncertain`, or `high_ai_likelihood`
- `confidence`: currently `low` or `medium`
- `signals`: strongest weighted evidence signals
- `caveats`: warnings that stay attached to the result
- `next_steps`: practical follow-up actions when useful

This is designed to help an agent say:

- "AI-like signals are present."
- "The sample is too short for a meaningful estimate."
- "This should be reviewed against known writing samples."

Not:

- "This was definitely written by AI."
- "The detector proves misconduct."

## Usage

### CLI

```bash
ai-detect examples/sample_ai_like.txt
cat essay.txt | ai-detect --json
python scripts/detect.py examples/sample_human_like.txt --json
```

### Python API

```python
from aidetect import analyze_text

text = open("essay.txt", encoding="utf-8").read()
result = analyze_text(text)

print(result.score, result.confidence, result.verdict)
for signal in result.strongest_signals():
    print(signal.name, signal.note)
```

### Local Skill Install

```bash
cp -R ai-detector-skill "$CODEX_HOME/skills/"
```

Use the root [SKILL.md](./SKILL.md) as the portable skill definition, and keep [AGENTS.md](./AGENTS.md) at the repo root for repo-aware agents.

## Evaluation

We ran a reproducible evaluation on the public [HC3 dataset](https://huggingface.co/datasets/Hello-SimpleAI/HC3), using the English `finance`, `medicine`, and `open_qa` subsets with the first 100 rows from each subset.

Snapshot of the current detector on that slice:

- Human mean score: `5.4`
- AI mean score: `18.4`
- Mean separation: `13.0` points
- Human coverage: `0.427`
- AI coverage: `0.920`
- Covered accuracy at `score >= 45`: `0.317`

What that means in practice:

- the detector separates human and AI answers on average, but only weakly on HC3
- the short-text guardrail is doing useful work, especially on shorter human answers
- the current thresholds are conservative, which keeps false confidence down but also lowers recall
- this tool works better as **triage + explanation** than as a stand-alone classifier

See the full report in [docs/HC3_EVALUATION.md](./docs/HC3_EVALUATION.md).

Reproduce it with:

```bash
make eval-hc3
```

## Use Cases

### Teacher Triage

A teacher receives a polished 400-word reflection and wants a cautious signal before doing manual review.

Suggested workflow:

1. Run `ai-detect submission.txt --json`.
2. Read the strongest signals and caveats.
3. Compare the passage with known writing samples before making any judgment.

### Editorial Review

An editor wants to spot formulaic product reviews or guest posts before spending time on manual edits.

Why it fits:

- medium-length prose works better than short snippets
- explainable signals help justify why a draft feels templated

### Trust And Safety Queueing

A moderation team wants to sort suspicious long-form posts into a manual review queue, not auto-remove them.

Why it fits:

- the tool is conservative by design
- it helps with prioritization more than enforcement

### Internal Content QA

A team compares human drafts and AI-assisted drafts to see where language starts sounding too generic.

Why it fits:

- the score is useful as a relative signal across versions
- strongest signals can guide rewriting

## Not For

- disciplinary decisions about a named student or employee
- treating a single score as proof of cheating or fraud
- very short samples under about 80 words
- high-stakes authorship disputes without known-sample comparison

## Project Structure

```text
ai-detector-skill/
├── SKILL.md
├── scripts/
│   ├── detect.py
│   ├── setup.sh
│   ├── benchmark.py
│   └── evaluate_hc3.py
├── references/
│   └── api-reference.md
├── assets/
│   ├── hero.svg
│   ├── score-bands.svg
│   ├── workflow.svg
│   └── templates/
│       └── report.md
├── src/aidetect/
├── tests/
├── AGENTS.md
└── README.md
```

## Dev Commands

```bash
make test
make demo
make benchmark
make eval-hc3
```

## CI

GitHub Actions automatically runs:

- `make test` on Python `3.9`, `3.11`, and `3.13`
- `make benchmark` to regenerate the synthetic benchmark report
- `make eval-hc3` to regenerate the HC3 evaluation report
- upload of `docs/BENCHMARK.md` and `docs/HC3_EVALUATION.md` as workflow artifacts

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md).

Good contributions usually improve one of these:

- clearer evidence signals
- safer wording and UX
- multilingual handling that stays explainable
- reproducible evaluation coverage
- agent integration ergonomics

## License

MIT
