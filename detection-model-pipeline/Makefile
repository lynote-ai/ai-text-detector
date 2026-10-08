# Recipe 使用 Bash。
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

# 保持工作区与发布包不产生 Python bytecode 缓存。
export PYTHONDONTWRITEBYTECODE := 1

# 本地默认使用项目约定的 Conda 环境；CI 可通过 PYTHON=python 覆盖。
PYTHON ?= conda run --no-capture-output -n test python
# 开源项目根目录。
PROJECT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
# 项目 manifest 与公开 demo 登记文件。
MANIFEST := $(PROJECT_DIR)/manifest.yaml
DEMO_ROOT := $(PROJECT_DIR)/data/demos
DEMO_MANIFEST := $(DEMO_ROOT)/manifest.yaml

# ① 数据准备：用户自行提供的上游检测日志导出 CSV（必填）。
DATA_CSV ?=
OLD_CSVS ?=
OLD_SCAN_JSONLS ?=
DATA_OUT_DIR ?= data
DATA_CSV_ARG := $(if $(DATA_CSV),--new-csv "$(DATA_CSV)",)
OLD_CSVS_ARG := $(if $(OLD_CSVS),--old-csvs $(OLD_CSVS),)
OLD_SCAN_JSONLS_ARG := $(if $(OLD_SCAN_JSONLS),--old-scan-jsonls $(OLD_SCAN_JSONLS),)

# ② 训练 / ③ 校准 / ④ 评测 / ⑤ 服务。
MODEL_TYPE ?= xlmr
INIT_FROM ?=
TOKENIZER ?=
RUN_NAME ?= run_xlmr
RUN_DIR ?= model_outputs_hier/$(RUN_NAME)
BEST ?= $(RUN_DIR)/best
TRAIN_JSONL ?= data/train_hier.jsonl
VAL_JSONL ?= data/val_hier.jsonl
ALPHA ?= 1.0
OVERSAMPLE_LANGS ?= es pt zh
OVERSAMPLE_FACTOR ?= 2.0
INIT_FROM_ARG := $(if $(INIT_FROM),--init-from "$(INIT_FROM)",)
TOKENIZER_ARG := $(if $(TOKENIZER),--tokenizer "$(TOKENIZER)",)
OVERSAMPLE_ARG := $(if $(OVERSAMPLE_LANGS),--oversample-langs $(OVERSAMPLE_LANGS) --oversample-factor $(OVERSAMPLE_FACTOR),)
HOST ?= 0.0.0.0
PORT ?= 8100
TEXT ?= Artificial intelligence is transforming education. Teachers must adapt their methods.

.PHONY: help env-check manifest-validate demo-validate docs-validate test test-all check \
	data train calibrate eval serve probe detect demo-refresh clean

help:
	@printf '%s\n' \
	  'env-check                         Validate the Python runtime' \
	  'manifest-validate                 Validate the project manifest' \
	  'demo-validate                     Enforce synthetic-only demos, pinned counts, and hashes' \
	  'docs-validate                     Validate bilingual docs, local links, and Mermaid blocks' \
	  'test                              Run the light test subset (no torch, CI-safe)' \
	  'test-all                          Run the full pytest suite (needs requirements-model.txt)' \
	  'check                             Run manifest, demo, docs, and light-test validation' \
	  'data                              Build the two-level soft-label dataset from a log export' \
	  '                                  DATA_CSV=<path-to-csv> (required)' \
	  'train                             Train the joint model (GPU recommended)' \
	  'calibrate                         Fit the document-head temperature on the val split' \
	  'eval                              Evaluate two-level metrics and upstream-label agreement' \
	  'serve                             Start the HTTP service on $(HOST):$(PORT)' \
	  '                                  BEST=<checkpoint-dir> (required)' \
	  'probe                             Smoke-test a running service with curl' \
	  'detect                            Detect one sentence; TEXT="<your text>" (optional)' \
	  'demo-refresh                      Regenerate data/demos/manifest.yaml (maintainer-only)' \
	  'clean                             Remove the training output directory'

env-check:
	@$(PYTHON) --version

manifest-validate: env-check
	@cd "$(PROJECT_DIR)" && $(PYTHON) code/harness/validate_manifest.py --manifest manifest.yaml

demo-validate: env-check
	@cd "$(PROJECT_DIR)" && $(PYTHON) code/harness/validate_demo_data.py \
	  --demo-root data/demos --manifest data/demos/manifest.yaml

docs-validate: env-check
	@cd "$(PROJECT_DIR)" && $(PYTHON) code/harness/validate_docs.py --project-dir .

test: env-check
	@cd "$(PROJECT_DIR)" && PYTHONPATH=code $(PYTHON) -m pytest \
	  tests/test_sentence_splitter.py tests/test_i_prepare_hier_dataset.py \
	  tests/test_harness_validators.py -q

test-all: env-check
	@cd "$(PROJECT_DIR)" && PYTHONPATH=code $(PYTHON) -m pytest tests -q

check: manifest-validate demo-validate docs-validate test

data: env-check
	@test -n "$(DATA_CSV)" || { echo 'DATA_CSV is required (path to a detection-log CSV export)'; exit 2; }
	@cd "$(PROJECT_DIR)" && $(PYTHON) code/i_prepare_hier_dataset.py \
	  $(DATA_CSV_ARG) $(OLD_CSVS_ARG) $(OLD_SCAN_JSONLS_ARG) --out-dir "$(DATA_OUT_DIR)"

train: env-check
	@cd "$(PROJECT_DIR)" && $(PYTHON) code/j_train_hier_model.py --model-type $(MODEL_TYPE) \
	  --train-jsonl "$(TRAIN_JSONL)" --val-jsonl "$(VAL_JSONL)" \
	  $(INIT_FROM_ARG) $(TOKENIZER_ARG) \
	  --output-dir "$(RUN_DIR)" --alpha $(ALPHA) $(OVERSAMPLE_ARG)

calibrate: env-check
	@test -d "$(BEST)" || { echo 'BEST is required (train first, or set BEST=<checkpoint dir>)'; exit 2; }
	@cd "$(PROJECT_DIR)" && $(PYTHON) code/k_calibrate_hier_model.py --model-type $(MODEL_TYPE) \
	  --model-dir "$(BEST)" --val-jsonl "$(VAL_JSONL)"

eval: env-check
	@test -d "$(BEST)" || { echo 'BEST is required (train first, or set BEST=<checkpoint dir>)'; exit 2; }
	@cd "$(PROJECT_DIR)" && $(PYTHON) code/l_eval_hier_model.py --model-type $(MODEL_TYPE) \
	  --model-dir "$(BEST)" --val-jsonl "$(VAL_JSONL)"

serve: env-check
	@test -d "$(BEST)" || { echo 'BEST is required (train first, or set BEST=<checkpoint dir>)'; exit 2; }
	@cd "$(PROJECT_DIR)" && AI_DETECT_MODEL_TYPE=$(MODEL_TYPE) \
	  AI_DETECT_MODEL_DIR="$(abspath $(BEST))" \
	  AI_DETECT_HOST=$(HOST) AI_DETECT_PORT=$(PORT) \
	  $(PYTHON) code/m_serve_hier_api.py

# 起服务后另开终端执行；返回应含 code=0 和 documents[0].predictedClass。
probe:
	curl -s -X POST localhost:$(PORT)/detect \
	  -H 'Content-Type: application/json' \
	  -d '{"text": "Artificial intelligence is transforming education. Teachers must adapt their methods."}' \
	  | python3 -c "import json,sys; d=json.load(sys.stdin); print('code =', d['code']); print('predictedClass =', d['data']['documents'][0]['predictedClass'])"

# 单句检测（句子可配置）：make detect TEXT="你要检测的句子。"
detect:
	TEXT="$(TEXT)" python3 -c "\
	import json, os, urllib.request;\
	req = urllib.request.Request(\
	    'http://localhost:$(PORT)/detect',\
	    data=json.dumps({'text': os.environ['TEXT']}).encode(),\
	    headers={'Content-Type': 'application/json'});\
	d = json.load(urllib.request.urlopen(req));\
	doc = d['data']['documents'][0];\
	print('input:      ', os.environ['TEXT'][:60]);\
	print('code:       ', d['code']);\
	print('predicted:  ', doc['predictedClass'], '/', doc['documentClassification']);\
	print('confidence: ', round(doc['confidenceScore'], 4), doc['confidenceCategory']);\
	print('P(h/a/m):   ', {k: round(v, 3) for k, v in doc['classProbabilities'].items()});\
	print('AI sentences:', sum(s['highlightSentenceForAi'] for s in doc['sentences']), '/', len(doc['sentences']))\
	"

# 编辑 data/demos/ 下任一文件后重算 sha256 并重写登记文件。
demo-refresh: env-check
	@cd "$(PROJECT_DIR)" && $(PYTHON) tools/refresh_demo_manifest.py \
	  --demo-root data/demos

# 清理训练产物（重训前执行；RUN_NAME 可换 run 名）。
clean:
	@test -n "$(RUN_DIR)" || { echo 'RUN_DIR is required'; exit 2; }
	rm -rf "$(PROJECT_DIR)/$(RUN_DIR)"
