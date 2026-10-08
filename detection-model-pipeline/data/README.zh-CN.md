# Demo 数据

[英文版](README.md)

本目录只发布 `demos/`：三个流水线阶段的 5 个文件——25 条合成 JSONL 记录
（20 训练 + 5 验证）加两个 JSON 产物（一份校准结果、一份完整 detect 响应
信封）。每条记录均为本仓库撰写，并携带溯源键 `"source": "synthetic"`——
检测服务日志中的真实用户文本从不写该字段，因此永远过不了
`make demo-validate`。

`data/demos/manifest.yaml` 锁定选取策略、各文件记录数与 SHA-256 摘要。
完整的训练/验证数据集**故意不发布**：它们派生自用户提供的日志导出，可能
包含个人数据。

## Schema

- `01_prepare/train_hier_demo.jsonl` 与 `01_prepare/val_hier_demo.jsonl`：
  双级软标签记录（`scan_id`、`text`、`doc_label`、
  `sentences[].{text, y_origin, y_human_ai}`、`gptzero_doc_class`、
  `gptzero_language`），外加 demo 独有的 `source` 键。训练文件中的每个
  `scan_id` 满足 `md5(scan_id)[:8] % 10 != 1`；验证文件的每个满足 `== 1`。
  语言覆盖 en / zh / es / pt，包含 `human`、`ai`、`mixed` 三类文档。
- `02_calibrate/calibration_demo.json`：与
  `code/k_calibrate_hier_model.py` 产出完全一致的产物——`temperature`、
  `doc_class_names`、`fitted_samples`。
- `03_serve/detect_response_demo.json`：单篇文档的完整 `{code, msg, data}`
  信封，与服务常量自洽（`V2-hier-1` / `V2h`、0.33/0.6/0.8 分档、中性
  resultMessage 模板、段落均值、高亮规则）。

记录 schema 与溯源说明见
[../docs/DATA_CONSTRUCTION.zh-CN.md](../docs/DATA_CONSTRUCTION.zh-CN.md)。

## 发布门禁

```bash
make demo-validate
```

以下任一情况都会让门禁失败：记录缺少 `source: synthetic` 溯源键；某个
`scan_id` 违反其文件的划分规则；概率向量和不等于 1；句子不是其文档文本的
子串；文件记录数或 SHA-256 与登记文件漂移；`data/demos/` 下出现未登记的
文件。登记文件本身由维护者专用的 `make demo-refresh` 重建。

## 为何两个阶段没有 demo 文件

`j_train_hier_model`（训练）与 `l_eval_hier_model`（评测）的产物是模型权重
与随运行变化的报表，不是有界、可复用的数据产物，因此不发布 demo 文件；
`manifest.yaml` 的 notes 同样注明。

## 发布前

自动门禁约束的是数量与完整性。若你以自己的名义再分发这些文件，仍需人工
复核全部 25 条记录的许可与来源——门禁替代不了判断。
