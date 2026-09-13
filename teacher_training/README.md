# 五个 Teacher 专家化训练（Qwen3-4B）

本目录只训练五个冻结的 Teacher，不进行 Student SFT。五个 Teacher 共享
`Qwen/Qwen3-4B` 基座模型，分别保存独立的 LoRA/QLoRA 适配器。

## 完整流程

1. 在 GPU 环境安装 `teacher_training/requirements.txt`。
2. 按各自许可证下载五个公开数据集，不把原始大文件提交到 Git。
3. 将数据转换为包含 `source_code`、`target_code`、`domain`、`source_id` 和
   `semantic_pass` 的 JSONL；CoT 还必须包含 `target_reasoning`。
4. 使用 `prepare_data` 分别处理五个领域，执行领域规则、语言标记和语法检查。
5. 检查 `accepted.jsonl`、`rejected.jsonl` 和 `manifest.json`。
6. 用人工标注验证集运行 `calibrate_thresholds`，把校准阈值传给 `prepare_data`。
7. 运行 `validate_configs`，再对每个配置执行 `--dry-run`。
8. 分别训练五个 LoRA Teacher；训练过程不会加载或更新 Student。
9. 在五个留出领域上进行交叉评估，生成 5×5 评分矩阵。
10. 将适配器路径填入 Stage 2 的 `teacher_path`。

## 数据格式与验证

原始 JSONL 可以使用 `before_code/after_code`、`source_code/target_code` 或
`original_code/rewritten_code`。Python 代码会在本地解析；Java 代码必须已通过上游
解析器和测试，并带有 `semantic_pass: true`。

领域证据要求：

- CoT：必须有目标推理文本；
- Style：必须有可观察的格式、文档或组织变化；
- AST：必须有结构变化，不能只有变量名变化；
- Variable：必须有标识符变化和 AST 证据；
- Control-flow：必须有控制流节点变化。

不合格样本写入 `rejected.jsonl`，合格样本写入 `accepted.jsonl`，统计信息写入
`manifest.json`。训练划分按 `source_id` 完成，避免同一道题的不同改写跨集合泄漏。

Python 样本的语义门禁复用项目 multi-expert 的
`SemanticEquivalenceChecker`：要求 AST 可解析、函数签名保持、安全检查通过、可编译、
测试全部通过。随后复用 `CodeQualityEvaluator` 计算风格、命名、复杂度和 AST 指标。
阈值不是论文规定的固定常数，必须使用人工标注验证集校准。

```bash
python -m teacher_training.prepare_data \
  --input /path/to/ast_pairs.jsonl \
  --domain ast \
  --language python \
  --output-dir teacher_training/data/processed/ast
```

校准示例：

```bash
python -m teacher_training.calibrate_thresholds \
  --input /path/to/human_validation.jsonl \
  --output /path/to/calibrated_thresholds.json \
  --min-precision 0.90

python -m teacher_training.prepare_data \
  --input /path/to/variable_pairs.jsonl \
  --domain variable \
  --thresholds /path/to/calibrated_thresholds.json \
  --output-dir teacher_training/data/processed/variable
```

人工验证集每行至少包含：`domain`、`accepted_by_human` 和 `metrics`。建议先由两名标注者
独立判断“是否真正属于该领域且语义保持”，再冻结满足 precision≥0.90 的阈值。

## 检查配置并训练

```bash
python -m teacher_training.validate_configs
python -m teacher_training.train_lora \
  --config teacher_training/configs/cot.json \
  --dry-run
python -m teacher_training.train_lora \
  --config teacher_training/configs/cot.json
```

训练前在 GPU 环境安装依赖：

```bash
pip install -r teacher_training/requirements.txt
```

## 五个主要数据来源

- CoT：CodeContests，由通过测试的解法生成算法推理；
- Style：CodeXGLUE Code Refinement，筛选风格和组织改动；
- AST：Refactory 或 RefactoringMiner，保存 AST 改动证据；
- Variable：CodeXGLUE Variable-Misuse，加上变量重命名改写对；
- Control-flow：ManySStuBs4J 中的控制流相关修改，辅以控制流重构数据。

CodeXGLUE 和 ManySStuBs4J 主要是 Java 数据，而当前下游 MBPP 流程主要使用 Python。
正式实验应记录每条样本的 `language`，并分别报告同语言结果和跨语言迁移结果。

## 交叉领域评估

推理结果每行需要包含 `teacher_domain`、`sample_domain`、`score` 和
`semantic_pass`：

```bash
python -m teacher_training.evaluate_teachers \
  --input /path/to/prediction_scores.jsonl \
  --output /path/to/teacher_matrix.json
```

理想情况下，5×5 矩阵的对角线分数高于非对角线，同时保持较高的语义通过率。
