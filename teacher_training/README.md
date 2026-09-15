# 五个 Teacher 专家化训练（Qwen3-4B）

本目录只训练五个冻结的 Teacher，不进行 Student SFT。五个 Teacher 共享
`Qwen/Qwen3-4B` 基座模型，分别保存独立的 LoRA/QLoRA 适配器。
`configs/teacher_template.json` 只是模板，不是第六个 Teacher；实际训练只使用
`cot.json`、`style.json`、`ast.json`、`variable.json` 和 `control_flow.json`。

## 完整流程

1. 在 GPU 环境安装 `teacher_training/requirements.txt`。
2. 按各自许可证下载五个公开数据集，不把原始大文件提交到 Git。
3. 将数据转换为包含 `source_code`、`target_code`、`domain`、`source_id` 和
   `semantic_pass` 的 JSONL。CoT 正式数据必须同时包含 `source_reasoning` 和
   `target_reasoning`；新协议不再读取或输出 plan 字段。
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

五个专家统一监督输出为：

```text
<reasoning>
CoT 为完整联合改写推理；其他领域为简短变换依据
</reasoning>

<code>
目标改写代码
</code>
```

`reasoning_origin` 和 `reasoning_from_source` 会记录文本来源。CoT 的 `<reasoning>` 是完整推理；
其他四个专家只是简短 transformation rationale，用于统一格式和后续 MOPD，不声称为 CoT。

规范化数据还会保存：

```json
{
  "domain": "ast",
  "domain_name": "Extract/Inline",
  "reasoning_type": "transformation_rationale",
  "output_schema_version": "teacher-reasoning-code-v2",
  "system_prompt_version": "teacher-specialization-v3"
}
```

CoT 的 `reasoning_type` 为 `full_reasoning`，其他四个领域均为 `transformation_rationale`。
训练前会严格检查领域显示名称、reasoning 类型、协议版本和 system prompt，防止旧数据或其他专家的
样本混入当前 LoRA。五个配置还必须使用同一基座和彼此不同的输出目录，因此训练其他专家不会
覆盖已经保存的 CoT 适配器。

领域证据要求：

- CoT：输出逐步代码分析和改写代码；正式数据禁止固定模板，冒烟数据才允许模板回退；
- Style：必须有可观察的格式、文档或组织变化；
- AST：必须有结构变化，不能只有变量名变化；
- Variable：必须有标识符变化和 AST 证据；
- Control-flow：必须有控制流节点变化。

内部字段保持稳定：`cot / style / ast / variable / control_flow`。论文和报告使用名称：
`Reasoning-guided Code Transformation`、`Style/Documentation`、`Identifier/Rename`、`Extract/Inline`、
`Control-flow`。其中内部 `ast` 仅表示 Extract/Inline 兼容 ID，不再表示所有 AST 变化；
内部 `variable` 表示 Identifier/Rename。

不合格样本写入 `rejected.jsonl`，合格样本写入 `accepted.jsonl`，统计信息写入
`manifest.json`。训练划分按 `source_id` 完成，避免同一道题的不同改写跨集合泄漏。

Python 样本的语义门禁复用项目 multi-expert 的 `SemanticEquivalenceChecker`。四个核心门禁为：
语言解析成功、函数签名/公开 API 保持、编译成功、原代码与改写代码测试通过。安全检查在数据
来源不可信或允许外部 API 时作为条件门禁启用。Java 等语言替换为相应解析器、编译器和测试框架。
随后复用 `CodeQualityEvaluator` 产生筛选证据。
阈值不是论文规定的固定常数，必须使用人工标注验证集校准。这些规则只负责数据筛选，
不能作为论文中的领域主指标。

## 正式主指标

每个 Teacher 只指定一个最匹配的领域主指标，详细机器可读定义见
`teacher_training/metric_registry.json`：

| Teacher | 唯一领域主指标 | 主要依据 |
|---|---|---|
| Reasoning-guided Code Transformation | pass@1 | Chen et al. (2021), HumanEval |
| Style/Documentation | code readability score change | Buse and Weimer (2010) |
| AST-local Extract/Inline | refactoring-type exact-match accuracy | Tsantalis et al. (2018) |
| Identifier/Rename | rename exact-match accuracy | Allamanis et al. (2018)；CodeXGLUE |
| Control-flow | control-flow refactoring-type exact-match accuracy | Tsantalis et al. (2018) |

正式论文结果必须调用相应标准/官方实现。旧版项目中的 AST 兼容相似度、风格增益、命名增益、
控制节点差值、关键词覆盖、推理结构完整度、推理—代码一致率、人工评分和 Judge 评分已从
自动正式汇总中删除；它们不能作为正式指标。每个领域另选 2～4 个有来源的辅助指标，仅用于
展示效果；具体列表和引用见指标注册表。训练目标对五个 Teacher 完全一致且只有一个：
`assistant-only causal language modeling loss`。

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

## 五个主要数据来源（v2）

- Reasoning-guided Code Transformation：CodeContests；从同题通过测试的实现构造代码对，并生成或提取逐样本分析；正式数据禁止固定模板；
- Style/Documentation：CommitPackFT/CommitPack 真实提交池，筛选非功能性风格与文档变更；
- Identifier/Rename：CommitPack 提交池，用 RefactoringMiner 检测 Rename 并保存旧名、新名和作用域；
- Extract/Inline：CommitPack 提交池，用 RefactoringMiner 只抽取 Extract/Inline 类型；
- Control-flow：CommitPack 提交池，使用预注册 AST/CFG 规则和测试过滤控制流变更。

这不是“五个天然一一对应的数据集”。除 CoT 外，三个重构领域可以共享同一真实 commit 原始池，
再按可验证标签形成互斥子集。CodeXGLUE Code Refinement、Variable-Misuse 和 ManySStuBs4J
分别包含 bug repair/variable misuse/single-statement bug fix，不再作为 Style、Rename 和语义保持
Control-flow 的天然主训练集；它们只用于辅助或对照。完整来源、下载地址、缺失字段和转换要求见
`dataset_sources.json`。

正式处理必须加入 `--formal`：

```bash
python -m teacher_training.prepare_data \
  --input /path/to/converted_domain_pairs.jsonl \
  --domain variable \
  --language java \
  --formal \
  --output-dir teacher_training/data/processed/variable
```

正式配置另设 `"formal_experiment": true`。此时训练入口拒绝未通过领域数据契约的样本。MBPP
冒烟数据不使用 `--formal`，配置保持 `formal_experiment: false`。正式实验应记录每条样本的语言，
并分别报告同语言结果和跨语言迁移结果。

## 交叉领域评估

基座与 LoRA 输出对比也可同时生成图表：

```bash
python -m teacher_training.compare_base_lora \
  --base-model /root/autodl-tmp/models/Qwen3-4B \
  --adapter /root/autodl-tmp/models/teacher-cot-lora \
  --test-file teacher_training/data/processed/cot/test.jsonl \
  --output /root/autodl-tmp/results/cot_base_vs_lora.jsonl \
  --plot /root/autodl-tmp/results/cot_base_vs_lora.png \
  --limit 20
```

图表展示每个测试样本的输出长度和推理要素覆盖数量。它们是可解释性辅助指标，不能
替代语义测试、领域质量指标或人工评估。

推理结果每行需要包含 `teacher_domain`、`sample_domain`、`score` 和
`semantic_pass`：

```bash
python -m teacher_training.evaluate_teachers \
  --input /path/to/prediction_scores.jsonl \
  --output /path/to/teacher_matrix.json
```

理想情况下，5×5 矩阵的对角线分数高于非对角线，同时保持较高的语义通过率。
