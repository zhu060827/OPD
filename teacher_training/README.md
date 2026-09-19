# 四个代码改写 Teacher 的专家化训练

本目录只训练四个语义保持、code-only 的 Teacher，不训练 Student，也不修改 Stage 1/MOPD。四个 Teacher 使用同一 Qwen3-4B 基座和一致的 LoRA/QLoRA 超参数：

| 内部 ID | 正式名称 | 训练范式依据 |
|---|---|---|
| `formatting` | Style/Formatting | STYLER：格式违规代码到规范代码 |
| `identifier` | Identifier Deobfuscation/Rename | DOBF：混淆标识符到原始自然标识符 |
| `local_structure` | Local Structural Transformation | NatGen：语义保持扰动代码到自然代码 |
| `control_flow` | Control-flow Transformation | NatGen；ContraCode 作为辅助依据 |

正式实验冻结为每个 Teacher 一个训练来源：Formatting=`bigcode/commitpackft`，Identifier=`code_search_net`，Local Structure=`code_search_net`，Control-flow=`deepmind/code_contests`。`prepare_data --formal` 和正式训练启动检查都会拒绝混入其他来源。

这四个来源不能未经适配直接训练：CommitPackFT 和 CodeSearchNet 通常缺少可执行测试；CodeContests 使用 stdin/stdout 测试而当前验证器使用函数级 `assert`。因此在 CodeContests 专用执行器和前三个来源的测试恢复完成前，配置必须保持 `formal_experiment=false`。不得为了增加样本数把“没有测试”记作测试通过。

四类不是严格正交分类。每条样本按照主要变换意图归类，允许少量伴随语法变化，但必须满足领域限制和通用语义门禁。

## 统一训练协议

所有 Teacher 使用 `assistant-only causal language modeling loss`。模型只生成代码：

```xml
<code>
完整目标代码
</code>
```

当前协议为 `teacher-code-only-v1` 和 `teacher-specialization-v4`。旧的 `cot/style/variable/ast` 领域名和 `<reasoning>` 输出不再接受，正式数据需要重新转换。

## 通用硬门禁

四个领域共享四项核心硬门禁，全部是通过/失败判定，不参加加权：

1. 使用对应语言的标准解析器确认解析成功；
2. 按 Fowler 重构定义保持函数签名和公开 API；
3. 使用固定版本语言工具链确认编译成功；
4. 原代码和目标代码在同一留出测试集上全部通过（HumanEval/MBPP 的 execution-based evaluation）。

安全隔离不是模型质量指标，但执行不可信程序时必须启用；静态类型检查仅在任务具有类型契约时启用。Python 当前实现使用标准 AST、编译器和隔离临时目录测试。正式执行不可信数据时还应在服务器容器层设置网络、文件系统、CPU、内存和超时限制。

- `formatting`：只处理格式和布局，不修改标识符及程序结构；
- `identifier`：只处理局部标识符，保持作用域、数据流和控制流；
- `local_structure`：只处理表达式、临时变量和局部语句结构；
- `control_flow`：处理条件、循环、提前返回和路径结构，禁止 bug fix 和算法替换。

## 准备数据

输入示例：

```json
{
  "source_id": "sample-001",
  "domain": "identifier",
  "task": "保持行为不变，将局部标识符恢复为自然名称。",
  "source_code": "def total(VAR_0): ...",
  "target_code": "def total(numbers): ...",
  "tests": ["assert total([1, 2]) == 3"],
  "semantic_pass": true,
  "transformation_type": "identifier_deobfuscation",
  "construction_method": "dobf_style_ast_obfuscation",
  "source_dataset": "CodeSearchNet",
  "source_paper": "DOBF"
}
```

处理一个领域：

```bash
python -m teacher_training.prepare_data \
  --input /path/to/identifier_raw.jsonl \
  --output-dir teacher_training/data/processed/identifier \
  --domain identifier \
  --language python \
  --formal
```

程序生成 `accepted.jsonl`、`rejected.jsonl`、`train.jsonl`、`validation.jsonl`、`test.jsonl` 和 `manifest.json`。划分按原始 `problem_id` 完成；`source_id` 只标识单个变体，从而避免同一问题的多个改写版本跨集合泄漏。

原始数据下载与确定性构造入口：

```bash
python -m teacher_training.scripts.download_source_data \
  --dataset codesearchnet --language python --split train \
  --revision <固定的-Hugging-Face-commit> \
  --output-dir /path/to/raw

python -m teacher_training.scripts.construct_domain_pairs \
  --input /path/to/python_with_tests.jsonl \
  --domain identifier \
  --output /path/to/identifier_raw.jsonl \
  --rejected /path/to/identifier_rejected.jsonl
```

同一构造脚本支持四个领域。它只接收带可执行测试的数据；公开自然代码下载结果必须先关联原始测试，不能因为能够解析就直接进入正式训练。

划分后必须运行独立泄漏审计；它同时检查 `problem_id` 交集和完全相同 target 代码的哈希交集：

```bash
python -m teacher_training.audit_splits \
  --train teacher_training/data/processed/identifier/train.jsonl \
  --validation teacher_training/data/processed/identifier/validation.jsonl \
  --test teacher_training/data/processed/identifier/test.jsonl \
  --output teacher_training/data/processed/identifier/leakage_audit.json
```

## 配置检查与训练

```bash
python -m teacher_training.validate_configs
python -m teacher_training.train_lora --config teacher_training/configs/formatting.json
python -m teacher_training.train_lora --config teacher_training/configs/identifier.json
python -m teacher_training.train_lora --config teacher_training/configs/local_structure.json
python -m teacher_training.train_lora --config teacher_training/configs/control_flow.json
```

正式训练前将四个配置的 `formal_experiment` 改为 `true`，并确认基座、数据和输出路径适合服务器。

改为 `true` 之前运行就绪审计。默认最低量是每领域 train=5000、validation=500、test=500；这是实验预注册的最低规模检查，不是论文指标，可通过参数调整并在实验记录中说明：

```bash
python -m teacher_training.check_training_readiness
```

## 评价设计

| 领域 | 正式主指标 |
|---|---|
| Formatting | exact formatting repair accuracy（STYLER） |
| Identifier | identifier recovery exact-match accuracy（DOBF） |
| Local Structure | CodeBLEU（官方实现） |
| Control-flow | Pass@1 |

语法、签名、编译和测试分别报告为硬门禁通过率，不合并成自定义 `all-gates` 指标，也不与主指标加权。辅助指标只用于解释结果。项目不再输出 `no_change`、`domain_purity_pass`、关键词覆盖或任何内部启发式质量分数。完整版本、实现和出处见 `metric_registry.json`。

生成测试集预测后运行：

```bash
python -m teacher_training.evaluate_domains \
  --predictions /path/to/predictions.jsonl \
  --references teacher_training/data/processed/identifier/test.jsonl \
  --output-dir /path/to/identifier_eval
```

正式依赖固定为 `black==25.1.0`、`pycodestyle==2.12.1`、`sacrebleu==2.5.1`、`codebleu==0.7.0` 和 `radon==6.0.1`。输出包含逐样本 CSV 和领域汇总 JSON。

各领域辅助指标数量不强制相同：

- Formatting：pycodestyle violation count、Levenshtein edit distance；
- Identifier：identifier subtoken F1、full-code exact match；
- Local Structure：BLEU、exact match；
- Control-flow：cyclomatic complexity、CodeBLEU。

交叉领域评估输出 4×4 矩阵。每个测试领域列使用该领域自己的注册主指标，四个门禁各自输出独立矩阵；只能在同一列中比较四个 Teacher，禁止跨列比较或加权：

```bash
python -m teacher_training.evaluate_teachers \
  --input /path/to/scores.jsonl \
  --output /path/to/four_teacher_matrix.json
```

## 文献边界

本项目依据已有论文实际采用过的训练范式组织四个专家，但不声称已有论文训练了完全相同的四 Teacher 系统：Formatting 依据 STYLER，Identifier 依据 DOBF，Local Structure 依据 NatGen，Control-flow 主要沿用 NatGen，ContraCode 仅提供语义保持程序变体训练的辅助依据。数据来源和正式字段见 `dataset_sources.json`。
