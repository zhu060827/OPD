# 四 Teacher 专家化训练的文献依据审计

## 结论边界

本分支训练四个 code-only Teacher：`formatting`、`identifier`、`local_structure`、`control_flow`。
四领域划分是受既有任务启发的实验分解，并非任何论文提出的标准完备 taxonomy。论文中必须如此表述，
不能声称已有工作证明代码改写天然分为这四类。

## 训练目标

四个 Teacher 统一使用 assistant-only causal language modeling loss，只对 assistant 的 `<code>` 内容计算
next-token cross-entropy。指令 SFT/causal LM 提供训练依据；LoRA/QLoRA 提供参数高效微调依据。
统一目标便于控制变量，但“四个适配器”的组合仍是本项目实验设计。

## 四个领域的直接依据

| 领域 | 数据构造/训练范式 | 主指标 | 主要依据 |
|---|---|---|---|
| Formatting | 格式违规代码恢复为规范代码 | exact formatting repair accuracy | STYLER |
| Identifier | DOBF 式局部标识符混淆到自然名称恢复 | identifier recovery exact match | DOBF |
| Local Structure | NatGen 式语义保持局部扰动到自然代码 | CodeBLEU | NatGen；CodeBLEU |
| Control-flow | NatGen/ContraCode 式语义保持控制流变体 | Pass@1 | NatGen；ContraCode；HumanEval |

ContraCode 是对比学习工作，只作为“语义保持程序变体”的辅助依据，不能写成它直接训练了当前生成式
Control-flow Teacher。CommitPackFT 和 CodeSearchNet 是自然代码来源，也不能未经转换直接视为四领域金标签。

## 四项核心硬门禁

1. 标准解析器解析成功；
2. 函数签名和公开 API 保持，符合 Fowler 对重构保持可观察行为的定义；
3. 固定语言工具链编译成功；
4. source 和 target 在同一留出测试上全部通过，遵循 HumanEval/MBPP 的执行式验证范式。

这些项目是通过/失败门禁，不参与加权。测试通过只表示通过给定测试，不证明数学意义上的完全语义等价。
安全隔离属于执行不可信代码的实验条件；静态类型检查只在数据提供类型契约时启用。

## 正式指标

每个领域只用一个主指标，不计算跨指标总分。辅助指标仅描述结果：

- Formatting：pycodestyle violation count；Levenshtein edit distance；
- Identifier：identifier subtoken F1；full-code exact match；
- Local Structure：SacreBLEU；AST 规范化 exact match；
- Control-flow：McCabe cyclomatic complexity；CodeBLEU。

所有实现及固定版本写在 `teacher_training/metric_registry.json`。AST 变化与允许变换类型只用于构造和
筛选数据，不作为评价指标。正式输出不包含领域纯度、no-change 或自定义综合通过率。

## 正式实验最低要求

1. 训练前冻结允许的 transformation type、工具版本、数据划分和指标注册表；
2. 同源变体按 `source_id` 分组划分，防止训练/测试泄漏；
3. 保存原始来源、转换方法、四门禁证据、数据 hash、模型版本和代码 commit；
4. 主结果分别报告四个主指标及置信区间，不加权成一个分数；
5. 报告基座、仅提示、独立 LoRA 以及后续 MOPD 的消融；
6. Stage 1/MOPD 接入前另行把旧五领域协议迁移为本分支的四领域协议。

## 参考文献

1. Palma et al. *STYLER: Learning Formatting Conventions to Repair Checkstyle Warnings*.
2. Lachaux et al. (2021). *DOBF: A Deobfuscation Pre-Training Objective for Programming Languages*.
3. Chakraborty et al. (2022). *NatGen: Generative Pre-training by Naturalizing Source Code*.
4. Jain et al. (2021). *ContraCode: Learning Contrastive Representations for Code*.
5. Ren et al. (2020). *CodeBLEU: a Method for Automatic Evaluation of Code Synthesis*.
6. Chen et al. (2021). *Evaluating Large Language Models Trained on Code*.
7. Austin et al. (2021). *Program Synthesis with Large Language Models*.
8. McCabe (1976). *A Complexity Measure*.
9. Papineni et al. (2002). *BLEU: a Method for Automatic Evaluation of Machine Translation*.
10. Post (2018). *A Call for Clarity in Reporting BLEU Scores*.
11. Levenshtein (1966). *Binary Codes Capable of Correcting Deletions, Insertions, and Reversals*.
12. Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*.
13. Dettmers et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*.
