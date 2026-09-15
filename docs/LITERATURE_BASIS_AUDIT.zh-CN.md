# 五 Teacher 专家化与 Open-MOPD 项目的文献依据审计

## 审计问题

1. 五个独立领域 Teacher 使用同一基座、分别进行 LoRA/QLoRA 专家化，是否有方法依据？
2. `Planning/CoT → Style/Documentation → Identifier/Rename → Extract/Inline → Control-flow`
   的划分及统一 `<plan>/<code>` 协议，哪些由文献直接支持，哪些是项目设计？
3. 当前数据来源、硬门禁和指标能否作为正式论文实验，哪些表述必须收缩？

## 证据分级

- **A级：直接依据**——论文直接提出或标准 benchmark 明确定义了该方法、任务或指标。
- **B级：组合依据**——组成部分分别有文献，但本项目把它们组合成新的流程；必须做消融。
- **C级：项目定义**——文献只提供概念来源，具体分类、阈值、模板或映射由本项目制定。
- **D级：当前不可作为正式结论**——实现只是兼容代理，或数据与目标任务不一致且未验证。

## 方法链审计

| 项目环节 | 等级 | 审计结论 |
|---|---:|---|
| 五个 Teacher 共享 Qwen3-4B 基座 | B | 控制基座变量是合理实验设计，但没有论文规定必须是五个或必须用 Qwen3-4B。 |
| 每个领域独立 LoRA/QLoRA | A/B | LoRA、QLoRA 有直接依据；把五个领域分别训练成适配器是本项目组合设计。 |
| Assistant-only causal LM loss | A | 指令 SFT 的标准做法；五专家统一损失有利于公平比较。 |
| Planning 先计划再生成代码 | A | Self-Planning 直接研究了 plan-first code generation；CoT、PAL、PoT 提供相邻依据。 |
| 五个专家能力通过 MOPD/Open-MOPD 进入 Student | A/B | MOPD/Open-MOPD 直接支持多教师能力整合；把代码重构五领域接入是本项目应用。 |
| 固定五领域划分 | C | 没有文献证明代码改写天然且完备地分为这五类；必须称为 taxonomy/design choice。 |
| 所有专家统一 `<plan>/<code>` 标签 | C | 统一文本接口受到 instruction tuning 和 text-to-text 范式启发，但字面标签及字段语义由项目定义。 |
| 非 Planning 专家的短 plan | C | 它是 MOPD 接口兼容字段，不是 CoT，也没有证据表明固定模板能提升专家能力。 |
| 5×5 专家分离矩阵 | B | 交叉领域测试是合理诊断；矩阵和“对角优势”判据是项目实验协议，需要报告置信区间和消融。 |

## 五个领域审计

### Reasoning-guided Code Transformation

Self-Planning Code Generation 明确采用先生成计划、再依据计划生成代码的路线，因此
`<plan> + <code>` 对 Reasoning-guided Teacher 有直接依据。这里不为 plan 构造新的质量总分；
其作用通过 `plan+code` 相对 `code-only` 的 Pass@1 消融验证。CodeContests/AlphaCode 提供题目、解法和
测试，但不原生提供人工 CoT；由强模型补全 plan 后再筛选属于合成数据管线，证据等级为 B，必须
报告生成模型、提示词、过滤规则和人工抽查结果。

### Style/Documentation

代码可读性研究和 CodeXGLUE Code Refinement 为该方向提供依据，但 Code Refinement 的核心是
小规模错误修复，不等于纯风格改写。只有在排除功能修复、Rename 和结构变化，并验证可读性标注后，
才能用作 Style 数据。因此“数据集直接对应 Style Teacher”是过强表述。

### Identifier/Rename

标识符预测、变量误用和程序图研究证明定义—使用关系对变量建模重要；但 Variable-Misuse 是定位并
修复错误变量使用，不是生成更好的新名字。将其转换为 Rename 数据属于 B/C 级设计。正式 Rename
训练最好使用真实 rename commit 或具有 `old_identifier/target_identifier/scope` 金标签的数据。

### Extract/Inline

RefactoringMiner 对真实重构检测、Extract/Inline 类型提供较强依据，因此把宽泛 `ast` 收缩为这一
方向是合理的。内部字段 `ast` 只是兼容 ID。RefactoringMiner 主要面向 Java；若正式实验以 Python
为主，需要 Python 重构检测器或人工验证，不能把 Java 检测结果直接宣称为 Python 证据。

### Control-flow

McCabe 的圈复杂度、CFG 研究和重构文献支持控制流分析；ManySStuBs4J 收集的却是 Java 单语句 bug
修复，并非语义保持的控制流重构。它最多是候选来源。必须筛选控制流样本、补充前后测试，并将
bug-fix 与 semantics-preserving refactoring 分开报告。

## 指标审计

| 领域 | 当前主指标 | 结论 | 正式实验建议 |
|---|---|---|---|
| Planning | pass@1 | A级，但只验证代码功能，不验证 plan 质量 | 保留；通过 `code-only` 与 `plan+code` 消融验证 plan 的贡献。 |
| Style | readability score change | B级；Buse–Weimer 有依据，但当前 `style_gain` 不是其正式实现 | 接入可复现可读性模型并做人类相关性校验；否则改用 CodeXGLUE 官方 BLEU/EM，但只能声称 refinement。 |
| Identifier | rename exact-match accuracy | B级；Exact Match 是成熟统计量，但前提是存在唯一金标准名称 | 保留，但必须使用真实 rename 金标签；同时报告 subtoken F1。 |
| Extract/Inline | refactoring-type exact-match accuracy | C级；不是 RefactoringMiner 论文规定的现成 benchmark 指标 | 表述为“基于 RefactoringMiner 标签的 accuracy”；预注册类型集合并人工审计检测误差。 |
| Control-flow | control-flow refactoring-type exact-match accuracy | C级；类型集合及映射由项目定义 | 不得称标准指标；用检测成功率/precision/recall，并独立报告测试通过率和 McCabe complexity。 |

解析、签名、编译和测试是硬门禁，不应与主指标加权。测试只是在给定测试集上的 functional
correctness 证据，不能证明完全语义等价。安全检查属于执行不可信代码时的工程门禁。

CodeBLEU、CodeBERTScore、CrystalBLEU、Self-BLEU、AST/CFG edit distance 和 cyclomatic
complexity 均可作为有来源的辅助指标，但“有论文提出该指标”不意味着它天然适合当前领域；必须解释
指标与研究问题的对应关系。尤其 edit distance 不应被解释为越大越好或越小越好，而应结合功能门禁和
目标重构类型报告分布。

## `<plan>` 协议的正确表述

建议论文写法：

> Inspired by self-planning code generation and instruction tuning, we serialize every teacher output with a
> common plan–code interface. The Planning teacher provides a task-specific reasoning plan, whereas the other
> teachers use a concise transformation-intent field solely for interface compatibility in subsequent
> multi-teacher distillation.

不能写成“已有文献证明所有代码重构专家都应生成 `<plan>`”。`<plan>` 和 `<code>` 标签、非 Planning
模板内容以及 `plan_type` 都是项目协议。若非 Planning 样本大量使用完全相同的 fallback plan，该字段
几乎没有领域监督信息，还会让模型学习固定前缀；正式训练应报告模板占比，并增加以下消融：

1. 四个非 Planning Teacher 使用短 intent；
2. 四个非 Planning Teacher 使用空/最小 intent；
3. Planning Teacher 使用 `plan+code` 与 `code-only`。

## 使正式实验不“凭空产生”的最低要求

1. 将五领域 taxonomy 明确标为本研究提出的任务分解，不冒充公认分类。
2. 冻结领域定义、允许/禁止操作、数据转换规则和标签集合，然后再查看测试结果。
3. 对 CodeContests 合成 plan、Code Refinement 风格筛选、Variable-Misuse→Rename 转换、
   ManySStuBs4J→Control-flow 转换分别做人工双标与一致性统计。
4. 每个主指标必须有真实实现；禁止把 `style_gain`、控制节点差值、关键词覆盖等代理改名为标准指标。
5. 报告四组消融：单基座、仅 prompt、五个独立 LoRA、五 Teacher Open-MOPD。
6. 报告 5×5 交叉领域矩阵；若没有显著对角优势，就不能声称专家已经分离。
7. Student 结果必须与单 Teacher、best Teacher、简单混合/路由及完整 Open-MOPD 比较，才能把收益归因
   于多教师能力整合。
8. 固定随机种子并报告至少多个训练种子或置信区间；保存数据 hash、模型版本、prompt 版本和代码 commit。

## 总结判断

项目的主方法链——领域 SFT/QLoRA、Planning-first code generation、测试驱动验证和多教师
Open-MOPD——均有可核验的文献基础。创新和风险集中在五领域 taxonomy、公开数据的二次转换、统一
plan 接口及领域指标映射；这些不是错误，但属于项目设计，必须通过标注审计、消融和交叉领域实验建立
证据。当前最不稳妥的说法是把五个数据集称为五个专家的天然数据，或把两个 refactoring-type accuracy
称为 RefactoringMiner 已定义的标准 benchmark 指标。

## 已核验参考文献

1. Jiang et al. Self-Planning Code Generation with Large Language Models. ACM TOSEM, 2024.
   DOI: <https://doi.org/10.1145/3672456>
2. Wei et al. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. 2022.
   <https://arxiv.org/abs/2201.11903>
3. Hinton, Vinyals, Dean. Distilling the Knowledge in a Neural Network. 2015.
   <https://arxiv.org/abs/1503.02531>
4. Open-MOPD: Diagnosing and Fixing Capability Imbalance in Multi-Teacher On-Policy Distillation. 2026.
   <https://arxiv.org/abs/2608.19098>
5. MOPD: Multi-Teacher On-Policy Distillation for Capability Integration in LLM Post-Training. 2026.
   <https://arxiv.org/abs/2606.30406>
6. Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. 2021.
   <https://arxiv.org/abs/2106.09685>
7. Dettmers et al. QLoRA: Efficient Finetuning of Quantized LLMs. 2023.
   <https://arxiv.org/abs/2305.14314>
8. Chen et al. Evaluating Large Language Models Trained on Code. 2021.
   <https://arxiv.org/abs/2107.03374>
9. Li et al. Competition-level Code Generation with AlphaCode. Science, 2022.
   DOI: <https://doi.org/10.1126/science.abq1158>
10. Lu et al. CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation. 2021.
    <https://arxiv.org/abs/2102.04664>
11. Buse and Weimer. Learning a Metric for Code Readability. IEEE TSE.
    DOI: <https://doi.org/10.1109/TSE.2009.70>
12. Tsantalis et al. RefactoringMiner / RefactoringMiner 2.0. IEEE TSE.
    DOI: <https://doi.org/10.1109/TSE.2020.3007722>
13. Karampatsis and Sutton. How Often Do Single-Statement Bugs Occur? The ManySStuBs4J Dataset. 2019.
    <https://arxiv.org/abs/1905.13334>
14. Ren et al. CodeBLEU: a Method for Automatic Evaluation of Code Synthesis. 2020.
    <https://arxiv.org/abs/2009.10297>
