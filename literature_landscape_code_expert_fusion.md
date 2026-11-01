# 面向代码专家训练与多教师融合的相关研究综述

**检索截止：2026-10-29**

## 1. 项目定位

本项目训练四个面向代码重写的 LoRA 专家：Formatting、Identifier、Local Structure 和 Control Flow；随后让一个 Student 在同一条 rollout 上接受多个 Teacher 的 token-level distillation。当前方法假设是：每个专家负责相对可分解的代码变换，Teacher 用 Student 生成轨迹上的 token log-probability gap 提供奖励，再通过任务间校准和路由选择合适的 Teacher。

需要强调的是，“训练四个 LoRA 专家”本身不是充分的新颖性。论文应把贡献集中在：

1. 任务解耦的代码重写专家体系；
2. 同一 Student rollout 上的多教师 token-level OPD/MOPD；
3. 面向异构任务奖励的校准、路由与 token 平衡；
4. 同时使用执行正确性、结构约束和最小编辑约束的评测协议。

## 2. 代表性工作

| 研究方向 | 代表工作 | 主要贡献 | 与本项目的关系 |
|---|---|---|---|
| 代码自对齐 | [SelfCodeAlign, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/72da102da91a8042a0b2aa968429a9f9-Abstract-Conference.html) | 自动生成任务、多次采样、测试过滤，再进行代码模型自对齐 | 应作为“单模型/自训练”基线；本项目应证明多专家路由比单一自对齐更有效 |
| 代码自调试与执行反馈 | [LeDex, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/3ea832724870c700f0a03c665572e2a9-Abstract-Conference.html) | 使用执行反馈构造 self-debug/refinement 轨迹，并结合 SFT/RL | 与 Control Flow 和 hard-gate 评测直接相关 |
| 多源代码训练 | [AlchemistCoder, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/040c816286b3844fd78f2124eec75f2e-Abstract-Conference.html) | 整合多源代码数据并进行统一调优 | 是 Student 的混合数据 SFT 基线；可检验专家解耦是否优于简单混合 |
| 编辑感知代码修复 | [QiMeng-PRepair, ACL 2026](https://aclanthology.org/2026.acl-long.1343/) | 通过受控 bug 注入和 edit-aware GRPO 减少 over-editing | 与 Formatting、Levenshtein、AST identity 和最小修改约束高度相关 |
| 动态修复策略选择 | [PATSSEL, Findings EMNLP 2026](https://aclanthology.org/2026.findings-emnlp.2173/) | 根据测试反馈动态选择 debug 或 regenerate | 是动态策略选择和 verifier-guided routing 的相邻基线 |
| 代码修复基准 | [Code Reffix, Findings ACL 2026](https://aclanthology.org/2026.findings-acl.762/) | 将 reflection generation 与 repair 分开评估 | 提醒论文不能只报告最终代码相似度，应区分中间能力与最终修复 |
| 代码评测可靠性 | [CodeJudgeBench, ACL 2026](https://aclanthology.org/2026.acl-long.888/) | 研究 LLM-as-a-judge 在代码生成/修复评价中的不稳定性 | 支持本项目优先采用执行、解析、签名和结构 hard gates，而不是只依赖 judge |
| 代码调试数据与基准 | [COAST, Findings NAACL 2025](https://aclanthology.org/2025.findings-naacl.139/) | 代码调试基准与 agent-based 数据合成 | 可作为数据构造和 debugging evaluation 的背景工作 |
| Adapter 合并 | [MergeRepair, EMSE 2026](https://link.springer.com/article/10.1007/s10664-026-10943-y) | 研究将任务专用 adapters 合并用于自动程序修复 | 是本项目路由/蒸馏方法必须比较的强基线；能回答“为何不直接 merge adapters” |
| 多维代码质量评测 | [PROBE, EMSE](https://link.springer.com/article/10.1007/s10664-026-10904-5) | 同时评估功能正确性、结果接近度和代码质量 | 支持本项目的主指标 + 辅助指标 + hard-gate 设计 |
| 静态质量修复 | [Static repair evaluation, EMSE 2026](https://link.springer.com/article/10.1007/s10664-026-10858-8) | 引入静态修复成功率和 non-regression 检查 | 可作为 Formatting 的质量/回归检查参考 |
| Reward calibration | [ReCal, arXiv 2026](https://arxiv.org/abs/2606.12479) | 指出异构任务 reward 不可直接比较，提出分量化和分布感知校准 | 支持“需要校准”的动机，但不能声称 median/MAD 本身来自 ReCal |
| OPD/MOPD | [NVIDIA NeMo MOPD documentation](https://github.com/NVIDIA-NeMo/RL/blob/main/docs/about/algorithms/mopd.md) | 使用 teacher-student token log-probability gap 作为 on-policy distillation advantage | 是本项目 token reward 的直接方法依据 |
| Open-MOPD | [Open-MOPD official project page](https://bytedtsinghua-sia.github.io/Open-MOPD/) | 路由 Teacher、top-k token reward、token-share balancing、gap-following、reward refresh | 是本项目 Student 训练方案的直接实现参考；当前公开材料主要是 2026 arXiv/官方实现，而非成熟的长期共识基线 |

## 3. 研究现状

### 3.1 从“生成代码”转向“可验证的代码修改”

近期工作普遍不再把 BLEU 或表面相似度作为唯一结论，而是把单元测试、编译/解析、结构保持、最小编辑和 non-regression 检查结合起来。SelfCodeAlign 和 LeDex 代表了“生成—验证—筛选/改进”的路线；QiMeng-PRepair 和 PATSSEL 则进一步把编辑范围控制和策略选择纳入训练或推理。因而，本项目的执行测试、AST/签名 hard gate 和 Levenshtein 指标是合理方向，但必须明确每个指标的任务含义。

### 3.2 多源数据与多任务训练已经成熟，但简单混合不一定足够

AlchemistCoder 等工作说明多源代码数据可以通过统一训练获得强结果；然而，简单混合会带来任务冲突、标签格式冲突和 reward 尺度不一致。MergeRepair 提供了一个重要对照：参数级 adapter merge 可以作为廉价融合方案，但不等价于在同一 rollout 上进行动态 Teacher 选择。因此论文应直接比较 mixed SFT、adapter merge 和 routed distillation。

### 3.3 OPD/MOPD/Open-MOPD 是较新的方法线

MOPD 的核心是对 Student 已生成 token 计算 teacher 与 student 的 log-probability gap，并将其作为 on-policy advantage；top-k 版本只需查询 Student 采样 token 的 Teacher log-probability。Open-MOPD 在此基础上加入 hard routing、token-share balancing、gap-following 和 reward refresh。这个方向与本项目高度匹配，但 Open-MOPD 仍属于较新的方法与实现，不能只引用公式后宣称贡献成立，必须进行 routing、校准、top-k、reward refresh 等消融。

### 3.4 评测可靠性本身已经成为研究问题

CodeJudgeBench 表明 LLM judge 会受到响应顺序、变量命名和误导性注释等因素影响；因此本项目不应把 judge preference 作为主指标。主指标应以执行正确性或任务定义的 exact recovery 为主，CodeBLEU/BLEU 只能作为辅助，compile/parse/signature/AST 等作为门控或安全性指标。

## 4. 论文必须进行的对比

### 4.1 专家层面对比

对每个专家至少报告：

- Base Qwen3-4B；
- 单任务 LoRA（当前四个专家）；
- 单一混合数据 Student SFT；
- 与任务相近的公开方法或数据构造路线（如 edit-aware repair、self-debug、identifier recovery）。

专家训练结果不能只展示训练 loss。应报告 held-out test 上的主指标、CodeBLEU/BLEU 或 F1、compile/parse/signature、样本数和失败案例。

### 4.2 Student 融合层面对比

建议至少包含以下方法：

1. **Student-only SFT**：四个领域训练数据直接混合；
2. **Equal-weight adapter merge**：将四个 LoRA adapter 等权合并；
3. **Static domain routing**：使用已知 domain label 选择对应专家；
4. **Oracle routing**：用真实最优 Teacher 作为上限；
5. **Raw MOPD**：不校准、直接使用 log-probability gap；
6. **Mean/std calibration**；
7. **Median/MAD calibration**；
8. **Open-MOPD-style**：top-k reward + token-share balancing + gap-following + reward refresh；
9. **去掉一个组件的消融**：无 token balance、无 gap following、无 refresh、无 routing margin。

其中 4 是上限，1 和 2 是最重要的廉价基线，5--8 用于证明方法组件确实有效。

### 4.3 评测层面对比

所有方法都应在相同 Student backbone、相同 rollout 数、相同 decoding、相同 max-new-tokens 和相同测试集上比较。按四个 domain 分别报告：

- task-specific primary metric；
- execution/test pass；
- compile/parse/signature；
- exact match 或 recovery rate；
- CodeBLEU/BLEU/F1；
- Levenshtein 或 edit distance；
- 失败类型：截断、语法错误、签名破坏、测试失败、过度编辑。

## 5. 对本项目的结论

四个专家当前的 held-out 结果可以作为论文的第一阶段证据，但不能单独支撑“多专家融合有效”。论文的关键实验应是：在独立 Student 数据集上，比较混合 SFT、adapter merge、raw MOPD、校准 MOPD 和 Open-MOPD-style training；同时验证 routing calibration 是否比 raw advantage 稳定。

特别需要避免以下表述：

- 不要说“median/MAD 是 ReCal 提出的”；它是本项目的校准扩展，ReCal 只能支持异构 reward 需要校准这一动机；
- 不要说“Open-MOPD 已经是成熟标准”；应写成近期方法基础，并通过消融验证；
- 不要把四个专家的提升直接等同于 Student 融合提升；专家结果与 Student 结果必须分开呈现；
- 不要用 BLEU 或 CodeBLEU 单独证明代码功能正确。

## 6. 建议的论文实验章节结构

1. **Data construction and task decomposition**：四类代码变换、数据来源、过滤和 problem-level split；
2. **Expert training**：Qwen3-4B、LoRA、assistant-only loss、最佳 checkpoint；
3. **Multi-teacher Student training**：same-rollout scoring、OPD gap、top-k reward、token balancing、calibration 和 routing；
4. **Baselines and ablations**：mixed SFT、adapter merge、raw/mean-std/median-MAD、Open-MOPD components；
5. **Results**：四域主指标、总体平均、hard gates、效率；
6. **Robustness and error analysis**：截断、语法错误、过度编辑、domain confusion；
7. **Limitations**：任务定义依赖、测试覆盖率、校准泛化、Teacher 计算开销。

## 7. 最低可发表实验清单

- 独立 Student 数据集，不能把四个专家训练样本直接当成融合训练和最终测试；
- 至少 3 个融合基线：mixed SFT、adapter merge、raw MOPD；
- 至少 3 个校准方案：raw、mean/std、median/MAD；
- Open-MOPD 组件消融；
- 每个 domain 的 paired win/loss/tie 和 bootstrap 95% CI；
- 训练和推理成本：Teacher 数量、显存、每样本 rollout 时间；
- 执行正确性和 hard-gate 失败案例；
- 公开配置、数据哈希、checkpoint 和评测脚本。

综上，现有研究的共同趋势是“可验证代码修改、多源/多任务训练、动态反馈和更可靠的评测”。本项目最有潜力的论文定位不是再次证明 LoRA 能提升单任务代码修复，而是证明“任务解耦专家 + 校准后的多教师 on-policy token distillation”能够在多个代码重写任务之间实现稳定迁移，并且这种提升在执行和结构约束下仍然成立。
