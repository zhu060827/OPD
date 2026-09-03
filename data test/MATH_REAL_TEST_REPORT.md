# Math 真实数据测试报告

本次测试使用主工程中的 `Math/math_reasoning_expander` 完整双层循环，
不是旧版单轮 fallback，也不再使用旧版 data test 外部替代评分。

## 测试配置

- 模型：`gpt-5.6-terra`
- 样本数：16
- 外层：最多 10 个遮盖点
- 内层：每个遮盖点最多反馈优化 3 次
- 保留规则：生成节点相对原节点 `gain > 0`
- API 状态：全部为真实 API 调用

## 总体结果

- 完成数：16
- 错误数：0
- 接受数：11
- 接受率：0.688
- 平均综合分：0.820
- 平均推理增益：0.759
- 平均新增节点数：3.125
- 平均尝试次数：12.250
- 平均单题耗时：91.624 秒
- 真实 API 调用：196
- mock 回退调用：0
- LaTeX 结构退化样本：4

## 五项平均分

| 指标 | 平均分 |
|---|---:|
| 去重复性 | 0.946 |
| 逻辑一致性 | 0.813 |
| 公式正确性 | 0.719 |
| 完整性 | 0.900 |
| 推理增益 | 0.759 |

## 分数据集结果

| 数据集 | 样本 | 接受率 | 综合分 | 推理增益 | 真实调用 | mock 回退 |
|---|---:|---:|---:|---:|---:|---:|
| MetaMathQA | 4 | 0.500 | 0.847 | 0.788 | 44 | 0 |
| NuminaMath-CoT | 12 | 0.750 | 0.811 | 0.750 | 152 | 0 |

## 结果解读与限制

- `accepted=true` 表示候选节点相对原遮盖节点产生正质量增益，不等于完成了形式化数学证明。
- 公式正确性是本次五项指标中相对较弱的一项，复杂符号推导仍需要更严格的 SymPy/规则验证。
- 个别原始 CoT 含有噪声或不规范 LaTeX；报告单独统计了扩充后结构变差的样本，没有把它们隐藏。
- 本次结果适合作为工程闭环和阶段性质量对比，不应表述为 100% 数学正确率。

## 样例

这里只截取前 3 条，完整过程见 `results/math_real_records.jsonl`。

### 样例 1

- 题目：Consider the terms of an arithmetic sequence: $-\frac{1}{3}, y+2, 4y, \ldots$. Solve for $y$.
- 是否保留：True
- 综合分：0.799
- 被遮盖节点：\[ y + 2 + \frac{1}{3} = 4y - y - 2 \] \[ \frac{7}{3} + 2 = 3y - y \] \[ y = \frac{13}{6} \] \[ (y + 2) - \left(-\frac{1}{3}\right) = 4y - (y+2) \]
- 恢复节点：\[ (y+2)+\frac{1}{3}=4y-y-2 \] \[ y+\frac{7}{3}=3y-2 \implies \frac{7}{3}+2=3y-y \] \[ y=\frac{13}{3}\cdot\frac{1}{2}=\frac{13}{6} \] \[ (y+2)-\left(-\frac13\right)=4y-(y+2) \]

### 样例 2

- 题目：Suppose that $g(x) = 5x - 3$. What is $g^{-1}(g^{-1}(14))$?
- 是否保留：True
- 综合分：0.801
- 被遮盖节点：\[ x = \frac{y + 3}{5} \]
- 恢复节点：\[ x=\frac{y+3}{5} \]

### 样例 3

- 题目：A farmer has a rectangular field with dimensions $3m+8$ and $m-3$ where $m$ is a positive integer. If the field has an area of 76 square meters, find the value of $m$.
- 是否保留：True
- 综合分：0.770
- 被遮盖节点：3m^2 - 9m + 8m - 24 = 76, Factoring the quadratic, we find: (3m+8)(m-3) = 76. (3m+8)(m-3)=76. \[ (3m+25)(m-4) = 0. \[
- 恢复节点：\[ (3m+8)(m-3)=3m^2-9m+8m-24=3m^2-m-24. Factoring the quadratic, we obtain: \[ (3m+8)(m-3)=76. \[ (3m+8)(m-3)=76. The area is the product of the two dimensions, so \[ (3m+25)(m-4)=0. Since the area of a rectangle is the product of its length and width, we equate the product of the given dimensions t
