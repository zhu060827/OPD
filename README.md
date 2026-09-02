# 大创结题版工程系统使用说明

本项目是“面向生成式模型的高质量数据特征扩充技术研究”的结题版工程化系统。
它不是另起炉灶，而是在原 OCTree 表格特征扩充基础上，继续扩展 Math 和 Code 两类数据。

## Code 五教师 Open-MOPD 正式框架

正式训练路径现在要求五个真实、互不相同的本地 Teacher checkpoint，并调用固定版本的
Open-MOPD `mt_opd.sh`。训练数据的 `domain` 字段采用
`cot/style/ast/variable/control_flow`，按标签 hard routing；token-share balancing、
gap-following allocation 和 reward refresh 会作为显式训练参数传入。当前未执行 GPU
训练，但不再使用单 Teacher `opd.sh` 冒充多教师训练。

- [English Stage-2 guide](docs/STAGE2_OPEN_MOPD_FIVE_TEACHER.md)
- [中文 Stage-2 说明](docs/STAGE2_OPEN_MOPD_FIVE_TEACHER.zh-CN.md)
- 配置模板：`configs/stage2_open_mopd_five_teacher.example.json`

`verl_example/opd.sh` 继续保留，仅作为单 Teacher 对照组。

## Code 多专家阶段一（可选路由消融）

这个模块是 Open-MOPD 之前的可选路由层：有 `method/domain/rewrite_method`
来源标签时直接 hard routing；只有无标签样本才让五个 Teacher 对同一条
completion 做对齐 OPD advantage 评分，经各 Teacher 独立校准后选 Top-1。
Top-1/Top-2 差距太小时默认拒绝伪标签。原先固定权重的 Reward+advantage
逻辑仍保留为 `heuristic_ablation`，不再是默认方案。

- [English design and run guide](docs/STAGE1_MULTI_EXPERT.md)
- [中文设计与运行说明](docs/STAGE1_MULTI_EXPERT.zh-CN.md)

本地无模型 smoke test：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_stage1_smoke.ps1
```

Mock 输出只用于验证工程闭环，文件中会明确标记 `formal_training_result=false`。

## 三类数据分别扩充什么

Tabular：扩充新特征列和特征生成代码。流程是 LLM 生成候选特征，沙盒检查，模型评估，再根据指标和决策树规则反馈下一轮。

Math：扩充数学 CoT 的中间推理步骤。流程是切分原始解答，构建 Reasoning Graph，遮盖中间节点，让 LLM 恢复，再做质量评估。数学模块已经接入队友的 `Math/math_reasoning_expander/`，支持多遮盖点和同一节点多轮反馈优化。

Code：现在支持两种模式。没有原始代码时，走“生成代码 -> 跑测试 -> 错误反馈 -> 修复”；有原始 code 和 reasoning 时，走队友的 rewrite 扩充流程，同步改写 reasoning 和 code，并做语义等价检查和质量评分。

## 安装依赖

在 VSCode 终端进入本目录：

```powershell
python -m pip install -r requirements.txt
```

## 启动网站

```powershell
python app.py
```

看到下面类似输出就说明后端已经启动：

```text
OCTree Flask backend is running.
Open http://127.0.0.1:5000 in your browser.
```

然后浏览器打开：

```text
http://127.0.0.1:5000
```

## 前后端关系

`app.py` 是后端，负责运行 Python 算法、保存 outputs、返回 JSON。

`index.html` 是前端，负责页面展示、按钮点击、上传文件、把后端返回的结果显示出来。

前端点击按钮后，会调用这些接口：

- `/api/run/tabular`
- `/api/run/math`
- `/api/run/code`
- `/api/run/all_demo`

所以前端同学主要改 `index.html` 的样式和展示逻辑，不需要动算法代码。

## outputs 结果文件

运行 demo 后会生成：

- `outputs/tabular_results.json`：表格特征扩充结果
- `outputs/tabular_augmented.csv`：增强后的表格
- `outputs/metrics.csv`：baseline / optimized / improvement 指标
- `outputs/math_expanded.jsonl`：数学推理链扩充结果
- `outputs/math_quality.svg`：数学质量图
- `outputs/code_repair_traces.jsonl`：代码生成、修复或 rewrite 扩充轨迹
- `outputs/code_quality.svg`：代码 rewrite 质量对比图

## Code 新增 rewrite 流程

队友的代码包已经接入到 `code_rewrite_feedback_expander/`。

如果输入里有：

```json
{
  "prompt": "...",
  "reasoning": ["..."],
  "code": "def ...",
  "tests": ["assert ..."]
}
```

就会走 rewrite 扩充：

```text
原始 code + reasoning
-> LLM Rewrite
-> 语义等价检查
-> 质量评分
-> 自然语言反馈
-> 再次 Rewrite
-> 有提升才保留
```

rewrite 策略包括：

- `cot`：让 reasoning 更清楚
- `style`：改善代码风格
- `ast`：做 AST 等价重构
- `variable`：替换更清晰的变量名
- `control_flow`：调整控制流

保留条件是：

```text
语义检查通过 + 质量分提升
```

语义检查包括 AST 解析、函数签名、危险调用过滤、单元测试、AST 相似度、CodeBLEU-like 相似度。

质量评分包括可读性、复杂度、长度平衡、多样性和代码风格。

## Math 完整迭代流程

队友的数学包保留在 `Math/math_reasoning_expander/`，主工程通过 `math_adapter.py` 调用，不需要单独启动 Math 目录。

```text
原始题目 + 原始 CoT
-> 切分推理节点并构建 Reasoning Graph
-> 外层选择一个中间节点
-> 内层最多反馈改进 3 次
-> 生成节点质量高于原节点时保留并合入 CoT
-> 继续选择下一个节点
-> 输出扩充结果、每轮轨迹和质量图
```

默认配置是外层最多 10 个遮盖点、内层每个点最多 3 次、连续 2 个节点没有提升就提前停止。可以在 `.env` 中调整：

```text
MATH_MAX_MASK_ROUNDS=10
MATH_MAX_REFINE_ROUNDS=3
MATH_PATIENCE=2
MATH_ACCEPT_THRESHOLD=0.80
```

真实 API 不可用时仍会自动使用 mock；完整数学包运行异常时会自动切回 `math_adapter.py` 的单轮兼容流程。

## 常见问题

1. `ModuleNotFoundError: No module named 'flask'`

说明依赖没装到当前 Python 环境，重新运行：

```powershell
python -m pip install -r requirements.txt
```

2. 网站打不开

先确认 `python app.py` 的终端没有关闭，然后打开：

```text
http://127.0.0.1:5000
```

3. LLM 调用失败

项目会自动使用 mock 兜底，所以 demo 仍然能跑。正式使用时可以在 `.env` 里配置：

```text
OPENAI_API_KEY=你的key
OPENAI_BASE_URL=你的base_url
OPENAI_MODEL=模型名
```

可以先复制 `.env.example` 并改名为 `.env`，再填写自己的配置。`.env` 已被 `.gitignore` 排除，不要把完整 API Key 发到群里、写进截图或放进压缩包。
