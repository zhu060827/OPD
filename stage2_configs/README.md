# Stage 2 配置与代码集合

## 本文件夹包含

### 数据读取适配
- `io_utils.py`：修复 MBPP 格式（`text` → `prompt`，`test_list` → `tests`）

### Stage 1 配置
- `stage1_multi_expert.heuristic.json`：Stage 1 路由配置

### Stage 2 配置
- `stage2_open_mopd_from_stage1.json`：Stage 2 五教师训练配置

### 核心模块
- `cli.py`：支持分批写入 + 断点续跑
- `mbpp_workflow.py`：MBPP 数据划分
- `stage2_open_mopd.py`：Stage 2 训练入口
- `config.py`、`router.py`、`models.py`、`pipeline.py`：Stage 1 核心

### 脚本
- `convert_routing_to_handoff.py`：routing → handoff 转换
- `prepare_mbpp_workflow.py`：按 train/test/val/prompt 拆分
- `run_stage2_open_mopd.sh`：Stage 2 启动入口

## 验证状态

- ✅ 3 条数据测试通过
- ✅ prompt 和 tests 字段正确读取
- ✅ verification_status 正确计算
