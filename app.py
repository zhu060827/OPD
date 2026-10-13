# app.py - 完整整合版（指标改为 F1、MSE、Log-Loss）

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, log_loss, mean_squared_error
import openai
import warnings
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import traceback
import os

warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# ==========================================
# 定义终端打印颜色
# ==========================================
COLOR_GREEN = '\033[92m'
COLOR_RED = '\033[91m'
COLOR_CYAN = '\033[96m'
COLOR_YELLOW = '\033[93m'
COLOR_RESET = '\033[0m'

# ==========================================
# API 配置
# ==========================================
API_KEY = "sk-1GiwWb8DQg3pHbGKF8F7B8D9EeA84723BbC425C9F614Ba4b"
BASE_URL = "https://api.gpt.ge/v1"

try:
    client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)
    print(f"{COLOR_GREEN}✅ API 客户端初始化成功！系统已连接真实大模型算力。{COLOR_RESET}")
    LLM_AVAILABLE = True
except Exception as e:
    print(f"{COLOR_RED}❌ API 配置出错: {e}{COLOR_RESET}")
    LLM_AVAILABLE = False


# ==========================================
# 辅助函数：JSON 序列化
# ==========================================
def make_json_serializable(obj):
    """将 numpy 类型转换为 Python 原生类型"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(item) for item in obj)
    else:
        return obj


# ==========================================
# 1. 数据预处理（带 LLM 语义理解）
# ==========================================
def get_data_with_llm(df, llm_client):
    """使用 LLM 理解数据，生成列名映射和领域知识"""

    df_clean = df.dropna().copy()
    print(f">>> ✅ 用户数据加载成功，共 {len(df_clean)} 行，{len(df_clean.columns)} 列")

    knowledge_base = "【领域知识库】: 用户提供的表格数据"

    if llm_client is not None and LLM_AVAILABLE:
        try:
            prompt = f"""
            你是一个数据分析专家。请分析以下数据集的结构和内容，并完成两项任务：

            数据集预览(前3行):
            {df_clean.head(3).to_string()}

            数据集信息:
            - 形状: {df_clean.shape}
            - 列名: {list(df_clean.columns)}
            - 数据类型: {df_clean.dtypes.to_dict()}

            任务:
            1. **列名重命名**: 如果列名是通用名（如'V1'、'col1'）或含义不清晰，请根据数据内容推断其物理/业务含义，并生成一个Python字典格式的列名映射。
               例如：{{'V1': 'fixed_acidity', 'V2': 'volatile_acidity', ...}}
               如果列名已清晰，请返回空字典{{}}。
            2. **领域知识库**: 用一段话描述这个数据集可能涉及的领域、关键特征的含义以及预测目标。

            **重要: 请严格按以下JSON格式返回，不要包含任何其他文本:**
            {{
                "column_mapping": {{"旧列名": "新列名", ...}},
                "knowledge": "领域知识描述..."
            }}
            """

            response = llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一个专业的数据科学家，擅长理解表格数据并生成清晰的元数据。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                timeout=15
            )

            raw_content = response.choices[0].message.content.strip()

            # 清理 Markdown 代码块
            if raw_content.startswith("```json"):
                raw_content = raw_content[7:]
            if raw_content.endswith("```"):
                raw_content = raw_content[:-3]
            raw_content = raw_content.strip()

            result = json.loads(raw_content)

            # 应用列名映射
            if result.get("column_mapping"):
                df_clean = df_clean.rename(columns=result["column_mapping"])
                print(f"✅ 已应用LLM生成的列名映射: {result['column_mapping']}")

            knowledge_base = f"【领域知识库 (LLM生成)】:\n{result.get('knowledge', '无')}"
            print(f"{COLOR_GREEN}✅ 已生成LLM领域知识库。{COLOR_RESET}")

        except Exception as e:
            print(f"⚠️ LLM处理出错: {e}")
            knowledge_base = "【领域知识库】: LLM处理出错，使用默认处理。"
    else:
        print("⚠️ LLM 不可用，跳过语义理解")

    # 识别目标列（使用最后一列）
    target_col = df_clean.columns[-1]
    print(f"✅ 识别目标列为: {target_col}")

    # 编码目标列
    le = LabelEncoder()
    if df_clean[target_col].dtype == 'object' or df_clean[target_col].dtype.name == 'category':
        df_clean['target'] = le.fit_transform(df_clean[target_col])
    else:
        df_clean['target'] = df_clean[target_col].values

    # 只保留数值列
    num_cols = df_clean.select_dtypes(include=['number']).columns
    df_numeric = df_clean[num_cols].copy()

    # 特征缩放
    feature_cols = [c for c in df_numeric.columns if c != 'target']
    if len(feature_cols) > 0:
        scaler = MinMaxScaler()
        df_numeric[feature_cols] = scaler.fit_transform(df_numeric[feature_cols])

    return df_numeric, knowledge_base, target_col


# ==========================================
# 2. 代码鲁棒性验证
# ==========================================
def check_feature_robustness(code, df_sample):
    try:
        local_vars = {'df': df_sample.copy(), 'np': np}
        exec(code, {}, local_vars)

        df_new = local_vars['df']
        if len(df_new.columns) <= len(df_sample.columns):
            return False, "代码执行成功但没有生成新列"

        new_col = df_new.iloc[:, -1]

        if new_col.isna().any():
            return False, "特征包含 NaN (空值)"
        if np.isinf(new_col).any():
            return False, "特征包含 Inf (无穷大)"
        if new_col.nunique() <= 1:
            return False, "特征值完全相同 (无区分度)"

        return True, "通过数学体检"

    except Exception as e:
        return False, f"代码报错: {e}"


# ==========================================
# 3. 评估与推理（五折交叉验证）- 只输出 F1、MSE、Log-Loss
# ==========================================
def evaluate_and_reason(df):
    X = df.drop(columns=['target'])
    y = df['target']

    # 检查样本数量，调整折数
    from collections import Counter
    class_counts = Counter(y)
    min_class_count = min(class_counts.values())
    n_splits = min(5, min_class_count)
    if n_splits < 2:
        n_splits = 2

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    f1_scores, logloss_scores, mse_scores = [], [], []

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0)

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)

        f1_scores.append(f1_score(y_val, preds, average='weighted'))
        logloss_scores.append(log_loss(y_val, probs))
        mse_scores.append(mean_squared_error(y_val, preds))

    # 决策树解释（用于反馈）
    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt.fit(X, y)
    reasoning = export_text(dt, feature_names=list(X.columns))

    return (np.mean(f1_scores), np.mean(logloss_scores), np.mean(mse_scores), reasoning)


# ==========================================
# 4. LLM 特征生成器（强制轮询策略）
# ==========================================
def get_strategy(round_idx):
    strategies = [
        "比率关系 (Ratio): 例如 A / (B + 0.01)",
        "非线性变换 (Non-linear): 例如 np.log(A + 1)",
        "复杂组合 (Complex): 例如 (A - B) / (C + 1)",
        "交互乘法 (Interaction): 例如 A * B"
    ]
    return strategies[(round_idx - 1) % len(strategies)]


def ask_llm_for_feature(columns, data_desc, history, knowledge, strategy):
    """调用 LLM 生成特征规则"""

    if not LLM_AVAILABLE:
        raise Exception("LLM 服务不可用，请检查 API 配置")

    system_prompt = "你是一个精通特征工程的数据科学家。请基于物理意义和数学逻辑构造新特征。"
    user_prompt = f"""
    【知识库】: {knowledge}
    【当前列名】: {columns}
    【统计】: {data_desc[:800]}
    【本轮强制策略】: 👉 {strategy}
    【历史反馈 (Decision Tree)】: {history[:500]}

    【要求】: 只返回一行 Python 代码。格式示例: df['new_feature'] = df['A'] * df['B']
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.8,
            timeout=20
        )
        code = response.choices[0].message.content.strip()
        code = code.replace("```python", "").replace("```", "").strip()
        print(f"{COLOR_GREEN}💡 LLM 提出公式: {code[:80]}...{COLOR_RESET}")
        return code
    except Exception as e:
        print(f"{COLOR_RED}❌ LLM 调用失败: {e}{COLOR_RESET}")
        raise Exception(f"LLM API 调用失败: {str(e)}")


# ==========================================
# 5. 核心 OCTree 分析
# ==========================================
def run_octree_analysis(df, max_iterations=5):
    """运行完整的 OCTree 分析"""

    results = {
        'baseline': {},
        'optimized': {},
        'iterations': [],
        'generated_rules': [],
        'improvement': {},
        'decision_tree_explanation': "",
        'knowledge_base': ""
    }

    # 数据预处理（带 LLM 语义理解）
    df_processed, knowledge_base, target_col = get_data_with_llm(df, client if LLM_AVAILABLE else None)
    results['knowledge_base'] = knowledge_base

    initial_feature_count = len(df_processed.columns) - 1
    print(f"✅ 预处理完成，特征数: {initial_feature_count}")

    # 基准评估（新指标：F1, LogLoss, MSE）
    base_f1, base_logloss, base_mse, base_reasoning = evaluate_and_reason(df_processed)

    results['baseline'] = {
        'f1': float(base_f1),
        'log_loss': float(base_logloss),
        'mse': float(base_mse)
    }
    results['decision_tree_explanation'] = base_reasoning

    print(f"🔥 初始性能指标 -> F1-Score: {base_f1:.4f} | Log-Loss: {base_logloss:.4f} | MSE: {base_mse:.4f}")

    current_df = df_processed.copy()
    current_f1, current_logloss, current_mse = base_f1, base_logloss, base_mse

    history_log = f"Round 0 F1:{base_f1:.4f}, LogLoss:{base_logloss:.4f}, MSE:{base_mse:.4f}\nReasoning:\n{base_reasoning}"

    # 迭代优化
    for i in range(1, max_iterations + 1):
        print(f"\n======== 第 {i} 轮深度特征寻优迭代 ========")
        strategy = get_strategy(i)
        print(f"🎯 本轮特征衍生策略: {strategy}")

        try:
            data_desc = current_df.describe().to_string()
            new_code = ask_llm_for_feature(
                list(current_df.columns), data_desc,
                history_log, knowledge_base, strategy
            )

            if not new_code:
                print("⚠️ 未生成有效代码")
                continue

            code = new_code.replace("```python", "").replace("```", "").strip()
            print(f"💡 LLM 提出公式: {code}")

            # 沙盒验证
            is_robust, msg = check_feature_robustness(code, current_df.head(50))

            if not is_robust:
                print(f"⛔ 沙盒拦截: {msg}")
                history_log += f"\nCode Error: {code} -> {msg}"
                results['iterations'].append({
                    'iteration': i,
                    'rule_code': code,
                    'accepted': False,
                    'reason': msg,
                    'f1_before': float(current_f1),
                    'f1_after': float(current_f1),
                    'mse_before': float(current_mse),
                    'mse_after': float(current_mse),
                    'logloss_before': float(current_logloss),
                    'logloss_after': float(current_logloss)
                })
                continue

            # 应用规则并评估
            df_temp = current_df.copy()
            exec(code, {'df': df_temp, 'np': np})

            new_f1, new_logloss, new_mse, new_reasoning = evaluate_and_reason(df_temp)
            print(f"[验证结果] F1: {new_f1:.4f} | Log-Loss: {new_logloss:.4f} | MSE: {new_mse:.4f}")

            # 判断是否接受（基于 F1 提升或 MSE 降低）
            f1_diff = new_f1 - current_f1
            mse_diff = current_mse - new_mse  # MSE 降低为正
            ll_diff = current_logloss - new_logloss

            is_accepted = False
            reason_msg = ""

            if f1_diff >= 0.005:
                is_accepted = True
                reason_msg = f"【F1 提升】F1-Score 提升了 {f1_diff:.4f}"
            elif mse_diff >= 0.005:
                is_accepted = True
                reason_msg = f"【MSE 降低】MSE 降低了 {mse_diff:.4f}"
            elif ll_diff >= 0.01:
                is_accepted = True
                reason_msg = f"【置信度增强】Log-Loss 改善了 {ll_diff:.4f}"
            else:
                if abs(f1_diff) <= 0.001 and abs(mse_diff) <= 0.001:
                    reason_msg = f"【结构冗余】指标波动极微"
                elif f1_diff < -0.005:
                    reason_msg = f"【性能下降】F1 降低了 {-f1_diff:.4f}"
                else:
                    reason_msg = f"【微小波动】指标变化不显著"

            if is_accepted:
                print(f"{COLOR_GREEN}🎉 进化成功！{COLOR_RESET}")
                print(f"👉 动态系统诊断：{reason_msg}")

                # 更新当前数据
                current_df = df_temp
                current_f1, current_logloss, current_mse = new_f1, new_logloss, new_mse
                history_log += f"\nSuccess: {code}\nReasoning: {new_reasoning}"

                results['generated_rules'].append({
                    'iteration': i,
                    'code': code,
                    'improvement': f"F1 +{f1_diff*100:.2f}%, MSE -{mse_diff*100:.2f}%",
                    'f1_gain': float(f1_diff),
                    'mse_reduction': float(mse_diff)
                })
            else:
                print(f"{COLOR_RED}❌ 进化停滞{COLOR_RESET}")
                print(f"👉 动态系统诊断：{reason_msg}")
                history_log += f"\nFailed: {code}"

            results['iterations'].append({
                'iteration': i,
                'rule_code': code,
                'accepted': is_accepted,
                'reason': reason_msg,
                'f1_before': float(current_f1 if not is_accepted else new_f1 - f1_diff),
                'f1_after': float(new_f1),
                'mse_before': float(current_mse if not is_accepted else new_mse + mse_diff),
                'mse_after': float(new_mse),
                'logloss_before': float(current_logloss if not is_accepted else new_logloss + ll_diff),
                'logloss_after': float(new_logloss),
                'improvement': float(f1_diff)
            })

        except Exception as e:
            print(f"⚠️ 迭代出错: {e}")
            results['iterations'].append({
                'iteration': i,
                'rule_code': "",
                'accepted': False,
                'reason': str(e),
                'f1_before': float(current_f1),
                'f1_after': float(current_f1),
                'mse_before': float(current_mse),
                'mse_after': float(current_mse),
                'logloss_before': float(current_logloss),
                'logloss_after': float(current_logloss)
            })

    # 最终结果
    results['optimized'] = {
        'f1': float(current_f1),
        'log_loss': float(current_logloss),
        'mse': float(current_mse)
    }

    results['improvement'] = {
        'f1': float((current_f1 - base_f1) / base_f1 * 100) if base_f1 > 0 else 0,
        'mse': float((base_mse - current_mse) / base_mse * 100) if base_mse > 0 else 0,
        'log_loss': float((base_logloss - current_logloss) / base_logloss * 100) if base_logloss > 0 else 0,
        'features_added': len(results['generated_rules'])
    }

    print(f"\n🎉 分析完成！F1: {base_f1:.4f} → {current_f1:.4f} (+{results['improvement']['f1']:.1f}%)")
    print(f"   MSE: {base_mse:.4f} → {current_mse:.4f} ({results['improvement']['mse']:.1f}% 降低)")

    return results


# ==========================================
# Flask API 路由
# ==========================================

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('.', path)


@app.route('/api/upload', methods=['POST'])
def upload_and_analyze():
    """上传数据并运行 OCTree 分析"""
    print("=" * 60)
    print("📤 收到上传请求！")

    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': '请选择文件'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': '文件名为空'}), 400

        # 读取文件
        filename = file.filename
        if filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            return jsonify({'success': False, 'error': '请上传 CSV 或 Excel 文件'}), 400

        print(f"✅ 文件: {filename}")
        print(f"📊 数据: {df.shape[0]} 行, {df.shape[1]} 列")

        # 运行完整 OCTree 分析
        results = run_octree_analysis(df, max_iterations=5)

        # 确保 JSON 可序列化
        serializable_results = make_json_serializable(results)

        response_data = {
            'success': True,
            'baseline': serializable_results['baseline'],
            'optimized': serializable_results['optimized'],
            'improvement': serializable_results['improvement'],
            'generated_rules': serializable_results['generated_rules'],
            'iterations': serializable_results['iterations'],
            'decision_tree_explanation': serializable_results.get('decision_tree_explanation', ''),
            'knowledge_base': serializable_results.get('knowledge_base', ''),
            'message': f"✅ 分析完成！新增 {serializable_results['improvement']['features_added']} 个特征，F1 提升 {serializable_results['improvement']['f1']:.1f}%，MSE 降低 {serializable_results['improvement']['mse']:.1f}%"
        }

        print(f"✅ 分析完成，返回结果")
        return jsonify(response_data)

    except Exception as e:
        error_msg = str(e)
        print(f"❌ 错误: {error_msg}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': error_msg}), 500


@app.route('/api/simulate', methods=['POST'])
def simulate_iteration():
    """模拟单次迭代"""
    data = request.json
    iteration = data.get('iteration', 1)

    return jsonify({
        'iter': iteration,
        'rule': f"df['feature_{iteration}'] = df['col1'] / (df['col2'] + 0.01)",
        'desc': f'第{iteration}轮生成的组合特征',
        'f1_before': round(0.70 + iteration * 0.005, 3),
        'f1_after': round(0.70 + iteration * 0.01, 3),
        'improved': True,
        'reason': '特征有效，已合入特征池'
    })


if __name__ == '__main__':
    print("=" * 60)
    print(f"{COLOR_CYAN}🚀 OCTree 完整版后端启动成功！{COLOR_RESET}")
    print(f"{COLOR_GREEN}🌐 本地访问: http://127.0.0.1:5000{COLOR_RESET}")
    print(f"{COLOR_GREEN}✅ LLM 状态: {'可用（真实 API）' if LLM_AVAILABLE else '不可用'}{COLOR_RESET}")
    print(f"{COLOR_GREEN}📊 评估指标: F1-Score, Log-Loss, MSE{COLOR_RESET}")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)