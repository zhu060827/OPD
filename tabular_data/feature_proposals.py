"""基于训练集画像提出可解释的多列、非线性和分段特征假说。"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
import hashlib
import itertools
import json
import os
from pathlib import Path
import time
import re

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from llm_client import LLMClient, LLMError, strip_code_fence
from tabular_data.feature_explanations import (
    derivation_steps,
    explain_expression,
    number,
    parameter_registry,
    quantile_details,
    symbol_name,
)
from tabular_data.feature_feedback import (
    DEFAULT_CANDIDATES,
    candidate_priority,
    feedback_summary,
    prior_feedback,
)


GENERATOR_VERSION = "cart_residual_v6"
FAMILIES = ("multicolumn", "nonlinear", "piecewise")
NP_FUNCTIONS = {
    "abs",
    "sign",
    "log",
    "log1p",
    "log10",
    "sqrt",
    "exp",
    "clip",
    "maximum",
    "minimum",
    "where",
    "sin",
    "cos",
    "tanh",
    "power",
}


@dataclass
class FeatureProposal:
    name: str
    family: str
    input_columns: list[str]
    expression: str
    hypothesis: str
    construction: str
    evidence: dict
    parameters: dict = field(default_factory=dict)
    source: str = "local_template"
    display_expression: str = ""
    constant_provenance: list = field(default_factory=list)
    derivation: list = field(default_factory=list)
    adaptation: dict = field(default_factory=dict)

    @property
    def code(self):
        return f"df[{self.name!r}] = {self.expression}"

    @property
    def signature(self):
        return hashlib.sha256(
            ast.dump(ast.parse(self.expression, mode="eval")).encode()
        ).hexdigest()

    def to_dict(self):
        return {
            **asdict(self),
            "code": self.code,
            "signature": self.signature,
            "evidence_split": "train",
            "interpretation": "待验证的特征关系假说，不代表因果或已证实规律",
        }


def validate_expression(expression, allowed_columns):
    """仅接受有限的逐行表达式，统计量必须事先在训练集计算并写成常数。"""
    tree = ast.parse(expression, mode="eval")
    nodes = list(ast.walk(tree))
    if len(nodes) > 240:
        raise ValueError("表达式过长，无法保持简洁可解释。")
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Compare,
        ast.Call,
        ast.keyword,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Subscript,
        ast.Attribute,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.BitAnd,
        ast.BitOr,
        ast.Invert,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.Eq,
        ast.NotEq,
    )
    inputs = []
    for node in nodes:
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"不允许的表达式节点：{type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in {"df", "np"}:
            raise ValueError(f"未知对象：{node.id}")
        if isinstance(node, ast.Attribute):
            if (
                not isinstance(node.value, ast.Name)
                or node.value.id != "np"
                or node.attr not in NP_FUNCTIONS | {"pi"}
            ):
                raise ValueError("只能调用指定的 NumPy 逐行函数。")
        if isinstance(node, ast.Call):
            if (
                not isinstance(node.func, ast.Attribute)
                or node.func.attr not in NP_FUNCTIONS
            ):
                raise ValueError("禁止拟合统计量、动态调用或跨行变换。")
        if isinstance(node, ast.Subscript):
            if (
                not isinstance(node.value, ast.Name)
                or node.value.id != "df"
                or not isinstance(node.slice, ast.Constant)
                or node.slice.value not in allowed_columns
            ):
                raise ValueError(
                    "只能读取许可列表中的单个特征列，不能访问标签或行位置。"
                )
            if node.slice.value not in inputs:
                inputs.append(node.slice.value)
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not np.isfinite(node.value)
        ):
            raise ValueError("表达式常数必须有限。")
    if not inputs or len(inputs) > 6:
        raise ValueError("每个特征必须使用 1 到 6 个输入列。")
    # 使用源码出现顺序，避免 AST 广度遍历把“第一列/第二列”的说明顺序打乱。
    references = sorted(
        (node for node in nodes if isinstance(node, ast.Subscript)),
        key=lambda node: (node.lineno, node.col_offset),
    )
    return list(dict.fromkeys(node.slice.value for node in references))


def training_profile(frame, columns, task_type="classification"):
    columns = [col for col in columns if col != "target" and frame[col].nunique() > 1]
    if not columns:
        raise ValueError("没有可用于生成特征的非恒定数值列。")
    x = frame[columns].astype(float)
    relevance = np.zeros(len(columns))
    if "target" in frame and len(frame) >= 10:
        estimator = (
            mutual_info_classif
            if task_type == "classification"
            else mutual_info_regression
        )
        relevance = estimator(
            x, frame.target, random_state=42, n_neighbors=min(3, len(frame) - 1)
        )
    records = {}
    for col, information in zip(columns, relevance):
        values = x[col]
        q25, q50, q75 = values.quantile([0.25, 0.5, 0.75]).tolist()
        scale = max(q75 - q25, float(values.std(ddof=0)), 1e-6)
        skew = float(values.skew())
        records[col] = {
            "mutual_information_with_target": float(information),
            "median": q50,
            "q25": q25,
            "q75": q75,
            "scale": scale,
            "iqr": q75 - q25,
            "std": float(values.std(ddof=0)),
            "sum": float(values.sum()),
            "mean": float(values.mean()),
            "squared_deviations": float(((values - values.mean()) ** 2).sum()),
            "quantile_details": {
                key: quantile_details(values, p)
                for key, p in (("q25", 0.25), ("median", 0.5), ("q75", 0.75))
            },
            "skew": skew if np.isfinite(skew) else 0.0,
        }
    ranked = sorted(
        columns,
        key=lambda col: (
            -records[col]["mutual_information_with_target"],
            columns.index(col),
        ),
    )
    return {
        "rows": len(frame),
        "ranked_columns": ranked,
        "columns": records,
        "spearman": x.corr(method="spearman").fillna(0).to_dict(),
        "fit_split": "train",
    }


def _proposal(
    name,
    family,
    expression,
    hypothesis,
    construction,
    profile,
    parameters=None,
    source="local_template",
):
    expression, display_expression, constants, used = explain_expression(
        expression, profile
    )
    inputs = validate_expression(expression, set(profile["columns"]))
    correlations = {
        f"{a} / {b}": profile["spearman"][a][b]
        for a, b in itertools.combinations(inputs, 2)
    }
    return FeatureProposal(
        name,
        family,
        inputs,
        expression,
        hypothesis,
        construction,
        {
            "training_rows": profile["rows"],
            "column_statistics": {c: profile["columns"][c] for c in inputs},
            "pairwise_spearman": correlations,
            "selection_basis": "训练集列与标签的互信息用于候选排序；相关性和分位数仅描述样本关系。",
        },
        {**(parameters or {}), **{key: value["value"] for key, value in used.items()}},
        source,
        display_expression,
        constants,
        derivation_steps(display_expression, construction),
    )


def _domain_candidates(profile):
    columns = set(profile["columns"])
    specs = []
    chess = {
        f"{color}_piece0_{field}"
        for color in ("white", "black")
        for field in ("strength", "file", "rank")
    }
    if chess <= columns:
        dx = "np.abs(df['white_piece0_file'] - df['black_piece0_file'])"
        dy = "np.abs(df['white_piece0_rank'] - df['black_piece0_rank'])"
        gap = "(df['white_piece0_strength'] - df['black_piece0_strength'])"
        distance = f"({dx} + {dy})"
        specs.extend(
            [
                (
                    "chess_manhattan",
                    "multicolumn",
                    distance,
                    "双方棋子相对距离可能比各自绝对坐标更容易表达接触关系。",
                    "横向坐标差和纵向坐标差各取绝对值后相加；这是网格距离代理，未计入河流、陷阱或行棋规则。",
                    {},
                ),
                (
                    "chess_strength_proximity",
                    "multicolumn",
                    f"{gap} / (1 + {distance})",
                    "棋子强度差与相互距离可能共同影响残局结果；强度顺序存在特殊规则，不能当成确定胜负公式。",
                    "计算双方 strength 差，再除以 1 加曼哈顿距离；距离增加时平滑衰减，用到六个原始列。",
                    {},
                ),
                (
                    "chess_alignment",
                    "piecewise",
                    f"np.where(({dx} == 0) | ({dy} == 0), {gap}, 0)",
                    "同列或同行的棋子可能呈现不同的直接交互关系，需要验证。",
                    "同 file 或同 rank 时输出强度差，否则输出 0；0 是坐标相等的比较基准及不满足条件的输出。",
                    {},
                ),
            ]
        )
    return [
        _proposal(
            "oct_" + name, family, expression, hypothesis, construction, profile, params
        )
        for name, family, expression, hypothesis, construction, params in specs
    ]


def _z(column):
    return f"((df[{column!r}] - {symbol_name(column, 'median')}) / {symbol_name(column, 'scale')})"


def _repair_candidates(profile, records):
    """根据具体失败类型改变表达式，而不只是把失败文本附在同一公式上。"""
    repairs = []
    rejected = [
        row
        for row in reversed(records)
        if row["outcome"]
        in {"invalid", "unused", "metric_regression", "insufficient_gain"}
    ]
    for row in rejected[:10]:
        columns = [c for c in row["input_columns"] if c in profile["columns"]]
        if len(columns) < 2:
            continue
        a, b = columns[:2]
        alternative = next(
            (c for c in profile["ranked_columns"] if c not in columns), b
        )
        za, zb, zc = (_z(a), _z(b), _z(alternative))
        outcome = row["outcome"]
        if outcome == "unused":
            family, expression = ("nonlinear", f"{za} * {zc}")
            action = f"移除多层调制项，保留两列标准化乘积；把原组合中的第二列换为 {alternative}。"
        elif outcome == "metric_regression":
            family, expression = ("nonlinear", f"np.tanh({za}) * np.tanh({zb})")
            action = "将两列分别压缩到 (−1,1) 后相乘，限制极端值；检验退化是否与不稳定幅度有关。此原因尚未证实。"
        elif outcome == "invalid":
            family, expression = ("nonlinear", f"np.tanh({za}) / (1 + np.abs({zb}))")
            action = "改用有界分子及至少为 1 的分母，减少溢出/除零；常数或重复问题仍需重新检查。"
        else:
            family = "piecewise"
            expression = f"np.where(df[{alternative!r}] <= {symbol_name(alternative, 'q25')}, {za} * {zb}, {za} / (1 + np.abs({zb})))"
            action = f"以不同列 {alternative} 的 Q25 设置门槛，检验特定子群的交互是否被全局组合掩盖。"
        proposal = _proposal(
            f"oct_repair_{len(repairs)}",
            family,
            expression,
            "依据上一轮验证反馈提出改进假说，仍需独立验证，不保证指标提升。",
            action,
            profile,
        )
        proposal.adaptation = {
            "kind": "targeted_repair",
            "source_round": row["round"],
            "source_candidate": row["name"],
            "outcome": outcome,
            "observed_reason": row["reason"],
            "degraded_metrics": row["degraded_metrics"],
            "action": action,
        }
        repairs.append(proposal)
        if row["family"] == "piecewise":
            tree = ast.parse(row["display_expression"], mode="eval")
            threshold = next(
                (
                    n
                    for n in ast.walk(tree)
                    if isinstance(n, ast.Name)
                    and n.id.startswith(("median_", "q25_", "q75_"))
                ),
                None,
            )
            if threshold:
                stat, col = threshold.id.split("_", 1)
                for replacement in ("q25", "q75"):
                    if replacement == stat or col not in profile["columns"]:
                        continue
                    expression = re.sub(
                        "\\b" + re.escape(threshold.id) + "\\b",
                        symbol_name(col, replacement),
                        row["display_expression"],
                    )
                    symbols = {
                        node.id
                        for node in ast.walk(ast.parse(expression, mode="eval"))
                        if isinstance(node, ast.Name) and node.id not in {"df", "np"}
                    }
                    if not symbols <= parameter_registry(profile).keys():
                        continue
                    proposal = _proposal(
                        f"oct_regate_{len(repairs)}",
                        "piecewise",
                        expression,
                        "旧分段未通过，检验另一预先规定分位点能否更好地区分条件关系。",
                        f"保留原分段函数，只将 {col} 的门槛从 {stat} 改为训练集 {replacement}；没有扫描测试集。",
                        profile,
                    )
                    proposal.adaptation = {
                        **repairs[-1].adaptation,
                        "action": proposal.construction,
                    }
                    repairs.append(proposal)
    return repairs


def local_candidates(profile, count, round_index, seen, records=None):
    records = records or []
    ranked = profile["ranked_columns"][:8]
    pool = _domain_candidates(profile)
    from .model_guidance import tree_specs

    for name, family, expression, hypothesis, construction, path in tree_specs(
        profile["model_guidance"]
    ):
        proposal = _proposal(
            name,
            family,
            expression,
            hypothesis,
            construction,
            profile,
            source="cart_rule",
        )
        proposal.evidence["cart_path"] = path
        pool.append(proposal)
    for index, (a, b) in enumerate(itertools.combinations(ranked, 2)):
        for operation, expression in (
            ("product", f"{_z(a)} * {_z(b)}"),
            ("ratio", f"df[{a!r}] / (np.abs(df[{b!r}]) + 1e-6)"),
            ("difference", f"df[{a!r}] - df[{b!r}]"),
        ):
            pool.append(
                _proposal(
                    f"oct_simple_{operation}_{index}",
                    "nonlinear",
                    expression,
                    "检验简单交互能否解释当前模型尚未捕获的关系；与复杂候选一起预筛。",
                    "按显示公式计算两列的乘积、比例或差；比值分母加固定 ε 防止除零。",
                    profile,
                )
            )
    tuples = list(itertools.combinations(ranked, min(3, len(ranked))))
    for index, group in enumerate(tuples):
        a = group[0]
        b = group[1] if len(group) > 1 else a
        c = group[2] if len(group) > 2 else b
        z = {col: _z(col) for col in group}
        params = {
            col: {
                "center": profile["columns"][col]["median"],
                "scale": profile["columns"][col]["scale"],
            }
            for col in group
        }
        if len(group) >= 3:
            pool.append(
                _proposal(
                    f"oct_composite_{index}",
                    "multicolumn",
                    f"{z[a]} * {z[b]} / (1 + np.abs({z[c]}))",
                    f"{a} 与 {b} 的联合状态可能受到 {c} 的调制，第三列以饱和倒数改变交互强度。",
                    f"只用训练集中心和尺度无量纲化，再计算 z({a})×z({b})/(1+|z({c})|)。",
                    profile,
                    params,
                )
            )
            pool.append(
                _proposal(
                    f"oct_composite_sum_{index}",
                    "multicolumn",
                    f"({z[a]} + {z[b]}) / (1 + np.abs({z[c]}))",
                    "前两列的联合水平可能比乘性交互稳定，第三列调节其幅度。",
                    "标准化后前两列相加，按第三列的绝对偏离程度作有界衰减；比较加性与乘性交互。",
                    profile,
                    params,
                )
            )
            pool.append(
                _proposal(
                    f"oct_log_balance_{index}",
                    "nonlinear",
                    f"np.log1p(np.abs({z[a]})) - np.log1p(np.abs({z[b]})) * np.tanh({z[c]})",
                    "前两列偏离基准的对数差异，可能随第三列所处区间改变。",
                    "前两列取绝对标准化偏离并 log1p 压缩，再用第三列的 tanh 调节第二项后求差。",
                    profile,
                    params,
                )
            )
        pool.append(
            _proposal(
                f"oct_signed_log_{index}",
                "nonlinear",
                f"np.sign({z[a]}) * np.log1p(np.abs({z[a]})) * np.tanh({z[b]}) / (1 + np.abs({z[c]}))",
                "重尾、饱和和反比例关系可能共同存在，需验证非线性复合是否改善分类。",
                f"{a} 使用有符号 log1p，{b} 使用 tanh，{c} 使用稳定倒数，再相乘。",
                profile,
                params,
            )
        )
        for stat in ("median", "q25", "q75"):
            threshold = symbol_name(a, stat)
            pool.append(
                _proposal(
                    f"oct_piecewise_{index}_{stat}",
                    "piecewise",
                    f"np.where(df[{a!r}] <= {threshold}, {z[b]} * {z[c]}, {z[b]} / (1 + np.abs({z[c]})))",
                    "第三列可能在不同条件区间表现为增强或抑制关系，分段特征将该假说交给验证集。",
                    f"用 {a} 的训练集 {stat} 作门槛，低区间计算 z({b})×z({c})，高区间计算 z({b})/(1+|z({c})|)。",
                    profile,
                    params,
                )
            )
    from .model_guidance import screen_candidates

    pool.extend(_repair_candidates(profile, records))
    pool = screen_candidates(
        pool, profile["_screen_frame"], profile["_guidance_context"]
    )
    chosen, signatures, selected_values = ([], set(seen), [])
    summary = feedback_summary(records)

    def take(proposal):
        if proposal.signature in signatures:
            return
        screen = proposal.evidence["prescreen"]
        values = proposal.evidence.get("_screen_values")
        if not screen["valid"] or screen["max_abs_spearman_existing"] >= 1 - 1e-10:
            return
        if any(
            (
                abs(float(np.corrcoef(values, previous)[0, 1])) > 0.98
                for previous in selected_values
            )
        ):
            return
        selected_values.append(values)
        proposal.evidence["prescreen"].update(
            {
                "pool_size": len(pool),
                "within_round_abs_pearson_limit": 0.98,
                "existing_monotonic_duplicate_limit": 1 - 1e-10,
            }
        )
        proposal.name = f"{proposal.name}_r{round_index}"
        if not proposal.adaptation:
            related = sorted(
                records,
                key=lambda r: len(
                    set(r["input_columns"]) & set(proposal.input_columns)
                ),
                reverse=True,
            )[:3]
            proposal.adaptation = {
                "kind": "feedback_ranked" if records else "initial_exploration",
                "action": "根据历史接受/拒绝类型调整候选优先级，并保留不同家族的探索。"
                if records
                else "首轮依据训练统计与领域公式探索。",
                "related_candidates": [
                    {k: r[k] for k in ("round", "name", "outcome", "reason")}
                    for r in related
                ],
            }
        proposal.adaptation.update(
            {
                "history_summary": summary,
                "priority_score": candidate_priority(proposal, records),
            }
        )
        chosen.append(proposal)
        signatures.add(proposal.signature)

    pool.sort(
        key=lambda p: (
            p.evidence["prescreen"]["score"],
            candidate_priority(p, records),
        ),
        reverse=True,
    )
    for proposal in (p for p in pool if p.source == "cart_rule"):
        take(proposal)
        if chosen:
            break
    families = FAMILIES
    for family in families:
        if len(chosen) >= count:
            break
        if family not in {p.family for p in chosen}:
            for match in (
                p for p in pool if p.family == family and p.signature not in signatures
            ):
                before = len(chosen)
                take(match)
                if len(chosen) > before:
                    break
    for proposal in pool:
        if len(chosen) >= count:
            break
        take(proposal)
    for proposal in pool:
        proposal.evidence.pop("_screen_values", None)
    return chosen


def propose_feature_candidates(
    frame,
    client,
    round_index,
    history,
    seen,
    *,
    count=DEFAULT_CANDIDATES,
    mode="reasoned",
    eligible_columns=None,
    task_type="classification",
    guidance=None,
    _response_retries=5,
    _response_feedback="",
    _partial_proposals=None,
):
    if mode != "reasoned":
        raise ValueError("只支持当前 reasoned 生成器。")
    columns = (
        eligible_columns
        if eligible_columns is not None
        else [col for col in frame if col != "target"]
    )
    profile = training_profile(frame, columns, task_type)
    if guidance is None:
        raise ValueError("reasoned 模式需要当前训练模型的 CART 与训练残差上下文。")
    from .model_guidance import public_guidance

    profile["model_guidance"] = public_guidance(guidance)
    profile["_extra_parameters"] = guidance["thresholds"]
    profile["_screen_frame"] = frame
    profile["_guidance_context"] = guidance
    records = prior_feedback(history, round_index)
    if isinstance(client, LLMClient):
        proposed = list(_partial_proposals or [])
        remaining = count - len(proposed)
        try:
            if not client.available:
                raise LLMError("LLM configuration is incomplete.")
            prompt = (
                f"""Propose {remaining} distinct interpretable numeric features as JSON: {{"features": [...]}}. Each item needs name (ASCII identifier), family, expression, hypothesis and construction. The hypothesis and construction must be concise Chinese explanations, not claims of proven effects. Expressions use df['column'] and only np.abs/sign/log/log1p/log10/sqrt/exp/clip/maximum/minimum/where/sin/cos/tanh/power and np.pi. No imports, target access, indexing rows, aggregations or fitted statistics. Refer to named training parameter symbols, never paste their numeric values. Allowed literal constants: 0, 1, 2, 4, 180, 1e-6 and np.pi, only with their mathematical/numerical meanings. Explain the mathematical derivation and purpose of each operation; do not provide private internal reasoning. Learn from ALL earlier validation outcomes: unused means simplify/change inputs; metric_regression means bound extremes or change gates; insufficient_gain means explore a new combination. Do not relax acceptance rules. When feedback exists, each feature also needs feedback_reference (an earlier candidate name) and adaptation_action (specific change in Chinese). Protect logs, roots, division and exponentials. Avoid huge expressions and redundant monotonic transforms alone. """
                + "Cover multicolumn (3+ columns), nonlinear and piecewise families. "
                + f"The family field MUST be exactly one of {json.dumps(FAMILIES)}. hypothesis and construction must be nonempty strings. "
                + "For EVERY multicolumn item, its expression MUST reference at least THREE DISTINCT df columns. A two-column ratio/product/log relation must use nonlinear, never multicolumn. A conditional np.where relation should use piecewise. Validate each item before responding. "
                + f"Do not repeat these already retained expressions: {[p.expression for p in proposed]}. Also do not repeat any prior history expression. Return exactly {remaining} items.\n"
                + f"Previous response validation error (repair it): {_response_feedback or 'None'}.\n"
                + f"Training parameters: {json.dumps({k: {'value': v['value'], 'column': v['column'], 'statistic': v['statistic']} for k, v in parameter_registry(profile).items()}, ensure_ascii=False)}\n"
                + f"Training profile: {json.dumps({k: v for k, v in profile.items() if not k.startswith('_')}, ensure_ascii=False)}\n"
                + f"Prior validation feedback: {json.dumps(records, ensure_ascii=False)}\n"
                + f"Already tried expressions: {sorted(seen)} (signature hashes; do not repeat prior feedback expressions).\n"
            )
            raw = client.chat(
                prompt,
                system_prompt="Return only the requested JSON feature proposals.",
                temperature=0.3,
            )
            client.real_call_count += 1
            audit_root = os.getenv("TABULAR_LLM_AUDIT_DIR")
            if audit_root:
                destination = Path(audit_root)
                destination.mkdir(parents=True, exist_ok=True)
                (destination / f"round_{round_index}_{time.time_ns()}.json").write_text(
                    json.dumps({"model": client.model, "round": round_index,
                                "prompt": prompt, "response": raw}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            payload = json.loads(strip_code_fence(raw))
            response_errors = []
            for item in payload["features"][:remaining]:
                try:
                    if not re.fullmatch("[A-Za-z][A-Za-z0-9_]{0,60}", item["name"]):
                        raise ValueError("特征名称不合法。")
                    proposal = _proposal(
                        f"oct_llm_{item['name']}_r{round_index}",
                        item["family"],
                        item["expression"],
                        item["hypothesis"],
                        item["construction"],
                        profile,
                        source="llm",
                    )
                    if records:
                        reference = next(
                            (
                                r
                                for r in records
                                if r["name"] == item.get("feedback_reference")
                            ),
                            None,
                        )
                        if (
                            reference is None
                            or not isinstance(item.get("adaptation_action"), str)
                            or (not item["adaptation_action"].strip())
                        ):
                            raise ValueError("LLM 未提供可核验的历史反馈引用与修改说明。")
                        proposal.adaptation = {
                            "kind": "llm_feedback",
                            "source_candidate": reference["name"],
                            "source_round": reference["round"],
                            "outcome": reference["outcome"],
                            "observed_reason": reference["reason"],
                            "action": item["adaptation_action"],
                            "history_summary": feedback_summary(records),
                        }
                    else:
                        proposal.adaptation = {
                            "kind": "initial_exploration",
                            "action": "首轮基于训练画像提出假说。",
                        }
                    allowed_families = FAMILIES
                    if (
                        proposal.family not in allowed_families
                        or not proposal.hypothesis
                        or (not proposal.construction)
                    ):
                        raise ValueError("候选家族或解释不符合当前实验设置。")
                    if proposal.family == "multicolumn" and len(proposal.input_columns) < 3:
                        raise ValueError(f"Candidate {item['name']} uses {len(proposal.input_columns)} distinct columns but family=multicolumn. Use nonlinear/piecewise for two-column relations, or genuinely reference at least three distinct df columns.")
                    if proposal.signature not in seen and proposal.signature not in {
                        p.signature for p in proposed
                    }:
                        proposed.append(proposal)
                except (ValueError, KeyError, TypeError, SyntaxError) as item_error:
                    response_errors.append(f"{type(item_error).__name__}: {item_error}")
            if len(proposed) != count:
                raise ValueError(f"Only {len(proposed)}/{count} distinct candidates retained. Some expressions repeat prior history or another item. Supply NEW expressions to complete the budget. Validation errors: {response_errors}")
            client.last_call_used_mock = False
            return proposed
        except (LLMError, ValueError, KeyError, TypeError, SyntaxError) as exc:
            client.last_error = (
                f"{type(exc).__name__}: structured feature proposal failed"
            )
            if not isinstance(exc, LLMError) and _response_retries > 0:
                print(f"LLM 候选格式校验失败，重新请求修复；剩余重试 {_response_retries} 次。", flush=True)
                return propose_feature_candidates(
                    frame, client, round_index, history, seen, count=count, mode=mode,
                    eligible_columns=eligible_columns, task_type=task_type, guidance=guidance,
                    _response_retries=_response_retries - 1,
                    _response_feedback=f"{type(exc).__name__}: {exc}",
                    _partial_proposals=proposed,
                )
            if not client.use_mock_when_fails:
                raise
            client.mock_fallback_count += 1
            client.last_call_used_mock = True
    else:
        client.mock_fallback_count += 1
    return local_candidates(profile, count, round_index, seen, records)


def print_proposal(proposal, round_index, candidate_index):
    print(
        f"  第 {round_index} 轮候选 {candidate_index} [{proposal.family}/{proposal.source}] {proposal.name}",
        flush=True,
    )
    print(f"    依据列：{', '.join(proposal.input_columns)}", flush=True)
    evidence = proposal.evidence["column_statistics"]
    print(
        "    训练集依据："
        + "；".join(
            f"{c}: MI={number(v['mutual_information_with_target'])}, 中位数={number(v['median'])}, 偏度={number(v['skew'])}"
            for c, v in evidence.items()
        ),
        flush=True,
    )
    correlations = proposal.evidence.get("pairwise_spearman", {})
    screening = proposal.evidence.get("prescreen")
    if screening:
        print(
            f"    模型误差预筛：{screening['formula']}；分子={number(screening['between_bin_residual_sum_squares'])}，"
            f"分母={number(screening['total_residual_sum_squares'])}，score={number(screening['score'])}，"
            f"分组数={screening['bins']}；仅训练内启发式，仍须通过验证。",
            flush=True,
        )
    if correlations:
        print(
            "    列间 Spearman："
            + "；".join(
                f"{pair}={number(value)}" for pair, value in correlations.items()
            ),
            flush=True,
        )
    adaptation = proposal.adaptation
    print(f"    历史反馈调整：{adaptation.get('action', '首轮探索')}", flush=True)
    if adaptation.get("source_candidate"):
        print(
            f"    来自第 {adaptation['source_round']} 轮 {adaptation['source_candidate']}：{adaptation['outcome']}；"
            f"{adaptation['observed_reason'].splitlines()[-1]}",
            flush=True,
        )
    elif adaptation.get("related_candidates"):
        print(
            "    参考历史："
            + "；".join(
                f"第 {item['round']} 轮 {item['name']}：{item['reason'].splitlines()[-1]}"
                for item in adaptation["related_candidates"]
            ),
            flush=True,
        )
    print(
        f"    关系假说：{proposal.hypothesis}\n    提取方式：{proposal.construction}\n    可读公式：{proposal.display_expression or proposal.expression}",
        flush=True,
    )
    for constant in proposal.constant_provenance:
        print(
            f"    参数 {constant['symbol']} ≈ {number(constant['value'])} [{constant['source']}]："
            f"{constant['calculation']} {constant['purpose']}",
            flush=True,
        )
    print(
        "    公式构造步骤：\n      " + "\n      ".join(proposal.derivation), flush=True
    )
