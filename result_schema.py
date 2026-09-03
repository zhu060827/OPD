from __future__ import annotations

"""统一结果格式。

Tabular / Math / Code 三条 pipeline 最终都包装成 ResultRecord。
前端展示和 outputs/ 里的 JSON/JSONL 文件都围绕这个结构，后续增加字段也不容易乱。
"""

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional


Modality = Literal["tabular", "math", "code"]


@dataclass
class ResultRecord:
    """结题版统一返回结构。"""

    id: str
    modality: Modality
    input_summary: Dict[str, Any]
    structured_representation: Dict[str, Any]
    generated: Dict[str, Any]
    verification: Dict[str, Any]
    feedback: str
    accepted: bool
    round: int
    errors: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    saved_path: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    @classmethod
    def create(
        cls,
        modality: Modality,
        input_summary: Optional[Dict[str, Any]] = None,
        structured_representation: Optional[Dict[str, Any]] = None,
        generated: Optional[Dict[str, Any]] = None,
        verification: Optional[Dict[str, Any]] = None,
        feedback: str = "",
        accepted: bool = False,
        round: int = 1,
        errors: Optional[List[str]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        saved_path: str = "",
    ) -> "ResultRecord":
        return cls(
            id=str(uuid.uuid4()),
            modality=modality,
            input_summary=input_summary or {},
            structured_representation=structured_representation or {},
            generated=generated or {},
            verification=verification or {},
            feedback=feedback,
            accepted=accepted,
            round=round,
            errors=errors or [],
            metrics=metrics or {},
            saved_path=saved_path,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def make_result_record(
    modality: Modality,
    input_summary: Optional[Dict[str, Any]] = None,
    structured_representation: Optional[Dict[str, Any]] = None,
    generated: Optional[Dict[str, Any]] = None,
    verification: Optional[Dict[str, Any]] = None,
    feedback: str = "",
    accepted: bool = False,
    round: int = 1,
    errors: Optional[List[str]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    saved_path: str = "",
) -> Dict[str, Any]:
    return ResultRecord.create(
        modality=modality,
        input_summary=input_summary,
        structured_representation=structured_representation,
        generated=generated,
        verification=verification,
        feedback=feedback,
        accepted=accepted,
        round=round,
        errors=errors,
        metrics=metrics,
        saved_path=saved_path,
    ).to_dict()


def write_json(path: str | Path, data: Any) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(output)


def append_jsonl(path: str | Path, data: Any) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
    return str(output)


def read_json_if_exists(path: str | Path) -> Any:
    input_path = Path(path)
    if not input_path.exists():
        return None
    return json.loads(input_path.read_text(encoding="utf-8"))


def read_jsonl_tail(path: str | Path, limit: int = 5) -> List[Dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in input_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows[-limit:]
