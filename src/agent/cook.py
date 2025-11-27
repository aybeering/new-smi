"""批量调用 workflow，对 CSV 每行文本运行结构化抽取并输出结果。"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# 保证可以直接 python src/agent/cook.py 运行
CURRENT_DIR = Path(__file__).resolve().parent
ROOT = CURRENT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

try:
    from src.agent.workflow import run_workflow
except ImportError:
    # 兼容在 src/agent 目录下直接运行
    from workflow import run_workflow


def _parse_time_span(time_data: Any) -> Tuple[str, str, str]:
    """从 workflow 返回的 time 字段中提取开始/结束/备注。"""
    if not isinstance(time_data, dict):
        return "", "", ""

    # 新格式：{"start": "...", "end": "...", "note": "..."}
    if "start" in time_data or "end" in time_data:
        return (
            time_data.get("start") or "",
            time_data.get("end") or "",
            time_data.get("note") or "",
        )

    # 兼容旧格式：{"intervals": [{"start": "...", "end": "...", "text": "..."}], "note": "..."}
    intervals = time_data.get("intervals")
    note = time_data.get("note") or ""
    if isinstance(intervals, list) and intervals:
        first = intervals[0] or {}
        start = first.get("start") or ""
        end = first.get("end") or ""
        if not note:
            note = first.get("text") or ""
        return start, end, note
    return "", "", note


def _parse_interest(event_data: Any) -> str:
    """从 event 字段中取第一条事件文本。"""
    if isinstance(event_data, dict):
        texts = event_data.get("texts")
        if isinstance(texts, list) and texts:
            first = texts[0]
            if isinstance(first, str):
                return first
    return ""


def _parse_subjects(person_data: Any, limit: int = 12) -> List[str]:
    """从 person.entities 中取前 N 个名称，不足填空。"""
    names: List[str] = []
    if isinstance(person_data, dict):
        entities = person_data.get("entities")
        if isinstance(entities, list):
            for ent in entities:
                if isinstance(ent, dict):
                    name = ent.get("name")
                    if isinstance(name, str):
                        names.append(name)
                if len(names) >= limit:
                    break

    while len(names) < limit:
        names.append("")
    return names[:limit]


def _extract_fields(state: Dict[str, Any]) -> Dict[str, Any]:
    """从 workflow 状态中提取目标字段，必要时解析 final_answer JSON。"""
    parsed: Dict[str, Any] = {}
    if isinstance(state, dict):
        fa = state.get("final_answer")
        if isinstance(fa, str):
            try:
                parsed = json.loads(fa)
            except Exception:
                parsed = {}
    # 先从状态取，没有则回退解析内容
    time_data = state.get("time_span") if isinstance(state, dict) else None
    if time_data is None:
        time_data = parsed.get("time_span")
    time_start, time_end, time_note = _parse_time_span(time_data)

    interest = ""
    if isinstance(state, dict):
        interest = state.get("interest") or ""
    if not interest:
        interest = parsed.get("interest") or ""

    subjects: List[str] = []
    if isinstance(state, dict):
        sub = state.get("subjects")
        if isinstance(sub, list):
            subjects = sub
    if not subjects:
        sub = parsed.get("subjects")
        if isinstance(sub, list):
            subjects = sub

    return {
        "time_span_start": time_start,
        "time_span_end": time_end,
        "time_span_note": time_note,
        "interest": interest,
        "subjects": _parse_subjects({"entities": [{"name": s} for s in subjects]}),
    }


def process_file(
    input_path: Path,
    output_path: Path,
    text_column: str = "text",
    limit: int | None = None,
) -> None:
    """读取输入 CSV，逐行调用 workflow 并写出结果 CSV。"""
    fieldnames = (
        [text_column]
        + ["time_span_start", "time_span_end", "time_span_note", "interest"]
        + [f"subjects_{i}" for i in range(1, 13)]
    )

    with input_path.open("r", encoding="utf-8", newline="") as fin, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for idx, row in enumerate(reader):
            if limit is not None and idx >= limit:
                break

            text = row.get(text_column) or ""
            if not text:
                extracted = _extract_fields({})
            else:
                state = run_workflow(text)
                extracted = _extract_fields(state)

            out_row: Dict[str, Any] = {
                text_column: text,
                "time_span_start": extracted["time_span_start"],
                "time_span_end": extracted["time_span_end"],
                "time_span_note": extracted["time_span_note"],
                "interest": extracted["interest"],
            }
            for i, name in enumerate(extracted["subjects"], start=1):
                out_row[f"subjects_{i}"] = name

            writer.writerow(out_row)


def main():
    parser = argparse.ArgumentParser(
        description="批量调用 workflow，对 CSV 文本列生成时间/兴趣/主体字段。"
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        type=Path,
        default=Path("/Users/ayang/agent/new-smi/data/very_sample.csv"),
        help="输入 CSV 路径（可选，默认 data/very_sample.csv）",
    )
    parser.add_argument(
        "output_csv",
        nargs="?",
        type=Path,
        default=Path("/Users/ayang/agent/new-smi/data/very_sample_output.csv"),
        help="输出 CSV 路径（可选，默认 data/very_sample_output.csv）",
    )
    parser.add_argument(
        "--text-column", default="title", help="输入 CSV 中文本列列名，默认 title"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="仅处理前 N 行，调试用，可选"
    )
    args = parser.parse_args()

    process_file(args.input_csv, args.output_csv, args.text_column, args.limit)


if __name__ == "__main__":
    main()
