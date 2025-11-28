import json
import os
import time
from typing import List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage
from openai import OpenAI

from src.sim import Sim


class JudgeState(TypedDict, total=False):
    """状态定义，围绕 query vs top1 判定同一事件。"""

    messages: List[HumanMessage | AIMessage]
    query: str
    dataset_path: str
    model_path: str | None
    retrieved_text: str | None
    index_path: str | None
    score: float | None
    llm_judgment: bool | str | None
    reason: str | None
    analysis: dict | None
    step: str | None
    error: str | None


def _get_llm() -> OpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY 未设置，无法调用 LLM")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def retrieve_node(state: JudgeState) -> dict:
    """步骤1：用 Sim.query 取回 top-1 命中。"""
    query = state["query"]
    dataset_path = state["dataset_path"]
    model_path = state.get("model_path")

    try:
        best_match_title, index_path = Sim.query(
            query,
            dataset_path,
            model_path=model_path,
        )
        msg = f"✅ 已检索到 top-1\n{best_match_title}"
        return {
            "retrieved_text": best_match_title,
            "index_path": str(index_path),
            "step": "retrieved",
            "messages": state["messages"] + [AIMessage(content=msg)],
        }
    except Exception as e:
        err_msg = f"检索失败：{e}"
        return {
            "error": err_msg,
            "step": "retrieve_failed",
            "messages": state["messages"] + [AIMessage(content=err_msg)],
        }


def judge_node(state: JudgeState) -> dict:
    """步骤2：LLM 判断两句话是否同一件事。"""
    if state.get("step") == "retrieve_failed":
        return {
            "llm_judgment": "unknown",
            "reason": state.get("error") or "检索失败，无法判定",
            "step": "completed",
            "messages": state["messages"],
        }

    query = state["query"]
    retrieved = state.get("retrieved_text") or ""

    prompt = f"""
You are an Event Equivalence Evaluation Assistant.
Your task is to determine whether two sentences describe the same real-world event.
Follow the rules below rigorously and output a structured conclusion with clear reasoning.

Core Evaluation Principles

Event Theme Consistency
Two sentences may be considered the same event if their core theme aligns, even when phrased differently.
Core elements include:
- Actor (person, organization, institution, country)
- Action (e.g., rises, falls, announces, investigated, wins, loses)
- Object or target (e.g., Bitcoin price, team ranking, company product, policy)

Time Tolerance Rule
If the time difference between the two sentences is within 30 days, treat them as potentially the same event.
If a sentence does not explicitly state a time but the context strongly aligns, it may still qualify as the same event.

Numerical Range Rule
When the event description contains numeric values (price, amount, percentage, ranking, population, etc.), evaluate them as follows:
- For a given number N, allow a tolerance window of ± 1× the highest digit unit.
- Examples:
  - For 10, acceptable range is 0–20.
  - For 90,000, acceptable range is 80,000–100,000.
- If both sentences' numeric values fall within each other's tolerance window, consider them equivalent.

Contextual and Causal Alignment
If background, causal links, affected subjects, and scenario conditions are aligned, consider the two sentences to be the same event even if wording differs significantly.

Conflict Priority Rule
The two sentences must be treated as different events if any of the following holds:
- Time difference exceeds 1 month and continuity is not implied
- Numeric difference falls outside the defined tolerance range
- Opposite directional behavior (e.g., rise vs. fall, increase vs. decrease)
- Different actors, different objects, or unrelated scenario context
- Mutually exclusive conditions (different locations, competitions, markets, etc.)

Output Format Requirements

Return your evaluation in the following JSON format:

{{
  "same_event": true or false,
  "reason": "A concise explanation referencing time, numbers, theme, and entities.",
  "analysis": {{
    "theme_similarity": "...",
    "time_evaluation": "...",
    "number_evaluation": "...",
    "entity_alignment": "..."
  }}
}}

Sentence A: {query}
Sentence B: {retrieved}
Respond with JSON only.
"""
    client = _get_llm()
    resp = client.chat.completions.create(
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        messages=[{"role": "system", "content": prompt}],
        temperature=0.0,
    )
    content = resp.choices[0].message.content if resp.choices else ""
    judgment: bool | str = "unknown"
    reason = ""
    analysis = None
    try:
        parsed = json.loads(content)
        raw_same = parsed.get("same_event")
        if isinstance(raw_same, bool):
            judgment = raw_same
        elif isinstance(raw_same, str):
            lowered = raw_same.strip().lower()
            if lowered in {"true", "yes", "1"}:
                judgment = True
            elif lowered in {"false", "no", "0"}:
                judgment = False
            else:
                judgment = "unknown"
        else:
            judgment = "unknown"
        reason = parsed.get("reason") or ""
        analysis = parsed.get("analysis") if isinstance(parsed.get("analysis"), dict) else None
    except Exception:
        # 回退：保留原始输出，标记未知
        judgment = "unknown"
        reason = f"LLM输出无法解析：{content}"

    return {
        "llm_judgment": judgment,
        "reason": reason,
        "analysis": analysis,
        "step": "completed",
        "messages": state["messages"] + [AIMessage(content=content)],
    }


def build_graph() -> StateGraph:
    """构建 LangGraph 工作流图。"""
    workflow = StateGraph(JudgeState)

    workflow.add_node("retrieve_node", retrieve_node)
    workflow.add_node("judge_node", judge_node)

    workflow.add_edge(START, "retrieve_node")
    workflow.add_edge("retrieve_node", "judge_node")
    workflow.add_edge("judge_node", END)

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app


def run_workflow(query: str, dataset_path: str, model_path: Optional[str] = None) -> dict:
    """对外调用工作流，返回完整状态。"""
    app = build_graph()
    final_state = app.invoke(
        {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "dataset_path": dataset_path,
            "model_path": model_path,
            "retrieved_text": None,
            "index_path": None,
            "score": None,
            "llm_judgment": None,
            "reason": None,
            "analysis": None,
            "step": None,
            "error": None,
        },
        config={"configurable": {"thread_id": f"cli-{int(time.time())}"}},
    )
    return final_state


def run_once(query: str, dataset_path: str, model_path: Optional[str] = None) -> str:
    final_state = run_workflow(query, dataset_path, model_path=model_path)
    result = {
        "same_event": final_state.get("llm_judgment", "unknown"),
        "reason": final_state.get("reason", ""),
        "title": final_state.get("retrieved_text", ""),
        "analysis": final_state.get("analysis") or {},
    }
    return json.dumps(result, ensure_ascii=False)
