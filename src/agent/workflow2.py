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
    llm_judgment: str | None
    reason: str | None
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
你需要判断两句话是否描述同一件具体事件,不用做到全都一致，有一些数字或者时间上的微小差距可以忽略判断标准：主体、时间、动作是否匹配。
句子A（用户原问）：{query}
句子B（检索Top-1）：{retrieved}

请回答 JSON：
{{
  "same_event": "yes|no|unknown",
  "reason": "简要说明主体/时间/动作是否匹配，若信息不足说明不足之处"
}}
"""
    client = _get_llm()
    resp = client.chat.completions.create(
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        messages=[{"role": "system", "content": prompt}],
        temperature=0.0,
    )
    content = resp.choices[0].message.content if resp.choices else ""
    judgment = "unknown"
    reason = ""
    try:
        parsed = json.loads(content)
        judgment = str(parsed.get("same_event") or "unknown").lower()
        reason = parsed.get("reason") or ""
    except Exception:
        # 回退：保留原始输出，标记未知
        judgment = "unknown"
        reason = f"LLM输出无法解析：{content}"

    return {
        "llm_judgment": judgment,
        "reason": reason,
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
    }
    return json.dumps(result, ensure_ascii=False)
