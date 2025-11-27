import argparse
import json
import os
import time
from typing import List, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from openai import OpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from tavily import TavilyClient


class SearchState(TypedDict, total=False):
    """状态定义，包含对话消息和各阶段产物。"""

    messages: List[HumanMessage | AIMessage]
    user_query: str
    search_query: str | None
    search_results: str | None
    step: str | None
    final_answer: str | None
    time_span: dict | None
    interest: str | None
    subjects: List[str] | None


def _get_llm() -> OpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY 未设置，无法调用 LLM")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def llm_node(state: SearchState) -> dict:
    """步骤1：理解用户查询并生成搜索关键词。"""
    user_message = state["messages"][-1].content
    understand_prompt = f"""分析用户的查询："{user_message}"
请完成两个任务：
1. 简洁总结用户想要了解什么（关注时间线、关心的事件、涉及主体）
2. 生成最适合搜索引擎的关键词（中英文均可，要精准，方便查到时间/主体）

格式：
理解：[用户需求总结]
搜索词：[最佳搜索关键词]"""

    client = _get_llm()
    resp = client.chat.completions.create(
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        messages=[{"role": "system", "content": understand_prompt}],
        temperature=0.0,
    )
    response_text = resp.choices[0].message.content if resp.choices else ""

    search_query = user_message  # 默认使用原始查询
    if "搜索词：" in response_text:
        search_query = response_text.split("搜索词：", 1)[1].strip()

    return {
        "user_query": user_message,
        "search_query": search_query,
        "step": "understood",
        "messages": state["messages"] + [AIMessage(content=f"我将为您搜索：{search_query}")],
    }


def _format_tavily_results(raw: dict) -> str:
    answer = raw.get("answer")
    hits = raw.get("results", [])
    lines = []
    if answer:
        lines.append(f"回答摘要：{answer}")
    for item in hits:
        title = item.get("title") or "未命名结果"
        url = item.get("url") or ""
        content = item.get("content") or ""
        lines.append(f"- {title} ({url})\n  {content}")
    return "\n".join(lines) if lines else "未获取到搜索结果"


def web_node(state: SearchState) -> dict:
    """步骤2：使用 Tavily API 进行真实搜索（无 Key 时自动跳过）。"""
    search_query = state.get("search_query") or state["user_query"]
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {
            "search_results": "未配置 TAVILY_API_KEY，跳过真实搜索。",
            "step": "search_failed",
            "messages": state["messages"] + [AIMessage(content="未配置 Tavily，改为直接回答。")],
        }

    client = TavilyClient(api_key=api_key)
    try:
        print(f"🔍 正在搜索: {search_query}")
        response = client.search(
            query=search_query, search_depth="basic", max_results=5, include_answer=True
        )
        search_results = _format_tavily_results(response)
        return {
            "search_results": search_results,
            "step": "searched",
            "messages": state["messages"] + [AIMessage(content="✅ 搜索完成！正在整理答案...")],
        }
    except Exception as e:
        return {
            "search_results": f"搜索失败：{e}",
            "step": "search_failed",
            "messages": state["messages"] + [AIMessage(content="❌ 搜索遇到问题，改为直接回答。")],
        }


def answer_node(state: SearchState) -> dict:
    """步骤3：基于搜索结果生成结构化三项答案。"""
    client = _get_llm()
    if state["step"] == "search_failed":
        prompt = (
            "搜索API暂时不可用，请仅基于已知常识回答，若不确定请写 unknown，"
            "同时在 note 中说明依据不足。\n"
            f"用户问题：{state['user_query']}"
        )
    else:
        prompt = f"""基于以下搜索结果，回答三个问题，若不确定请输出 unknown 并附 note：
用户问题：{state['user_query']}
搜索结果：
{state['search_results']}

输出 JSON，格式：
{{
  "time_span": {{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD", "note": "不确定时说明推断依据"}},
  "interest": "用户最关心的具体事件/问题",
  "subjects": ["受影响的主体1", "主体2"]
}}
要求：
- start/end 尽量精确到日；无法精确则给出最合理日期或 unknown。
- subjects 给出12个，使用简短名称，按相关性排序。"""

    resp = client.chat.completions.create(
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        messages=[{"role": "system", "content": prompt}],
        temperature=0.2,
    )
    answer = resp.choices[0].message.content if resp.choices else ""
    try:
        parsed = json.loads(answer)
    except Exception:
        parsed = {}

    return {
        "final_answer": answer,
        "time_span": parsed.get("time_span"),
        "interest": parsed.get("interest"),
        "subjects": parsed.get("subjects"),
        "step": "completed",
        "messages": state["messages"] + [AIMessage(content=answer)],
    }


def build_graph():
    """构建 LangGraph 工作流图。"""
    workflow = StateGraph(SearchState)

    workflow.add_node("llm_node", llm_node)
    workflow.add_node("web_node", web_node)
    workflow.add_node("answer_node", answer_node)

    workflow.add_edge(START, "llm_node")
    workflow.add_edge("llm_node", "web_node")
    workflow.add_edge("web_node", "answer_node")
    workflow.add_edge("answer_node", END)

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app


def run_workflow(question: str) -> dict:
    """对外调用工作流，返回完整状态。"""
    app = build_graph()
    final_state = app.invoke(
        {
            "messages": [HumanMessage(content=question)],
            "user_query": question,
            "search_query": None,
            "search_results": None,
            "step": None,
            "final_answer": None,
            "time_span": None,
            "interest": None,
            "subjects": None,
        },
        config={"configurable": {"thread_id": f"cli-{int(time.time())}"}},
    )
    return final_state


def run_once(question: str) -> str:
    final_state = run_workflow(question)
    return final_state.get("final_answer", "")


def main():
    parser = argparse.ArgumentParser(description="简单的搜索-回答工作流演示")
    parser.add_argument("query", nargs="?", help="要提问的内容（为空则进入交互输入）")
    args = parser.parse_args()

    question = args.query or input("请输入你的问题：").strip()
    if not question:
        raise SystemExit("问题不能为空")

    answer = run_once(question)
    print("\n=== 回答 ===")
    print(answer)


if __name__ == "__main__":
    main()
