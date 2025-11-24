from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import END, START, StateGraph
from langchain_core.runnables.graph_mermaid import draw_mermaid_png

from src.state import State
from src.tools import container_stats, list_containers, system_df
from src.utils import get_llm
from src.prompts import planner_system_prompt, executor_system_prompt, writer_system_prompt
from langchain.agents import create_agent

TOOLS = [list_containers, container_stats, system_df]

planner_llm = get_llm()
executor_llm = get_llm()
writer_llm = get_llm()

executor_llm.bind_tools(TOOLS)

executor_agent_graph = create_agent(
    model=executor_llm, tools=TOOLS, system_prompt=executor_system_prompt, # debug=True
)

def planner_node(state: State) -> State:
    sys = SystemMessage(content=planner_system_prompt)
    
    res = planner_llm.invoke([sys] + state.get("messages", []))
    
    steps = [s.strip("- •").strip() for s in (res.content or "").split("\n") if s.strip()]
    new_state = State(state)
    new_state["messages"] = state.get("messages", []) + [res]
    new_state["plan"] = steps 
    
    return new_state

def executor_node(state: State) -> State:
    history = state.get("messages")
    result = executor_agent_graph.invoke({"messages": history})
    draft = result["messages"][-1].content
    appended = result["messages"][2:] if len(result["messages"]) > 2 else []

    new_state = State(state)
    new_state["messages"] = state.get("messages", []) + appended
    new_state["draft"] = draft
 
    return new_state

def writer_node(state: State) -> State:
    history = str(state.get("messages", []))
    chain = writer_system_prompt | writer_llm
    res = chain.invoke({"history": history})

    new_state = State(state)
    new_state["messages"] = state.get("messages", []) + [AIMessage(content=f"[summary] {res.content}")]
    new_state["result"] = res.content

    return new_state

graph = StateGraph(State)

graph.add_node("planner", planner_node)
graph.add_node("executor", executor_node)
graph.add_node('writer', writer_node)

graph.add_edge(START, "planner")
graph.add_edge("planner", "executor")
graph.add_edge("executor", "writer")
graph.add_edge("writer", END)

app = graph.compile()
mermaid_syntax = app.get_graph().draw_mermaid()
png_bytes = draw_mermaid_png(
    mermaid_syntax,
    output_file_path="workflow_graph.png",
    background_color="white",
    padding=10
)

if __name__ == "__main__":
    query = 'How much storage is used on my system?'
    init: State = {
        "messages": [HumanMessage(content=query)],
        "plan": list(),
        "draft": list(),
        "result": ""
    }
    state = app.invoke(init)
    print("\n--- SUMMARY ---\n", state.get("result"))