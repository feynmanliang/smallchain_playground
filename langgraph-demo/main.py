import json
from typing import Annotated, Any, Dict, Type, TypedDict, TypeVar

from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import BaseTool, ToolNode, tools_condition
from pydantic import BaseModel


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


class AddTool(BaseTool):
    class AddToolInput(BaseModel):
        a: int
        b: int

    name: str = "add"
    description: str = "Add two numbers together"
    args_schema: Type[BaseModel] = AddToolInput

    def _run(self, a: int, b: int) -> int:
        return a + b


tool = AddTool()
tools = [tool]
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)


tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)


# The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
# it is fine directly responding. This conditional routing defines the main agent loop.
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

graph = graph_builder.compile()

with open("graph.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())


def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


stream_graph_updates("Add (2 + 5) and (3 + 4), then add the results")
