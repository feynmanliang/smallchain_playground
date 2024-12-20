from typing import Annotated, Type, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import BaseTool, ToolNode, tools_condition
from pydantic import BaseModel

memory = MemorySaver()


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

graph = graph_builder.compile(
    checkpointer=memory,
    interrupt_before=["tools"],
)

with open("graph.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())

user_input = "Add (2 + 5) and (3 + 4), then add the results"
config = {"configurable": {"thread_id": "1"}}
# The config is the **second positional argument** to stream() or invoke()!
events = graph.stream(
    {"messages": [("user", user_input)]}, config, stream_mode="values"
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()


print("*"*5 + "PAUSED FOR HUMAN INPUT" + "*"*5)

snapshot = graph.get_state(config)
print(snapshot.next)

existing_message = snapshot.values["messages"][-1]
existing_message.tool_calls

# `None` will append nothing new to the current state, letting it resume as if it had never been interrupted
events = graph.stream(None, config, stream_mode="values")
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

print("*"*5 + "PAUSED FOR HUMAN INPUT" + "*"*5)

snapshot = graph.get_state(config)
print(snapshot.next)

existing_message = snapshot.values["messages"][-1]
existing_message.tool_calls

# `None` will append nothing new to the current state, letting it resume as if it had never been interrupted
events = graph.stream(None, config, stream_mode="values")
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
