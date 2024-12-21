import json
from datetime import datetime
from typing import Annotated, Literal

from humanlayer import FunctionCall, FunctionCallSpec, HumanLayer
from langchain_community.tools import DuckDuckGoSearchResults, OpenWeatherMapQueryRun
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import NodeInterrupt
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode
from langgraph.utils.runnable import RunnableCallable

from agents.llama_guard import LlamaGuard, LlamaGuardOutput, SafetyAssessment
from agents.tools import (
    add_to_linear,
    calculator,
    list_linear_people,
    list_linear_projects,
    list_linear_teams,
)
from core import get_model, settings


class AgentState(MessagesState, total=False):
    """`total=False` is PEP589 specs.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """

    safety: LlamaGuardOutput
    remaining_steps: RemainingSteps
    pending_tool_calls: list[str]


tools = [
    calculator,
    add_to_linear,
    list_linear_teams,
    list_linear_people,
    list_linear_projects,
]

# Add weather tool if API key is set
# Register for an API key at https://openweathermap.org/api/
if settings.OPENWEATHERMAP_API_KEY:
    wrapper = OpenWeatherMapAPIWrapper(
        openweathermap_api_key=settings.OPENWEATHERMAP_API_KEY.get_secret_value()
    )
    tools.append(OpenWeatherMapQueryRun(name="Weather", api_wrapper=wrapper))

current_date = datetime.now().strftime("%B %d, %Y")
instructions = f"""
    You are a helpful research assistant with the ability to search the web and use other tools.
    Today's date is {current_date}.

    NOTE: THE USER CAN'T SEE THE TOOL RESPONSE.

    A few things to remember:
    - Please include markdown-formatted links to any citations used in your response. Only include one
    or two citations per response unless more are needed. ONLY USE LINKS RETURNED BY THE TOOLS.
    - Use calculator tool with numexpr to answer math questions. The user does not understand numexpr,
      so for the final response, use human readable format - e.g. "300 * 200", not "(300 \\times 200)".
    """


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    model = model.bind_tools(tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model


def format_safety_message(safety: LlamaGuardOutput) -> AIMessage:
    content = (
        f"This conversation was flagged for unsafe content: {', '.join(safety.unsafe_categories)}"
    )
    return AIMessage(content=content)


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)
    response = await model_runnable.ainvoke(state, config)

    # Run llama guard check here to avoid returning the message if it's unsafe
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("Agent", state["messages"] + [response])
    if safety_output.safety_assessment == SafetyAssessment.UNSAFE:
        return {"messages": [format_safety_message(safety_output)], "safety": safety_output}

    if state["remaining_steps"] < 2 and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, need more steps to process this request.",
                )
            ]
        }
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define the graph


class HumanLayerNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools_requiring_human_review: list[str]) -> None:
        self.tools_requiring_human_review = tools_requiring_human_review

    def __call__(self, inputs: dict):
        message = inputs["messages"][-1]
        tools_requiring_human_review = [
            t for t in message.tool_calls if t["name"] in self.tools_requiring_human_review
        ]
        new_pending_tool_calls = []
        if any(tools_requiring_human_review):
            hl = HumanLayer()
            for tool_call in tools_requiring_human_review:
                hl.create_function_call(
                    call_id=tool_call["id"],
                    spec=FunctionCallSpec(
                        fn=tool_call["name"],
                        kwargs=tool_call["args"],
                    ),
                )
                new_pending_tool_calls.append(tool_call["id"])
        return {"messages": [], "pending_tool_calls": new_pending_tool_calls}


class HumanLayerWaitNode:
    def __call__(self, inputs: dict):
        pending_tool_calls = inputs.get("pending_tool_calls", [])
        if not pending_tool_calls:
            return {"messages": []}
        new_tool_call_messages = []
        hl = HumanLayer()
        for tool_call_id in pending_tool_calls:
            call = hl.get_function_call(tool_call_id)
            if call.status is None or call.status.approved is None:
                raise NodeInterrupt(
                    f"Function call {tool_call_id} still awaiting human review - {call.spec.fn}({json.dumps(call.spec.kwargs, indent=2)})"
                )
            elif call.status.approved is False:
                new_tool_call_messages.append(
                    ToolMessage(
                        content=f"Function call {tool_call_id} was rejected by human review: {call.status.comment}",
                        name=call.spec.fn,
                        tool_call_id=tool_call_id,
                    )
                )
        return {"messages": new_tool_call_messages}


class IncrementalToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            pass
        else:
            raise ValueError("No message found in input")

        index = -1
        already_answered_tool_calls = []
        while not isinstance(messages[index], AIMessage):
            if isinstance(messages[index], ToolMessage):
                already_answered_tool_calls.append(messages[index].tool_call_id)
            else:
                raise ValueError(
                    "found non tool message while rewinding to find initial list of tool calls"
                )
            index -= 1
        message = messages[index]

        outputs = []
        for tool_call in message.tool_calls:
            if tool_call.id in already_answered_tool_calls:
                continue
            tool_result = self.tools_by_name[tool_call["name"]].invoke(tool_call["args"])
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


agent = StateGraph(AgentState)
agent.add_node("model", acall_model)
agent.set_entry_point("model")


agent.add_node("start_human_review", HumanLayerNode(["add_to_linear"]))
agent.add_node("await_human_response", HumanLayerWaitNode())
agent.add_edge("start_human_review", "await_human_response")

agent.add_node("tools", ToolNode(tools))
agent.add_edge("await_human_response", "tools")

agent.add_edge("tools", "model")


# After "model", if there are tool calls, run "tools". Otherwise END.
def pending_tool_calls(state: AgentState) -> Literal["tools", "done"]:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(last_message)}")
    if last_message.tool_calls:
        return "tools"
    return "done"


agent.add_conditional_edges(
    "model", pending_tool_calls, {"tools": "start_human_review", "done": END}
)

research_assistant = agent.compile(
    checkpointer=AsyncSqliteSaver.from_conn_string("checkpoints.sqlite")
)
