from typing import Annotated, Any, Generator, Optional, Sequence, TypedDict, Union

import mlflow
from databricks_langchain import (
    ChatDatabricks,
    UCFunctionToolkit,
    VectorSearchRetrieverTool,
    # DatabricksFunctionClient
)
from langchain_core.messages import AIMessage, AIMessageChunk, AnyMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
    to_chat_completions_input,
)
from unitycatalog.ai.core.databricks import DatabricksFunctionClient

#Define the endpoint to use for the agent foundation and system prompt
LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"
llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)
system_prompt = "You are a helpful assistant that can run Python code." #Give my agent a better description later.
tools = []

# You can use UDFs in Unity Catalog as agent tools
# Below, we add the `system.ai.python_exec` UDF, which provides
# a python code interpreter tool to our agent
# You can also add local LangChain python tools. See https://python.langchain.com/docs/concepts/tools

# TODO: Add additional tools
UC_TOOL_NAMES = ["system.ai.python_exec"]
client = DatabricksFunctionClient(execution_mode="serverless")
uc_toolkit = UCFunctionToolkit(function_names=UC_TOOL_NAMES, client=client)
tools.extend(uc_toolkit.tools)
#Vector search tools are used for unstructured text tools. Useful for a RAG agent.
VECTOR_SEARCH_TOOLS = []

# To add vector search retriever tools,
# use VectorSearchRetrieverTool and create_tool_info,
# then append the result to TOOL_INFOS.
# Example:
# VECTOR_SEARCH_TOOLS.append(
#     VectorSearchRetrieverTool(
#         index_name="",
#         # filters="..."
#     )
# )
tools.extend(VECTOR_SEARCH_TOOLS)

class AgentState(TypedDict):
    messages: Annotated[Sequence[AnyMessage], add_messages]
    custom_inputs: Optional[dict[str, Any]]
    custom_outputs: Optional[dict[str, Any]]
    
def create_tool_calling_agent(
    model: ChatDatabricks,
    tools: Union[ToolNode, Sequence[BaseTool]],
    system_prompt: Optional[str] = None,
):
    model = model.bind_tools(tools)

    # Define the function that determines which node to go to
    def should_continue(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        # If there are function calls, continue. else, end
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "continue"
        else:
            return "end"

    if system_prompt:
        preprocessor = RunnableLambda(
            lambda state: [{"role": "system", "content": system_prompt}] + state["messages"]
        )
    else:
        preprocessor = RunnableLambda(lambda state: state["messages"])
    model_runnable = preprocessor | model

    def call_model(
        state: AgentState,
        config: RunnableConfig,
    ):
        response = model_runnable.invoke(state, config)

        return {"messages": [response]}

    workflow = StateGraph(AgentState)

    workflow.add_node("agent", RunnableLambda(call_model))
    workflow.add_node("tools", ToolNode(tools))

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()
    
class LangGraphResponsesAgent(ResponsesAgent):
    def __init__(self, agent):
        self.agent = agent

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)

    def predict_stream(
        self,
        request: ResponsesAgentRequest,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        cc_msgs = to_chat_completions_input([i.model_dump() for i in request.input])

        for event in self.agent.stream({"messages": cc_msgs}, stream_mode=["updates", "messages"]):
            if event[0] == "updates":
                for node_data in event[1].values():
                    if len(node_data.get("messages", [])) > 0:
                        yield from output_to_responses_items_stream(node_data["messages"])
            # filter the streamed messages to just the generated text messages
            elif event[0] == "messages":
                try:
                    chunk = event[1][0]
                    if isinstance(chunk, AIMessageChunk) and (content := chunk.content):
                        yield ResponsesAgentStreamEvent(
                            **self.create_text_delta(delta=content, item_id=chunk.id),
                        )
                except Exception as e:
                    print(e)


mlflow.langchain.autolog()
agent = create_tool_calling_agent(llm, tools, system_prompt)
AGENT = LangGraphResponsesAgent(agent)
mlflow.models.set_model(AGENT)
