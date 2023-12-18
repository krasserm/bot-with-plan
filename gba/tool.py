import json
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain.chat_models.base import BaseChatModel
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatGeneration,
    ChatResult,
    FunctionMessage,
    HumanMessage,
    SystemMessage
)
from langchain.tools import StructuredTool


SYSTEM_TEMPLATE = """You are an AI agent who answers a user's request in one or more steps by calling one of the following tools: 

{tool_descriptions}

Call exactly one of these tools at each step. Call tool 'respond_to_user' if you want to respond to the user directly or provide a final answer after having called other tools from the list.

Avoid guessing tool call results. Only rely on information returned from tool calls."""


TOOL_CALL_TEMPLATE = "Please make tool call {tool_call_num} for me: {tool_call}"
TOOL_CALL_RESULT_TEMPLATE = "Use the result of tool call {tool_call_num}: {tool_call_result}"
TOOL_CALL_GEN_PROMPT = "Now make the tool call, nothing else."

REASON_TEMPLATE_1 = """The user request is: {user_request}

Can you directly respond to the user with an answer or do you need to call another tool to obtain more information? Answer with a single sentence."""

REASON_TEMPLATE_N = """The result of tool call {tool_call_num} is: {tool_call_result}

Do you have enough information to respond with a final answer to the user or do you need to call another tool to obtain more information? Answer with a single sentence."""


def respond_to_user(answer: str):
    """A tool for responding to the user directly with the final answer."""
    return answer


class ToolCalling(BaseChatModel):
    model: BaseChatModel
    respond_to_user_fn: Callable[[str], Any] = respond_to_user
    system_template: str = SYSTEM_TEMPLATE
    tool_call_template: str = TOOL_CALL_TEMPLATE
    tool_call_result_template: str = TOOL_CALL_RESULT_TEMPLATE
    tool_call_gen_prompt: str = TOOL_CALL_GEN_PROMPT
    reason_template_1: str = REASON_TEMPLATE_1
    reason_template_n: str = REASON_TEMPLATE_N
    reason_step: bool = True

    @property
    def _llm_type(self) -> str:
        return "tool-call-support"

    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            tools: Optional[List[StructuredTool]] = None,
            **kwargs: Any,
    ) -> ChatResult:
        if not tools:
            return self.model._generate(messages, **kwargs)

        *prev_messages, last_message = messages

        if not isinstance(last_message, (HumanMessage, FunctionMessage)):
            raise ValueError("last message in messages must be a HumanMessage or FunctionMessage")

        response_tool = StructuredTool.from_function(self.respond_to_user_fn)

        tools = tools + [response_tool]
        tools_schema = tools_to_schema(tools)
        prop_order = ["tool", "arguments"]

        augmented_messages, tool_call_num = self._augment_messages(prev_messages, tools)

        if isinstance(last_message, HumanMessage):
            thought_prompt = self.reason_prompt_1(user_request=last_message.content)
        else:  # function message
            thought_prompt = self.reason_prompt_n(tool_call_result=last_message.content, tool_call_num=tool_call_num)

        augmented_message = HumanMessage(content=thought_prompt)
        augmented_messages.append(augmented_message)

        response_message = self.model.predict_messages(
            augmented_messages,
            stop=stop,
            schema=None if self.reason_step else tools_schema,
            prop_order=prop_order,
        )
        print(f"\nReasoning: {response_message.content}")

        if self.reason_step:
            augmented_messages.append(response_message)

            tool_call_gen_request = HumanMessage(content=self.tool_call_gen_prompt)
            augmented_messages.append(tool_call_gen_request)

            response_message = self.model.predict_messages(
                augmented_messages,
                stop=stop,
                schema=tools_schema,
                prop_order=prop_order,
            )

        tool_call = json.loads(response_message.content)

        if tool_call["tool"] == response_tool.name:
            response = response_tool.invoke(tool_call["arguments"])
            response_message = AIMessage(content=response)
        else:
            response_message = AIMessage(content="")
            response_message.additional_kwargs["tool_call"] = tool_call

        return ChatResult(generations=[ChatGeneration(message=response_message)])

    def _augment_messages(
            self,
            messages: List[BaseMessage],
            tools: List[StructuredTool],
    ) -> Tuple[List[BaseMessage], int]:
        tool_call_num = 1

        if messages and isinstance(messages[0], SystemMessage):
            raise ValueError("User-provided SystemMessage not allowed. Use ToolProtocol(system_template=...) instead.")

        augmented_messages = [SystemMessage(content=self.system_prompt(tools))]

        for message in messages:
            if isinstance(message, AIMessage) and "tool_call" in message.additional_kwargs:
                tool_call = message.additional_kwargs["tool_call"]
                augmented_message = AIMessage(content=self.tool_call_prompt(tool_call, tool_call_num))
            elif isinstance(message, FunctionMessage):
                augmented_message = HumanMessage(content=self.tool_call_result_prompt(message.content, tool_call_num))
                tool_call_num += 1
            else:
                augmented_message = message

            augmented_messages.append(augmented_message)

        return augmented_messages, tool_call_num

    def reason_prompt_1(self, user_request: str):
        return self.reason_template_1.format(user_request=user_request)

    def reason_prompt_n(self, tool_call_result: str, tool_call_num: int):
        return self.reason_template_n.format(tool_call_result=tool_call_result, tool_call_num=tool_call_num)

    def tool_call_prompt(self, tool_call: Dict[str, Any], tool_call_num: int):
        return self.tool_call_template.format(tool_call=json.dumps(tool_call), tool_call_num=tool_call_num)

    def tool_call_result_prompt(self, tool_call_result: str, tool_call_num: int):
        return self.tool_call_result_template.format(tool_call_result=tool_call_result, tool_call_num=tool_call_num)

    def system_prompt(self, tools: List[StructuredTool]) -> str:
        tool_descriptions = "\n".join([f"- {tool.description}" for tool in tools])
        return self.system_template.format(tool_descriptions=tool_descriptions)


def tools_to_schema(tools: list[StructuredTool]) -> Dict[str, Any]:
    return {"oneOf": [tool_to_schema(tool) for tool in tools]}


def tool_to_schema(tool: StructuredTool) -> Dict[str, Any]:
    schema = tool.args_schema.schema()

    properties = {}

    for k, v in schema["properties"].items():
        properties[k] = {"type": v["type"]}

    return {
        "type": "object",
        "properties": {
            "tool": {"const": tool.name},
            "arguments": {
                "type": "object",
                "properties": properties
            }
        }
    }
