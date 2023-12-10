from typing import Any, List, Optional, Tuple, Union

from langchain.tools import StructuredTool
from langchain.agents import BaseSingleActionAgent
from langchain.agents.format_scratchpad.openai_functions import format_to_openai_function_messages
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.manager import Callbacks
from langchain.prompts.chat import (
    BaseMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import (
    AgentAction,
    AgentFinish,
    BasePromptTemplate,
)
from langchain.schema.agent import AgentActionMessageLog
from langchain.schema.messages import AIMessage, BaseMessage


from gba.tool import ToolCalling


# ----------------------------------------------
#  Inspired by LangChain's OpenAIFunctionsAgent
# ----------------------------------------------


class Agent(BaseSingleActionAgent):
    model: ToolCalling
    tools: List[StructuredTool]
    prompt: BasePromptTemplate

    @property
    def input_keys(self) -> List[str]:
        return ["input"]

    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        with_functions: bool = True,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        agent_scratchpad = format_to_openai_function_messages(intermediate_steps)
        selected_inputs = {k: kwargs[k] for k in self.prompt.input_variables if k != "agent_scratchpad"}
        full_inputs = dict(**selected_inputs, agent_scratchpad=agent_scratchpad)

        messages = self.prompt.format_prompt(**full_inputs).to_messages()

        if with_functions:
            predicted_message = self.model.predict_messages(messages, tools=self.tools, callbacks=callbacks)
        else:
            predicted_message = self.model.predict_messages(messages, callbacks=callbacks)

        return self._parse_prediction(predicted_message)

    async def aplan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        raise NotImplementedError

    @classmethod
    def from_llm_and_tools(
        cls,
        model: ToolCalling,
        tools: List[StructuredTool],
        callback_manager: Optional[BaseCallbackManager] = None,
        extra_prompt_messages: Optional[List[BaseMessagePromptTemplate]] = None,
        **kwargs: Any,
    ) -> BaseSingleActionAgent:
        prompt = cls.create_prompt(extra_prompt_messages=extra_prompt_messages)
        return cls(
            model=model,
            prompt=prompt,
            tools=tools,
            callback_manager=callback_manager,
            **kwargs,
        )

    @classmethod
    def create_prompt(
            cls, extra_prompt_messages: Optional[List[BaseMessagePromptTemplate]] = None
    ) -> BasePromptTemplate:
        _prompts = extra_prompt_messages or []
        messages = [
            *_prompts,
            HumanMessagePromptTemplate.from_template("{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
        return ChatPromptTemplate(messages=messages)

    @staticmethod
    def _parse_prediction(message: BaseMessage) -> Union[AgentAction, AgentFinish]:
        if not isinstance(message, AIMessage):
            raise TypeError(f"Expected an AI message got {type(message)}")

        tool_call = message.additional_kwargs.get("tool_call")

        if tool_call is None:
            return AgentFinish(return_values={"output": message.content}, log=message.content)
        else:
            tool_name = tool_call["tool"]
            tool_input = tool_call["arguments"]

            return AgentActionMessageLog(
                tool=tool_name,
                tool_input=tool_input,
                log=f"\nInvoking: `{tool_name}` with `{tool_input}`\n",
                message_log=[message],
            )

