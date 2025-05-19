import json
from dataclasses import dataclass
from typing import (
    Callable,
    Any,
    Literal,
    TypedDict,
    Union,
    cast,
    overload,
    Sequence,
)
from abc import ABC
from mcp.server.fastmcp.tools import Tool as FastMCPTool
from mcp.server.fastmcp import FastMCP
from enum import Enum
import logging
import anthropic

logger = logging.getLogger(__name__)


class ToolParamSchema(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    MCP = "mcp"


AnthropicToolParam = anthropic.types.ToolParam


class OpenAIFunctionDef(TypedDict):
    name: str
    description: str
    parameters: dict[str, Any]


# Define TypedDict for OpenAI's tool format
class OpenAIToolParam(TypedDict):
    type: Literal["function"]
    function: OpenAIFunctionDef


UnionToolParam = Union[OpenAIToolParam, anthropic.types.ToolParam]


class Tool(FastMCPTool):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @overload
    def as_param(
        self, mode: Literal[ToolParamSchema.ANTHROPIC]
    ) -> anthropic.types.ToolParam:
        ...

    @overload
    def as_param(self, mode: Literal[ToolParamSchema.OPENAI]) -> OpenAIToolParam:
        ...

    @overload
    def as_param(
        self, mode: ToolParamSchema = ToolParamSchema.OPENAI
    ) -> UnionToolParam:
        ...

    def as_param(
        self, mode: ToolParamSchema = ToolParamSchema.OPENAI
    ) -> UnionToolParam:
        """Convert the tool to a schema that can be used as parameters to pass to, e.g. LLM call"""
        match mode:
            case ToolParamSchema.OPENAI:
                # See: https://platform.openai.com/docs/api-reference/chat/create#chat-create-tools
                return OpenAIToolParam(
                    type="function",
                    function=OpenAIFunctionDef(
                        name=self.name,
                        description=self.description,
                        parameters=self.parameters,
                    ),
                )
            case ToolParamSchema.ANTHROPIC:
                # See: https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview
                # Create ToolParam using Anthropic's API directly
                return anthropic.types.ToolParam(
                    name=self.name,
                    description=self.description,
                    input_schema=self.parameters,
                )
            case _:
                raise NotImplementedError(f"unsupported mode: {mode}")


class Toolkit(ABC):
    """A set of tools that supposed to work together"""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def tool_names(self) -> list[str]:
        return [tool.name for tool in self.get_tools()]

    @overload
    def as_param(
        self, mode: Literal[ToolParamSchema.ANTHROPIC]
    ) -> Sequence[anthropic.types.ToolParam]:
        ...

    @overload
    def as_param(
        self, mode: Literal[ToolParamSchema.OPENAI]
    ) -> Sequence[OpenAIToolParam]:
        ...

    @overload
    def as_param(
        self, mode: ToolParamSchema = ToolParamSchema.OPENAI
    ) -> Sequence[UnionToolParam]:
        ...

    def as_param(
        self, mode: ToolParamSchema = ToolParamSchema.OPENAI
    ) -> Sequence[UnionToolParam]:
        """Convert the tools in toolkit to a schema that can be used as parameters to pass to, e.g. LLM call"""
        tools = self.get_tools()
        return [tool.as_param(mode) for tool in tools]

    def get_tools(self) -> list[Tool]:
        """Automatically pickup all tools in the class, which:
        - are functions
        - are decorated with `tool_def`
        """
        tool_functions = []
        for attr_name in dir(self):
            # First check if the attribute on the class is a property
            class_attr = getattr(type(self), attr_name, None)
            if isinstance(class_attr, property):
                # Skip properties
                continue

            # Then get the instance attribute
            attr = getattr(self, attr_name)
            if callable(attr) and hasattr(attr, "_tool_def"):
                tool_functions.append(attr)

        return [
            cast(
                Tool,
                Tool.from_function(
                    fn=tool_function,
                    name=tool_function._tool_def.name,
                    description=tool_function._tool_def.description,
                ),
            )
            for tool_function in tool_functions
        ]

    def register_mcp(self, mcp: FastMCP):
        """Register the tools in the toolkit with the given MCP instance."""

        for tool in self.get_tools():
            mcp.add_tool(
                tool.fn,
                name=tool.name,
                description=tool.description,
            )


class ToolkitSuite(ABC):
    """A collection of toolkits for different purposes"""

    _toolkits: list[Toolkit]

    @property
    def tool_names(self) -> list[str]:
        result = []
        for toolkit in self._toolkits:
            result.extend(toolkit.tool_names)
        return result

    def __init__(self, toolkits: list[Toolkit]):
        self._toolkits = toolkits

    @overload
    def as_param(
        self, mode: Literal[ToolParamSchema.ANTHROPIC]
    ) -> Sequence[anthropic.types.ToolParam]:
        ...

    @overload
    def as_param(
        self, mode: Literal[ToolParamSchema.OPENAI]
    ) -> Sequence[OpenAIToolParam]:
        ...

    @overload
    def as_param(
        self, mode: ToolParamSchema = ToolParamSchema.OPENAI
    ) -> Sequence[UnionToolParam]:
        ...

    def as_param(
        self, mode: ToolParamSchema = ToolParamSchema.OPENAI
    ) -> Sequence[UnionToolParam]:
        """Convert the tools in toolkits to a schema that can be used as parameters to pass to, e.g. LLM call"""
        params = []
        for toolkit in self._toolkits:
            params.extend(toolkit.as_param(mode))
        return params

    def register_mcp(self, mcp: FastMCP):
        """Register the tools in toolkits for the given MCP instance"""
        for toolkit in self._toolkits:
            toolkit.register_mcp(mcp)

    def add_toolkit(self, toolkit: Toolkit):
        self._toolkits.append(toolkit)

    def get_toolkits(self) -> list[Toolkit]:
        return self._toolkits

    def get_tools(self) -> list[Tool]:
        """Get all tools from all toolkits"""
        return [tool for toolkit in self._toolkits for tool in toolkit.get_tools()]

    async def execute_tool(self, name: str, arguments: dict[str, Any] | str):
        """Execute the tool with the given name and arguments"""
        for tool in self.get_tools():
            if tool.name == name:
                logger.info(
                    f"Executing tool: {name} with arguments:\n{json.dumps(arguments, indent=2)}"
                )
                processed_arguments: dict[str, Any]
                try:
                    if isinstance(arguments, str):
                        processed_arguments = json.loads(arguments)
                    else:
                        processed_arguments = arguments
                except Exception as e:
                    logger.error(f"Error parsing arguments: {e}")
                    return f"Error Parsing Arguments: {e}"
                try:
                    return await tool.run(processed_arguments)
                except Exception as e:
                    logger.error(f"Error executing tool: {e}")
                    return f"Error Executing Tool: {e}"
        logger.error(f"Tool with name {name} not found")
        return f"Error Tool Not Found: {name}"


@dataclass(frozen=True)
class ToolDef:
    fn: Callable
    name: str
    description: str


def tool_def(name: str, description: str | None = None):
    """Decorator that attaches tool metadata to a method."""

    def deco(fn: Callable) -> Callable:
        setattr(
            fn,
            "_tool_def",
            ToolDef(
                fn=fn, name=name, description=description or (fn.__doc__ or "").strip()
            ),
        )
        return fn

    return deco
