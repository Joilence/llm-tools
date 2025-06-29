import json
import functools
from dataclasses import dataclass

from typing import (
    Annotated,
    Callable,
    Optional,
    Any,
    Literal,
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
from openai.types.chat.chat_completion_tool_param import (
    ChatCompletionToolParam as OpenAIToolParam,
)
from openai.types.shared_params import FunctionDefinition
from anthropic.types.tool_param import ToolParam as AnthropicToolParam

logger = logging.getLogger(__name__)


class ToolParamSchema(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    MCP = "mcp"


UnionToolParam = Union[OpenAIToolParam, AnthropicToolParam]


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
                    function=FunctionDefinition(
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

    def __init__(self, tools: Optional[list[Tool]] = None):
        self._external_tools: list[Tool] = tools or []

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

    @functools.cached_property
    def _member_tools(self) -> list[Tool]:
        """Automatically pickup all tools in the class, which:
        - are functions
        - are decorated with `tool_def`
        """
        tool_functions = []
        for attr_name in dir(self):
            # First check if the attribute on the class is a property or cached_property
            class_attr = getattr(type(self), attr_name, None)
            if isinstance(class_attr, (property, functools.cached_property)):
                # Skip properties and cached_properties
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

    def get_tools(self) -> list[Tool]:
        """Get all tools: member tools + external tools (external overrides member)"""
        tools_dict = {tool.name: tool for tool in self._member_tools}
        for tool in self._external_tools:
            tools_dict[tool.name] = tool  # External tools override member tools
        return list(tools_dict.values())

    def add_tools(self, tools: Union[Tool, list[Tool]]) -> None:
        """Add external tools to the toolkit"""
        if isinstance(tools, Tool):
            tools = [tools]
        self._external_tools.extend(tools)

    def register_mcp(self, mcp: FastMCP):
        """Register the tools in the toolkit with the given MCP instance."""

        for tool in self.get_tools():
            mcp.add_tool(
                tool.fn,
                name=tool.name,
                description=tool.description,
            )

    async def execute_tool(self, name: str, arguments: dict[str, Any]):
        """Execute the tool with the given name and arguments"""

        # Find the tool
        tool = next((tool for tool in self.get_tools() if tool.name == name), None)
        if not tool:
            raise ValueError(f"Tool with name {name} not found")

        # Try to parse the arguments
        if not isinstance(arguments, dict):
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except Exception:
                    logger.warning(
                        f"Tool {name} received non-dict arguments and failed to parse as JSON: {arguments}"
                    )
            else:
                logger.warning(
                    f"Tool {name} received non-dict and non-string arguments: {arguments}"
                )

        # Log tool usage
        logger.info(
            f"Executing tool: {name} with arguments:"
            f"\n{json.dumps(arguments, indent=2) if isinstance(arguments, dict) else str(arguments)}"
        )

        return await tool.run(arguments)


class ToolkitSuite(ABC):
    """A collection of toolkits for different purposes"""

    _toolkits: list[Toolkit]

    @property
    def tool_names(self) -> list[str]:
        result = []
        for toolkit in self._toolkits:
            result.extend(toolkit.tool_names)
        return result

    def __init__(self, toolkits: Optional[list[Toolkit]] = None):
        self._toolkits = toolkits or []

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

    async def execute_tool(self, name: str, arguments: dict[str, Any]):
        """Execute the tool with the given name and arguments"""
        for toolkit in self._toolkits:
            if name in toolkit.tool_names:
                return await toolkit.execute_tool(name, arguments)
        raise ValueError(f"Tool with name {name} not found")


@dataclass(frozen=True)
class ToolDef:
    fn: Callable
    name: str
    description: str


def tool_def(
    name: Annotated[
        Optional[str], "Name of the tool, by default will be the name of the function"
    ] = None,
    description: Annotated[
        Optional[str],
        "Description of the tool, by default will be the docstring of the function",
    ] = None,
):
    """Decorator that attaches tool metadata to a method."""

    def deco(fn: Callable) -> Callable:
        setattr(
            fn,
            "_tool_def",
            ToolDef(
                fn=fn,
                name=name or fn.__name__,
                description=description or (fn.__doc__ or "").strip(),
            ),
        )
        return fn

    return deco
