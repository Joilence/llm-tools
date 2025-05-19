import json
from dataclasses import dataclass
from typing import Callable, List, Dict, Any, cast
from abc import ABC
from mcp.server.fastmcp.tools import Tool as FastMCPTool
from mcp.server.fastmcp import FastMCP
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ToolParamMode(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    MCP = "mcp"


class Tool(FastMCPTool):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def as_param(self, mode: ToolParamMode = ToolParamMode.OPENAI) -> Dict[str, Any]:
        """Convert the tool to a schema that can be used as parameters to pass to, e.g. LLM call"""
        match mode:
            case ToolParamMode.OPENAI:
                # See: https://platform.openai.com/docs/api-reference/chat/create#chat-create-tools
                return {
                    "type": "function",
                    "function": {
                        "name": self.name,
                        "description": self.description,
                        "parameters": self.parameters,
                    },
                }
            case ToolParamMode.ANTHROPIC:
                # See: https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview
                return {
                    "name": self.name,
                    "description": self.description,
                    "input_schema": self.parameters,
                }
            case _:
                raise NotImplementedError(f"unsupported mode: {mode}")


class Toolkit(ABC):
    """A set of tools that supposed to work together"""

    @property
    def name(self) -> str:
        if hasattr(self, "name"):
            return self.name
        return self.__class__.__name__

    @property
    def tool_names(self) -> List[str]:
        return [tool.name for tool in self.get_tools()]

    def as_params(
        self, mode: ToolParamMode = ToolParamMode.OPENAI
    ) -> List[Dict[str, Any]]:
        """Convert the tools in toolkit to a schema that can be used as parameters to pass to, e.g. LLM call"""
        tools = self.get_tools()
        return [tool.as_param(mode) for tool in tools]

    def get_tools(self) -> list[Tool]:
        """Automatically pickup all tools in the class, which:
        - are functions
        - are decorated with `tool_def`
        """
        tool_functions = [
            getattr(self, attr_name)
            for attr_name in dir(self)
            if callable(getattr(self, attr_name))
            and hasattr(getattr(self, attr_name), "_tool_def")
        ]
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

    _toolkits: List[Toolkit]

    @property
    def tool_names(self) -> List[str]:
        return [tool.name for toolkit in self._toolkits for tool in toolkit.tool_names]

    def __init__(self, toolkits: List[Toolkit]):
        self._toolkits = toolkits

    def as_params(
        self, mode: ToolParamMode = ToolParamMode.OPENAI
    ) -> List[Dict[str, Any]]:
        """Convert the tools in toolkits to a schema that can be used as parameters to pass to, e.g. LLM call"""
        params = []
        for toolkit in self._toolkits:
            params.extend(toolkit.as_params(mode))
        return params

    def register_mcp(self, mcp: FastMCP):
        """Register the tools in toolkits for the given MCP instance"""
        for toolkit in self._toolkits:
            toolkit.register_mcp(mcp)

    def add_toolkit(self, toolkit: Toolkit):
        self._toolkits.append(toolkit)

    def get_toolkits(self) -> List[Toolkit]:
        return self._toolkits

    def get_tools(self) -> List[Tool]:
        """Get all tools from all toolkits"""
        return [tool for toolkit in self._toolkits for tool in toolkit.get_tools()]

    async def execute_tool(self, name: str, arguments: Dict[str, Any] | str):
        """Execute the tool with the given name and arguments"""
        for tool in self.get_tools():
            if tool.name == name:
                logger.info(
                    f"Executing tool: {name} with arguments:\n{json.dumps(arguments, indent=2)}"
                )
                processed_arguments: Dict[str, Any]
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
