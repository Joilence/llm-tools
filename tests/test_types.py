import pytest
from ltls.types import ToolParamSchema, Tool, Toolkit, ToolkitSuite, tool_def
from tests.conftest import external_tool_function
from mcp.server.fastmcp import FastMCP
import json


class TestToolParamMode:
    def test_enum_values(self):
        """Test that enum values are correctly set."""
        assert ToolParamSchema.ANTHROPIC == "anthropic"
        assert ToolParamSchema.OPENAI == "openai"
        assert ToolParamSchema.MCP == "mcp"




class TestTool:
    def test_as_param_openai(self, external_tool):
        """Test OpenAI parameter format with real tool."""
        param = external_tool.as_param(ToolParamSchema.OPENAI)
        assert param["type"] == "function"
        assert param["function"]["name"] == "external_tool_function"
        assert param["function"]["description"] == "External tool with annotated parameters"
        assert "parameters" in param["function"]

    def test_as_param_anthropic(self, external_tool):
        """Test Anthropic parameter format with real tool."""
        param = external_tool.as_param(ToolParamSchema.ANTHROPIC)
        assert param["name"] == "external_tool_function"
        assert param["description"] == "External tool with annotated parameters"
        assert "input_schema" in param

    def test_as_param_unsupported(self, external_tool):
        """Test that using an unsupported mode raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            external_tool.as_param(mode=ToolParamSchema.MCP)


class TestToolkit:
    def test_name_returns_class_name(self, test_toolkit):
        """Test that Toolkit.name returns the class name."""
        assert test_toolkit.name == "TestToolkit"

    def test_get_tools_finds_decorated_methods(self, test_toolkit):
        """Test that Toolkit.get_tools finds methods decorated with tool_def."""
        tools = test_toolkit.get_tools()
        assert len(tools) == 1
        assert tools[0].name == "tool_with_params"
        assert tools[0].description == "Annotated parameter tool from docstring"

    def test_tool_names(self, test_toolkit):
        """Test that Toolkit.tool_names returns the names of all tools."""
        assert test_toolkit.tool_names == ["tool_with_params"]

    def test_as_param_openai(self, test_toolkit):
        """Test that Toolkit.as_param returns OpenAI format parameters."""
        params = test_toolkit.as_param(mode=ToolParamSchema.OPENAI)
        assert len(params) == 1
        assert params[0]["type"] == "function"
        assert params[0]["function"]["name"] == "tool_with_params"

    def test_as_param_anthropic(self, test_toolkit):
        """Test that Toolkit.as_param returns Anthropic format parameters."""
        params = test_toolkit.as_param(mode=ToolParamSchema.ANTHROPIC)
        assert len(params) == 1
        assert params[0]["name"] == "tool_with_params"


class TestToolkitExternalTools:
    def test_add_tools_single_tool(self, test_toolkit, external_tool):
        """Test adding a single external tool."""
        test_toolkit.add_tools(external_tool)
        tools = test_toolkit.get_tools()
        assert len(tools) == 2  # 1 member + 1 external
        tool_names = [t.name for t in tools]
        assert "tool_with_params" in tool_names
        assert "external_tool_function" in tool_names

    def test_add_tools_list_of_tools(self, test_toolkit, external_tool):
        """Test adding a list of external tools."""
        test_toolkit.add_tools([external_tool])
        assert len(test_toolkit.get_tools()) == 2

    def test_constructor_with_external_tools(self, test_toolkit_constructor_with_external):
        """Test creating toolkit with external tools in constructor."""
        toolkit = test_toolkit_constructor_with_external
        assert len(toolkit.get_tools()) == 2
        tool_names = [t.name for t in toolkit.get_tools()]
        assert "tool_with_params" in tool_names
        assert "external_tool_function" in tool_names

    def test_external_tools_override_member_tools(self, test_toolkit):
        """Test that external tools override member tools with same name."""
        override_tool = Tool.from_function(
            fn=external_tool_function,
            name="tool_with_params",  # Same name as member tool
            description="Override description",
        )
        test_toolkit.add_tools(override_tool)
        tools = test_toolkit.get_tools()
        assert len(tools) == 1  # Same name, so override
        assert tools[0].description == "Override description"

    def test_tool_names_includes_external(self, test_toolkit_with_external_tools):
        """Test that tool_names includes external tools."""
        names = test_toolkit_with_external_tools.tool_names
        assert len(names) == 2
        assert "tool_with_params" in names
        assert "external_tool_function" in names


@pytest.fixture
def another_test_toolkit():
    class AnotherTestToolkit(Toolkit):
        """Another simple toolkit for testing."""

        @tool_def(name="another_tool", description="Another test tool")
        def another_tool(self, param1: str):
            """Another tool for testing."""
            return f"Another executed with {param1}"

    return AnotherTestToolkit()


class TestToolkitSuite:
    def test_init(self, test_toolkit, another_test_toolkit):
        """Test that ToolkitSuite initializes with a list of toolkits."""
        suite = ToolkitSuite([test_toolkit, another_test_toolkit])
        assert len(suite._toolkits) == 2

    def test_tool_names(self, test_toolkit, another_test_toolkit):
        """Test that ToolkitSuite.tool_names returns all tool names from all toolkits."""
        suite = ToolkitSuite([test_toolkit, another_test_toolkit])
        names = suite.tool_names
        assert "tool_with_params" in names
        assert "another_tool" in names
        assert len(names) == 2

    def test_as_param(self, test_toolkit, another_test_toolkit):
        """Test that ToolkitSuite.as_param returns all tool parameters from all toolkits."""
        suite = ToolkitSuite([test_toolkit, another_test_toolkit])
        params = suite.as_param(mode=ToolParamSchema.OPENAI)
        assert len(params) == 2
        tool_names = [p["function"]["name"] for p in params]
        assert "tool_with_params" in tool_names
        assert "another_tool" in tool_names

    def test_add_toolkit(self, test_toolkit, another_test_toolkit):
        """Test that ToolkitSuite.add_toolkit adds a toolkit to the suite."""
        suite = ToolkitSuite([test_toolkit])
        suite.add_toolkit(another_test_toolkit)
        assert len(suite._toolkits) == 2
        assert suite._toolkits[1] == another_test_toolkit

    def test_get_toolkits(self, test_toolkit, another_test_toolkit):
        """Test that ToolkitSuite.get_toolkits returns all toolkits."""
        suite = ToolkitSuite([test_toolkit, another_test_toolkit])
        toolkits = suite.get_toolkits()
        assert len(toolkits) == 2
        assert toolkits[0] == test_toolkit
        assert toolkits[1] == another_test_toolkit

    def test_get_tools(self, test_toolkit, another_test_toolkit):
        """Test that ToolkitSuite.get_tools returns all tools from all toolkits."""
        suite = ToolkitSuite([test_toolkit, another_test_toolkit])
        tools = suite.get_tools()
        assert len(tools) == 2
        tool_names = [t.name for t in tools]
        assert "tool_with_params" in tool_names
        assert "another_tool" in tool_names

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self):
        """Test that ToolkitSuite.execute_tool raises ValueError when the tool is not found."""
        suite = ToolkitSuite([])
        with pytest.raises(
            ValueError, match="Tool with name non_existent_tool not found"
        ):
            await suite.execute_tool("non_existent_tool", {})


class TestToolDef:
    def test_tool_def_decorator(self):
        """Test that tool_def decorator attaches _tool_def attribute to the function."""

        @tool_def(name="decorated_function", description="A decorated function")
        def decorated_function():
            pass

        # Use hasattr first to avoid attribute errors
        assert hasattr(decorated_function, "_tool_def")
        tool_def_attr = getattr(decorated_function, "_tool_def")
        assert tool_def_attr.name == "decorated_function"
        assert tool_def_attr.description == "A decorated function"

    def test_tool_def_uses_docstring_if_no_description(self):
        """Test that tool_def uses the function's docstring if no description is provided."""

        @tool_def(name="docstring_function")
        def docstring_function():
            """This is the docstring."""
            pass

        # Use hasattr first to avoid attribute errors
        assert hasattr(docstring_function, "_tool_def")
        tool_def_attr = getattr(docstring_function, "_tool_def")
        assert tool_def_attr.description == "This is the docstring."

    def test_tool_def_empty_if_no_description_or_docstring(self):
        """Test that tool_def sets empty description if neither description nor docstring is provided."""

        @tool_def(name="empty_function")
        def empty_function():
            pass

        # Use hasattr first to avoid attribute errors
        assert hasattr(empty_function, "_tool_def")
        tool_def_attr = getattr(empty_function, "_tool_def")
        assert tool_def_attr.description == ""
