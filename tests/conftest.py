import json
from typing import Annotated, Optional
import ltls
from pydantic import Field
import pytest


class TestToolkit(ltls.Toolkit):
    """A simple toolkit for testing."""

    def __init__(self, tools=None):
        super().__init__(tools)

    @ltls.tool_def(name="tool_with_params")
    def tool_with_params(
        self,
        str_param: Annotated[str, Field(description="A required string parameter")],
        int_param: Annotated[
            Optional[int], Field(description="An optional int parameter")
        ] = None,
        bool_param: Annotated[
            bool,
            Field(
                description="An optional boolean parameter with default value to False",
            ),
        ] = False,
    ):
        """Annotated parameter tool from docstring"""
        return f"Executed with {str_param}, {int_param} and {bool_param}"


@ltls.tool_def()
def external_tool_function(
    str_param: Annotated[str, Field(description="A required string parameter")],
    int_param: Annotated[
        Optional[int], Field(description="An optional int parameter")
    ] = None,
    bool_param: Annotated[
        bool,
        Field(
            description="An optional boolean parameter with default value to False",
        ),
    ] = False,
):
    """External tool with annotated parameters"""
    return f"Executed external tool with {str_param}, {int_param} and {bool_param}"


@pytest.fixture
def test_toolkit():
    return TestToolkit()

@pytest.fixture
def external_tool():
    return ltls.Tool.from_function(
        fn=external_tool_function,
        name=external_tool_function._tool_def.name,
        description=external_tool_function._tool_def.description,
    )

@pytest.fixture
def test_toolkit_with_external_tools(external_tool):
    toolkit = TestToolkit()
    toolkit.add_tools(external_tool)
    return toolkit

@pytest.fixture
def empty_test_toolkit():
    return TestToolkit()

@pytest.fixture 
def test_toolkit_constructor_with_external(external_tool):
    return TestToolkit(tools=[external_tool])


if __name__ == "__main__":
    """Test output of param by running `uv run -- python3 tests/conftest.py`"""
    toolkit = TestToolkit()
    print(json.dumps(toolkit.as_param(ltls.ToolParamSchema.OPENAI), indent=2))
