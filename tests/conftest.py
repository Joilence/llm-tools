import json
import ltls
from pydantic import BaseModel, Field
import pytest


class TestToolkit(ltls.Toolkit):
    """A simple toolkit for testing."""

    @ltls.tool_def(
        name="simple_param_tool_name", description="Description from decorator"
    )
    def simple_param_tool(
        self, param: str = Field(description="Annotation to the parameter")
    ):
        """simple param tool description from docstring"""
        return f"Executed with {param}"

    class ComplicatedParam(BaseModel):
        str_param: str = Field(description="A string parameter")
        int_param: int = Field(description="An int parameter")

    @ltls.tool_def(name="complicated_param_tool_name")
    def complicated_param_tool(
        self,
        param: ComplicatedParam,
    ):
        """Annotated parameter tool from docstring"""
        return f"Executed with {param}"


@pytest.fixture
def test_toolkit():
    return TestToolkit()


if __name__ == "__main__":
    """Test output of param by running `uv run -- python3 tests/conftest.py`"""
    toolkit = TestToolkit()
    print(json.dumps(toolkit.as_param(ltls.ToolParamSchema.OPENAI), indent=2))
