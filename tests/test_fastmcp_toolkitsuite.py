from typing import Optional
import ltls
from pydantic import BaseModel
from fastmcp import FastMCP


class SimpleToolkit(ltls.Toolkit):
    @ltls.tool_def()
    def optional_int(self, optional_int_param: Optional[int]) -> Optional[int]:
        return optional_int_param


class ComplexToolkit(ltls.Toolkit):
    class ComplexParams(BaseModel):
        x: int
        y: int

    @ltls.tool_def()
    def complex_params(self, params: ComplexParams) -> ComplexParams:
        return params


toolkit_suite = ltls.ToolkitSuite([SimpleToolkit(), ComplexToolkit()])

if __name__ == "__main__":
    mcp = FastMCP("test-fastmcp-toolkitsuite")
    toolkit_suite.register_mcp(mcp)
    mcp.run()
