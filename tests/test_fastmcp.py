from typing import Optional, List, Annotated
from fastmcp import FastMCP
from pydantic import BaseModel, Field

mcp = FastMCP(name="test-fastmcp")


@mcp.tool
def int_tool(params: int) -> int:
    """An int tool"""
    return params


@mcp.tool
def optional_int_tool(params: Optional[int]) -> Optional[int]:
    """An optional int tool"""
    return params


@mcp.tool
def default_int_tool(params: int = 1) -> int:
    """An int tool with a default value"""
    return params


@mcp.tool
def list_ints_tool(params: list[int]) -> list[int]:
    """A list of ints tool"""
    return params


@mcp.tool
def optional_list_ints_tool(params: Optional[List[int]]) -> Optional[List[int]]:
    """An optional list of ints tool"""
    return params


@mcp.tool
def default_list_ints_tool(
    params: Annotated[list[int], Field(default_factory=list)],
) -> list[int]:
    """A list of ints tool with a default factory"""
    return params


@mcp.tool
def float_tool(params: float) -> float:
    """A float tool"""
    return params


@mcp.tool
def optional_float_tool(params: Optional[float]) -> Optional[float]:
    """An optional float tool"""
    return params


@mcp.tool
def default_float_tool(params: float = 1.0) -> float:
    """A float tool with a default value"""
    return params


@mcp.tool
def list_floats_tool(params: list[float]) -> list[float]:
    """A list of floats tool"""
    return params


@mcp.tool
def optional_list_floats_tool(params: Optional[List[float]]) -> Optional[List[float]]:
    """An optional list of floats tool"""
    return params


@mcp.tool
def default_list_floats_tool(
    params: Annotated[list[float], Field(default_factory=list)],
) -> list[float]:
    """A list of floats tool with a default factory"""
    return params


@mcp.tool
def str_tool(params: str) -> str:
    """A string tool"""
    return params


@mcp.tool
def optional_str_tool(params: Optional[str]) -> Optional[str]:
    """An optional list of floats tool"""
    return params


@mcp.tool
def default_str_tool(params: str = "default_str_tool") -> str:
    """A string tool with a default value"""
    return params


@mcp.tool
def list_strs_tool(params: list[str]) -> list[str]:
    """A list of strings tool"""
    return params


@mcp.tool
def optional_list_strs_tool(params: Optional[List[str]]) -> Optional[List[str]]:
    """An optional list of strings tool"""
    return params


@mcp.tool
def default_list_strs_tool(
    params: Annotated[list[str], Field(default_factory=list)],
) -> list[str]:
    """A list of strings tool with a default factory"""
    return params


@mcp.tool
def bool_tool(params: bool) -> bool:
    """A bool tool"""
    return params


@mcp.tool
def optional_bool_tool(params: Optional[bool]) -> Optional[bool]:
    """An optional bool tool"""
    return params


@mcp.tool
def default_bool_tool(params: bool = True) -> bool:
    """A bool tool with a default value"""
    return params


@mcp.tool
def list_bools_tool(params: list[bool]) -> list[bool]:
    """A list of bools tool"""
    return params


@mcp.tool
def optional_list_bools_tool(params: Optional[List[bool]]) -> Optional[List[bool]]:
    """An optional list of bools tool"""
    return params


@mcp.tool
def default_list_bools_tool(
    params: Annotated[list[bool], Field(default_factory=list)],
) -> list[bool]:
    """A list of bools tool with a default factory"""
    return params


@mcp.tool
def mix_types_tool(
    str_param: str,
    int_param: int,
    bool_param: bool,
) -> dict:
    """Annotated parameter tool from docstring"""
    return {
        "str_param": str_param,
        "int_param": int_param,
        "bool_param": bool_param,
    }


@mcp.tool
def optional_mix_types_tool(
    optional_str_param: Optional[str] = None,
    optional_int_param: Optional[int] = None,
    optional_bool_param: Optional[bool] = None,
) -> dict:
    """A optional mix types tool"""
    return {
        "optional_str_param": optional_str_param,
        "optional_int_param": optional_int_param,
        "optional_bool_param": optional_bool_param,
    }


@mcp.tool
def default_mix_types_tool(
    default_str_param: str = "default_str_param",
    default_int_param: int = 1,
    default_bool_param: bool = True,
) -> dict:
    """A default mix types tool"""
    return {
        "default_str_param": default_str_param,
        "default_int_param": default_int_param,
        "default_bool_param": default_bool_param,
    }


class StructuredParamsWithInt(BaseModel):
    """A complex parameters dictionary"""

    int_param: int = Field(description="An int parameter")


@mcp.tool
def structured_params_with_int_tool(
    params: Annotated[
        StructuredParamsWithInt, Field(description="A complex parameters dictionary")
    ],
) -> StructuredParamsWithInt:
    """A complex parameters tool"""
    return params


@mcp.tool
def structured_params_with_int_workaround_tool(
    params: Annotated[
        StructuredParamsWithInt | str,
        Field(description="A complex parameters dictionary"),
    ],
) -> StructuredParamsWithInt:
    """A complex parameters tool"""
    if isinstance(params, str):
        params = StructuredParamsWithInt.model_validate_json(params)
    return params


@mcp.tool
def list_structured_params_with_int_tool(
    params: List[StructuredParamsWithInt],
) -> List[StructuredParamsWithInt]:
    """A list of complex parameters tool"""
    return params


class StructuredParamsWithStr(BaseModel):
    """A complex parameters dictionary"""

    str_param: str = Field(default="", description="A string parameter")


@mcp.tool
def structured_params_with_str_tool(
    params: Annotated[
        StructuredParamsWithStr, Field(description="A complex parameters dictionary")
    ],
) -> StructuredParamsWithStr:
    """A complex parameters tool"""
    return params


@mcp.tool
def structured_params_with_str_workaround_tool(
    params: Annotated[
        StructuredParamsWithStr | str,
        Field(description="A complex parameters dictionary"),
    ],
) -> StructuredParamsWithStr:
    """A complex parameters tool"""
    if isinstance(params, str):
        params = StructuredParamsWithStr.model_validate_json(params)
    return params


@mcp.tool
def list_structured_params_with_str_tool(
    params: List[StructuredParamsWithStr],
) -> List[StructuredParamsWithStr]:
    """A list of complex parameters tool"""
    return params


class StructuredParamsWithBool(BaseModel):
    """A complex parameters dictionary"""

    bool_param: bool = Field(default=True, description="A bool parameter")


@mcp.tool
def structured_params_with_bool_tool(
    params: Annotated[
        StructuredParamsWithBool,
        Field(description="A complex parameters dictionary"),
    ],
) -> StructuredParamsWithBool:
    """A complex parameters tool"""
    return params


@mcp.tool
def structured_params_with_bool_workaround_tool(
    params: Annotated[
        StructuredParamsWithBool | str,
        Field(description="A complex parameters dictionary"),
    ],
) -> StructuredParamsWithBool:
    """A complex parameters tool"""
    if isinstance(params, str):
        params = StructuredParamsWithBool.model_validate_json(params)
    return params


@mcp.tool
def list_structured_params_with_bool_tool(
    params: List[StructuredParamsWithBool],
) -> List[StructuredParamsWithBool]:
    """A list of complex parameters tool"""
    return params


class StructuredParamsWithFloat(BaseModel):
    """A complex parameters dictionary"""

    float_param: float = Field(default=1.0, description="A float parameter")


@mcp.tool
def structured_params_with_float_tool(
    params: Annotated[
        StructuredParamsWithFloat, Field(description="A complex parameters dictionary")
    ],
) -> StructuredParamsWithFloat:
    """A complex parameters tool"""
    return params


@mcp.tool
def structured_params_with_float_workaround_tool(
    params: Annotated[
        StructuredParamsWithFloat | str,
        Field(description="A complex parameters dictionary"),
    ],
) -> StructuredParamsWithFloat:
    """A complex parameters tool"""
    if isinstance(params, str):
        params = StructuredParamsWithFloat.model_validate_json(params)
    return params


@mcp.tool
def list_structured_params_with_float_tool(
    params: List[StructuredParamsWithFloat],
) -> List[StructuredParamsWithFloat]:
    """A list of complex parameters tool"""
    return params


if __name__ == "__main__":
    """Test FastMCP server integration."""
    mcp.run()
