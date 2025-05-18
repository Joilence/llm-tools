import click
import sys
from runpy import run_module
import logging

from ltls.plugins import load_plugins, pm
from ltls import Toolkit
from ltls.types import ToolkitSuite

logger = logging.getLogger(__name__)


@click.group()
def cli():
    """ltls (llm llm tools) offers common tools to build tools for LLM"""
    pass


@cli.command()
def mcp():
    """Start ltls as mcp server with all available tools"""
    from mcp.server.fastmcp import FastMCP

    tool_suite = ToolkitSuite(toolkits=[])

    def register(toolkit: Toolkit):
        tool_suite.add_toolkit(toolkit)

    load_plugins()
    pm.hook.register_toolkit(register=register)

    if not tool_suite.get_tools():
        logger.warning(
            "No toolkits / tools found, install tools by `ltls install <tool_name>`"
        )

    mcp = FastMCP()
    tool_suite.register_mcp(mcp)
    mcp.run()


@cli.command()
@click.argument("packages", nargs=-1, required=False)
@click.option(
    "-U", "--upgrade", is_flag=True, help="Upgrade packages to latest version"
)
@click.option(
    "-e",
    "--editable",
    help="Install a project in editable mode from this path",
)
@click.option(
    "--force-reinstall",
    is_flag=True,
    help="Reinstall all packages even if they are already up-to-date",
)
@click.option(
    "--no-cache-dir",
    is_flag=True,
    help="Disable the cache",
)
def install(packages, upgrade, editable, force_reinstall, no_cache_dir):
    """Install packages into the same environment as ltls"""
    args = ["pip", "install"]
    if upgrade:
        args += ["--upgrade"]
    if editable:
        args += ["--editable", editable]
    if force_reinstall:
        args += ["--force-reinstall"]
    if no_cache_dir:
        args += ["--no-cache-dir"]
    args += list(packages)
    sys.argv = args
    run_module("pip", run_name="__main__")


@cli.command()
@click.option(
    "-d",
    "--detail",
    is_flag=True,
    help="Show detail of the tools",
)
def list(detail: bool):
    """List all available tools"""
    load_plugins()

    suite = ToolkitSuite(toolkits=[])

    def register(toolkit: Toolkit):
        suite.add_toolkit(toolkit)

    pm.hook.register_toolkit(register=register)

    if toolkits := suite.get_toolkits():
        for toolkit in toolkits:
            print(f"Toolkit: {toolkit.name}")
            for tool in toolkit.get_tools():
                print(f"  Tool: {tool.name}")
                if detail:
                    print(f"    Desc: {tool.description}")
                    print(f"    Params: {tool.parameters}")
    else:
        print(
            "No toolkits installed. You can install toolkits by `ltls install <tool_name>`"
        )
