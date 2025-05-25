# ltls (llm tools)

- A minimal library to let you focus on building LLM tools. It automatically enables the tools to be used by LLM calls, MCP clients, etc.
- A minimal CLI to install tools and run as MCP server.

## Usage

### As Package

<details>

<summary>Develop tools</summary>

```python
import ltls

class MinimalFsToolkit(ltls.Toolkit):

    @ltls.tool_def(
        name="read_file",
        description="Read content from a file"
    )
    def read_file(self, file_path: str):
        return open(file_path).read()

    @ltls.tool_def(
        name="write_file",
        description="Write content to a file"
    )
    def write_file(self, file_path: str, content: str):
        with open(file_path, "w") as f:
            f.write(content)

min_fs_toolkit = MinimalFsToolkit()
```

</details>

<details>

<summary>Use for LLM calls</summary>

```python
import litellm
from somewhere import min_fs_toolkit

response = litellm.completion(
    model='openai/gpt-4o-mini',
    messages=[
        { "role": "user",  "content": "hi there, what tools do you have?"}
    ],
    tools=min_fs_toolkit.as_params('openai')
)
```

</details>

<details>

<summary>Use in MCP server</summary>

```python
from mcp import FastMCP
from somewhere import min_fs_toolkit

# Use in mcp
mcp = FastMCP()
min_fs_toolkit.register(mcp)
mcp.run()
```

</details>

### As CLI

<details>

<summary>Standalone as MCP server</summary>

```bash
uv tool install git+https://github.com/Joilence/llm-tools

# Check commands
ltls --help

# Install a tool (uses pip install under the hood)
ltls install <package_name_or_git_url>

# List installed tools
ltls list
# List installed tools with details (description and parameters)
ltls list --detail

# Start MCP server with installed tools
ltls mcp
```

</details>

<details>

<summary>Use with <a href="https://github.com/simonw/llm">simonw/llm</a> (🚧WIP)</summary>

```bash
# install ltls
llm install llm-ltls

# install tools
ltls install <tool_name>

# use tools
llm --ltls "what tools do you have?"
```

</details>

## Contributing

Feel free to submit issues and PRs.

## Credits

Heavily inspired by [simonw/llm](https://github.com/simonw/llm)
