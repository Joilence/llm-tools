# TLBG Toolkit Plugin

The TLBG Toolkit plugin allows for extending the toolkit collection through plugins.

## Install a plugin

```bash
ltls install <plugin_name>
```

## Develop a plugin

```python
import ltls


@ltls.hookimpl
def register_toolkit(register):
    register(MyToolkit())

class MyToolkit(ltls.Toolkit):

    def _echo_helper(self, content: str) -> str:
        return f"echo: {content}"

    @ltls.tool_def(
        name="echo",
        description="Echo a message",
    )
    def echo(self, content: str) -> str:
        return self._echo_helper(content)

    @ltls.tool_def(
        name="echo_twice",
        description="Echo a message twice",
    )
    def echo_twice(self, content: str) -> str:
        return self._echo_helper(content) + self._echo_helper(content)
```

You can install the plugin with:

```bash
ltls install -e .
# or
ltls install -e <path_to_plugin_dir>

# check if the plugin is installed
ltls plugins
```
