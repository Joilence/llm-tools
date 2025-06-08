import logging
import pluggy

from ltls import hookspecs

logger = logging.getLogger(__name__)


pm = pluggy.PluginManager("ltls")
pm.add_hookspecs(hookspecs)

_loaded = False


def load_plugins():
    global _loaded
    if _loaded:
        return
    _loaded = True

    logger.debug(f"Loading plugins, before: {pm.get_plugins()}")

    import importlib.metadata

    eps = importlib.metadata.entry_points()

    # Handle different Python versions and entry_points structures
    if hasattr(eps, "select"):
        # Python 3.10+ style
        ltls_eps = eps.select(group="ltls")
        logger.debug(f"Found ltls entry points: {list(ltls_eps)}")
    elif isinstance(eps, dict):
        # Older style
        if "ltls" in eps:
            logger.debug(f"Found ltls entry points: {eps['ltls']}")
    else:
        # Python 3.12+ style with different structure
        for group in eps.groups:
            if group == "ltls":
                logger.debug("Found ltls entry points in group")

    pm.load_setuptools_entrypoints("ltls")

    logger.debug(f"Loaded: {pm.get_plugins()}")
