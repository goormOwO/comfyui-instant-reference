try:
    from .src.extension import comfy_entrypoint
    from .src.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    from .src import server as _server
except ImportError:
    from src.extension import comfy_entrypoint
    from src.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    import src.server as _server

WEB_DIRECTORY = "./web/js"

__all__ = [
    "WEB_DIRECTORY",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "comfy_entrypoint",
]
