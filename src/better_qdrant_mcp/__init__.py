from .tools import run  # re-export entrypoint

__all__ = ["run", "hello"]


def hello() -> str:
    return "Hello from better-qdrant-mcp!"
