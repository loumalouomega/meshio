try:
    from importlib import metadata
except ImportError:
    try:
        import importlib_metadata as metadata
    except ImportError:
        metadata = None


if metadata is not None:
    try:
        __version__ = metadata.version("meshio")
    except Exception:
        __version__ = "unknown"
else:
    __version__ = "unknown"
