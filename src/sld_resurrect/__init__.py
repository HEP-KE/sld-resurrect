try:
    from ._version import __version__ as __version__
except ImportError:  # source tree without build metadata (no editable install yet)
    __version__ = "0.0.0"
