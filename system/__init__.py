import os, glob
from importlib import import_module
from pathlib import Path
"""
Import all modules starts_with 'startWith' inside the package 
"""
modules = glob.glob(os.path.join(os.path.dirname(__file__), "*.py"))
startWith = "system"

__all__ = [
    os.path.basename(f)[:-3] for f in modules
    if not f.endswith("__init__.py") and Path(f).name.startswith(startWith)
]

for each in __all__:
    import_module(f".{each}", __package__)

from .custom_logging import CustomLogger, LatestCheckpoint