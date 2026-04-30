"""Plot helpers for the 2AFC drug adapter.

The drug task uses the same behavioral columns and binary choice geometry as
the base Alexis 2AFC task, so the task-owned plot surface is re-exported.
"""

from src.plots import two_afc as _two_afc

for _name in dir(_two_afc):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_two_afc, _name)

__all__ = [_name for _name in globals() if not _name.startswith("_")]
