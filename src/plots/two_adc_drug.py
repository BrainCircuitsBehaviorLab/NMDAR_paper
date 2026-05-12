"""Plot helpers for the 2ADC drug adapter."""

from src.plots import two_adc as _two_adc

for _name in dir(_two_adc):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_two_adc, _name)

__all__ = [_name for _name in globals() if not _name.startswith("_")]
