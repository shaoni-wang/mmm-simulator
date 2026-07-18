# data/__init__.py
# Makes the data directory a Python package

from .generate_data import generate_marketing_data, generate_abm_simulation_data

__all__ = [
    'generate_marketing_data',
    'generate_abm_simulation_data'
]