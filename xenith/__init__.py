"""
See the README for detailed documentation and examples.
"""
from pkg_resources import get_distribution, DistributionNotFound

from . import convert
from .models import from_percolator, load_model, new_model
from .dataset import load_psms

__version__ = get_distribution(__name__).version
