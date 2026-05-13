# gfn/constants.py
# Universal constants for the GFN framework.
# This file re-exports constants from the G-SSM and ISN realizations for top-level access.

try:
    from .realizations.gssm.constants import *
except ImportError:
    pass

# Add ISN specific constants if any are added in the future
