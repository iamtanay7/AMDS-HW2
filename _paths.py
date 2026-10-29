"""Make the core simulator (model.py, physics.py) importable.

Looks for it in ./msd_jax (student package) or ../msd_jax (development tree).
"""
import os
import sys

import jax

jax.config.update("jax_enable_x64", True)   # HW2 runs in double precision

_here = os.path.dirname(os.path.abspath(__file__))
for _candidate in (os.path.join(_here, "msd_jax"), os.path.join(_here, "..", "msd_jax")):
    if os.path.isfile(os.path.join(_candidate, "physics.py")):
        if _candidate not in sys.path:
            sys.path.insert(0, _candidate)
        break
else:
    raise ImportError("Could not find the msd_jax simulator folder next to hw2/.")
