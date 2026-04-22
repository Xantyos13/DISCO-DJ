# cosmology_loader.py

import os

use_nu = os.environ.get("USE_MASSIVE_NEUTRINOS") == "0"

if use_nu:
    from .cosmology_nu import Cosmology
else:
    from .cosmology import Cosmology