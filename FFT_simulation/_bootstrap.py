# -*- coding: utf-8 -*-
"""Make the repository importable from inside FFT_simulation/.

The solver package (`fg`) lives here, while its support modules
(`simulation_config`, `run_case`'s plotting/metadata helpers) live at the
repository root. Importing this module first puts both directories on
``sys.path``, so every entry point below FFT_simulation/ works regardless of
the working directory it was started from.

    import _bootstrap  # noqa: F401  (must precede the fg / root imports)
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from project_paths import ensure_import_paths  # noqa: E402

ensure_import_paths()
