"""Access to the standalone relay tools in tools/realkk (rktap, depth_listener, tap_inspect).

Those scripts stay standard-library-only and runnable on their own; roomd imports them
instead of keeping a second copy of the tap format and the depth header parser.
"""

import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[2]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

# keep tools/realkk free of untracked __pycache__ directories
_dont_write = sys.dont_write_bytecode
sys.dont_write_bytecode = True
try:
    import depth_listener  # noqa: E402
    import rktap  # noqa: E402
    import tap_inspect  # noqa: E402
finally:
    sys.dont_write_bytecode = _dont_write

__all__ = ["TOOLS_DIR", "depth_listener", "rktap", "tap_inspect"]
