import pathlib
import sys

# tools/realkk/roomd on sys.path so `import roomd.semantics` works without a pyproject.
ROOMD_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOMD_ROOT) not in sys.path:
    sys.path.insert(0, str(ROOMD_ROOT))
