import sys
from pathlib import Path

# tools/realkk is shared with stdlib-only scripts; leave no __pycache__ there
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
# tools/realkk (rktap, depth_listener, tap_replay, tap_inspect) and its tests/synth.py
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / "tests"))
