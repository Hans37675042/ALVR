"""roomd entry point: python -m roomd [--relay-port 9944] [--plugin-port 9945]
[--tap-in FILE.rktap [--tap-speed S] [--tap-loop N]] [--record FILE.rktap]
[--fusion noop|module:factory] [--room-id stage] [--seconds S] [--dump-model FILE.json]"""

import argparse
import signal
import sys
import threading
import time
from pathlib import Path

from . import __version__
from . import model as M
from .io.plugin_server import PluginServer
from .io.relay import RelayInbox, RelayViewer
from .io.tapsource import TapSource
from .service import RoomService
from .sink import load_sink


class Roomd:
    """Wires the input (9944 viewer or a tap file), the service and the 9945 server."""

    def __init__(self, relay_port=9944, plugin_port=9945, sink=None, tap_in=None, tap_speed=1.0,
                 tap_loop=1, record=None, room_id="stage", host="127.0.0.1", log=print):
        self.log = log
        self.inbox = RelayInbox()
        self.plugin = PluginServer(plugin_port, host=host, log=log)
        self.relay = None
        self.tap = None
        if tap_in is not None:
            self.tap = TapSource(tap_in, self.inbox, speed=tap_speed, loop=tap_loop, log=log)
        else:
            self.relay = RelayViewer(relay_port, self.inbox, host=host, record=record, log=log)
        self.sink = sink if sink is not None else load_sink("noop")
        self.service = RoomService(self.sink, self.plugin, self.relay, room_id=room_id, log=log)

    def start(self):
        self.plugin.start()
        if self.relay is not None:
            self.relay.start()
        if self.tap is not None:
            self.tap.start()

    def run(self, stop, until=None):
        if until is None and self.tap is not None:
            until = self.tap.finished
        self.service.run(self.inbox, stop, until=until)

    def close(self):
        if self.tap is not None:
            self.tap.close()
        if self.relay is not None:
            self.relay.close()
        self.plugin.close()
        self.sink.close()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m roomd", description="realkk room service %s" % __version__)
    ap.add_argument("--relay-port", type=int, default=9944, help="ALVR relay viewer port")
    ap.add_argument("--plugin-port", type=int, default=9945, help="KKS plugin server port")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--tap-in", type=Path, help="read this .rktap instead of listening on the relay port")
    ap.add_argument("--tap-speed", type=float, default=1.0, help="tap pacing multiplier, 0 = as fast as possible")
    ap.add_argument("--tap-loop", type=int, default=1, help="tap passes, 0 = forever")
    ap.add_argument("--record", type=Path, help="record every inbound relay message to this .rktap")
    ap.add_argument("--fusion", default="noop", help="FusionSink: noop or module:factory")
    ap.add_argument("--room-id", default="stage")
    ap.add_argument("--seconds", type=float, default=0, help="stop after S seconds, 0 = run until Ctrl+C")
    ap.add_argument("--dump-model", type=Path, help="write the last RoomModel JSON here on exit")
    args = ap.parse_args(argv)
    if args.tap_in and args.record:
        ap.error("--record only applies to the live relay, not --tap-in")

    app = Roomd(args.relay_port, args.plugin_port, load_sink(args.fusion), tap_in=args.tap_in,
                tap_speed=args.tap_speed, tap_loop=args.tap_loop, record=args.record, room_id=args.room_id,
                host=args.host)
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    if args.seconds > 0:
        threading.Timer(args.seconds, stop.set).start()
    app.start()
    try:
        app.run(stop)
    finally:
        app.close()
        st = app.service.stats
        print("roomd: %d depth frames integrated, %d fake dropped, %d skipped (slow consumer), "
              "%d snapshots, model revision %d"
              % (st.depth_frames, st.fake_frames, app.inbox.skipped_depth, st.snapshots, app.service.revision))
        if args.dump_model and app.service.model is not None:
            args.dump_model.write_text(M.room_to_json(app.service.model, indent=2), encoding="utf-8")
            print("roomd: model written to", args.dump_model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
