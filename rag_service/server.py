"""Run the search service:  python -m rag_service.server [--port N]

Listens on 127.0.0.1 only. Port priority: --port, then RAG_PORT, then 2190.
"""
import argparse
import sys
import threading

from rag_service.config import Config, ConfigError, load_config

HOST = "127.0.0.1"
MAINTAIN_INTERVAL_SECONDS = 30.0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="python -m rag_service.server")
    parser.add_argument("--port", type=int, default=None, help="overrides RAG_PORT (default 2190)")
    return parser.parse_args(argv)


def pick_port(args, cfg: Config) -> int:
    if args.port is None:
        return cfg.port
    if not 1 <= args.port <= 65535:
        raise ConfigError(f"--port must be between 1 and 65535, got {args.port}")
    return args.port


class MaintenanceThread(threading.Thread):
    """Calls service.maintain() every `interval` seconds until stopped."""

    def __init__(self, service, interval: float = MAINTAIN_INTERVAL_SECONDS):
        super().__init__(name="rag-maintenance", daemon=True)
        self._service = service
        self._interval = interval
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.wait(self._interval):
            try:
                self._service.maintain()
            except Exception as exc:  # housekeeping must never take the thread down
                print(f"maintenance error: {type(exc).__name__}: {exc}", file=sys.stderr)

    def stop(self) -> None:
        self._stop_event.set()


def main(argv=None) -> int:
    import uvicorn

    from rag_service.api import create_app
    from rag_service.service import SearchService

    try:
        cfg = load_config()
        port = pick_port(parse_args(argv), cfg)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    service = SearchService(cfg)
    maintenance = MaintenanceThread(service)
    maintenance.start()
    print(f"rag_service listening on http://{HOST}:{port}", flush=True)
    try:
        uvicorn.run(create_app(service, cfg), host=HOST, port=port, log_level="warning")
    finally:
        maintenance.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
