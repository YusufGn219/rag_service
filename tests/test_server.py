import threading
import time

import pytest

from rag_service.config import ConfigError, load_config
from rag_service.server import MaintenanceThread, parse_args, pick_port


def _cfg(tmp_path, **env):
    return load_config({"RAG_VAULT_ROOT": str(tmp_path), **env})


def test_port_default_env_and_flag_priority(tmp_path):
    assert pick_port(parse_args([]), _cfg(tmp_path)) == 2190
    assert pick_port(parse_args([]), _cfg(tmp_path, RAG_PORT="3000")) == 3000
    assert pick_port(parse_args(["--port", "4000"]), _cfg(tmp_path, RAG_PORT="3000")) == 4000


@pytest.mark.parametrize("bad", ["0", "70000", "-5"])
def test_bad_port_flag_rejected(tmp_path, bad):
    with pytest.raises(ConfigError, match="--port"):
        pick_port(parse_args(["--port", bad]), _cfg(tmp_path))


class FakeService:
    def __init__(self, fail_first=False):
        self.calls = 0
        self.fail_first = fail_first

    def maintain(self):
        self.calls += 1
        if self.fail_first and self.calls == 1:
            raise RuntimeError("boom")
        return []


def _wait(cond, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end and not cond():
        time.sleep(0.005)
    return cond()


def test_maintenance_thread_calls_maintain_until_stopped():
    svc = FakeService()
    t = MaintenanceThread(svc, interval=0.01)
    t.start()
    assert _wait(lambda: svc.calls >= 3)
    t.stop()
    t.join(timeout=2)
    assert not t.is_alive()
    after = svc.calls
    time.sleep(0.05)
    assert svc.calls == after


def test_maintenance_thread_survives_errors():
    svc = FakeService(fail_first=True)
    t = MaintenanceThread(svc, interval=0.01)
    t.start()
    assert _wait(lambda: svc.calls >= 3)
    t.stop()
    t.join(timeout=2)


def test_maintenance_thread_is_daemon():
    assert MaintenanceThread(FakeService(), interval=1).daemon is True
