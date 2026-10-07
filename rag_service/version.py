"""A short fingerprint of the code on disk, so a running service can be told apart from a newer one.

The service reports the version it started with; the MCP bridge compares it with the code
that is on disk now and replaces a service that runs older code (see client.py).
"""
import hashlib
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent


def code_version(package_dir: Path = PACKAGE_DIR) -> str:
    """Hash of the names and contents of the package's modules (top-level .py files only)."""
    digest = hashlib.sha1()
    for module in sorted(Path(package_dir).glob("*.py")):
        digest.update(module.name.encode("utf-8") + b"\0")
        digest.update(module.read_bytes() + b"\0")
    return digest.hexdigest()[:10]
