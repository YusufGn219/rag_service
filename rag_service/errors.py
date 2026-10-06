"""Errors the service reports to its callers. Kept apart so light clients can import them cheaply."""


class ServiceError(RuntimeError):
    """A problem the caller can understand and act on (the message says how)."""


class Busy(ServiceError):
    """The index is being updated by another run right now."""


class NoIndex(ServiceError):
    """There is no index yet."""


class ServiceUnavailable(ServiceError):
    """The service is not running and could not be started."""
