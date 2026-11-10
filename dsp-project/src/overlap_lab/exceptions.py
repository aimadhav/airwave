"""Typed errors at the lab boundary."""


class LabError(Exception):
    """Base lab storage, inference or validation error."""


class LabInputError(LabError):
    """An experiment/reference violates the lab contract."""


class LabConflict(LabError):
    """Another tab saved a newer annotation revision."""


class LabBusy(LabError):
    """The bounded local experiment queue is full."""
