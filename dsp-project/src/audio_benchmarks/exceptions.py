"""Typed errors crossing the benchmark boundary."""


class BenchmarkError(Exception):
    """Base benchmark-domain error."""


class BenchmarkInputError(BenchmarkError):
    """Invalid annotations, audio, or metric preconditions."""


class BenchmarkConfigurationError(BenchmarkError):
    """Invalid experiment policy."""


class BenchmarkProviderError(BenchmarkError):
    """An inference or storage adapter failed."""
