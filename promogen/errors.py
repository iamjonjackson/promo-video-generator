"""Error types with readable CLI messages."""


class PromogenError(Exception):
    """Base error with user-readable message."""


class FfmpegMissing(PromogenError):
    pass


class AudioMissing(PromogenError):
    pass


class BadImages(PromogenError):
    pass
