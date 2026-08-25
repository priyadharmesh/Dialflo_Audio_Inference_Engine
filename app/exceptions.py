"""
Custom exceptions for the Dialflo Voice Demographic Service.

These exceptions provide granular error handling throughout the pipeline,
allowing upstream code to handle specific failure modes appropriately.
"""


class AudioDecoderError(Exception):
    """Base exception for all audio decoding errors."""

    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class UnsupportedFormatError(AudioDecoderError):
    """Raised when the audio format is not supported or cannot be detected."""

    def __init__(self, message: str = "Unsupported or unrecognized audio format",
                 detected_format: str = None):
        details = {"detected_format": detected_format} if detected_format else {}
        super().__init__(message, details)


class CorruptedAudioError(AudioDecoderError):
    """Raised when the audio data is corrupted, truncated, or malformed."""

    def __init__(self, message: str = "Audio data is corrupted or malformed",
                 original_error: str = None):
        details = {"original_error": original_error} if original_error else {}
        super().__init__(message, details)


class EmptyAudioError(AudioDecoderError):
    """Raised when the audio contains no samples or is effectively silent."""

    def __init__(self, message: str = "Audio contains no usable samples",
                 num_samples: int = 0):
        details = {"num_samples": num_samples}
        super().__init__(message, details)


class AudioTooShortError(AudioDecoderError):
    """Raised when the audio is too short for meaningful inference."""

    def __init__(self, message: str = "Audio duration too short for inference",
                 duration_seconds: float = 0.0,
                 minimum_seconds: float = 0.5):
        details = {
            "duration_seconds": duration_seconds,
            "minimum_seconds": minimum_seconds
        }
        super().__init__(message, details)


class AudioTooLongError(AudioDecoderError):
    """Raised when the audio exceeds maximum allowed duration."""

    def __init__(self, message: str = "Audio duration exceeds maximum allowed",
                 duration_seconds: float = 0.0,
                 maximum_seconds: float = 60.0):
        details = {
            "duration_seconds": duration_seconds,
            "maximum_seconds": maximum_seconds
        }
        super().__init__(message, details)


class DecodingTimeoutError(AudioDecoderError):
    """Raised when audio decoding takes too long."""

    def __init__(self, message: str = "Audio decoding timed out",
                 timeout_seconds: float = 10.0):
        details = {"timeout_seconds": timeout_seconds}
        super().__init__(message, details)
