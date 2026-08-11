"""Domain errors for configuration, data, training, artifacts, and submission."""


class ConfigurationError(Exception):
    """Invalid or unknown configuration."""


class DataValidationError(Exception):
    """Official competition data failed schema or integrity checks."""


class TrainingError(Exception):
    """Training or cross-validation failed."""


class ArtifactError(Exception):
    """Experiment artifact storage or resume failed."""


class SubmissionValidationError(Exception):
    """Submission CSV failed validation."""


class AlignmentError(Exception):
    """Prediction ids failed uniqueness, set, order, or join checks."""
