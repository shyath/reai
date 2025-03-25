from .base import BaseConfig

ENV_VALIDATOR_INTERVAL = "VALIDATOR_INTERVAL"


class ValidatorConfig(BaseConfig):
    """
    A configuration class for retrieving validator-specific settings from environment variables.

    Methods:
        get_validator_interval() -> int:
            Retrieves the VALIDATOR_INTERVAL environment variable as an integer.
    """

    def get_validator_interval(self) -> int:
        """
        Retrieves the VALIDATOR_INTERVAL environment variable as an integer.

        Returns:
            int: 
                The value of the VALIDATOR_INTERVAL environment variable, or 10 if not set.

        Raises:
            ValueError: 
                If the VALIDATOR_INTERVAL environment variable contains non-digit characters.
        """
        interval = self._get(ENV_VALIDATOR_INTERVAL, '10')

        if not interval.isdigit():
            raise ValueError(
                f"The environment variable '{ENV_VALIDATOR_INTERVAL}' should only contain digits.")

        return int(interval)
