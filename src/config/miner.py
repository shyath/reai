
from .base import BaseConfig

ENV_MINER_URL = "MINER_URL"


class MinerConfig(BaseConfig):
    """
    A configuration class for retrieving miner-specific settings from environment variables.

    Methods:
        __init__(env_path='.env', ignore_config_file=False):
            Initializes the MinerConfig instance and loads the environment file if not ignored.
        get_miner_url() -> str:
            Retrieves the MINER_URL environment variable.
    """

    def get_miner_url(self) -> str:
        """
        Retrieves the MINER_URL environment variable.

        Returns:
            str: 
                The value of the MINER_URL environment variable, or "http://0.0.0.0:5000" 
                if not set.
        """
        return str(self._get(ENV_MINER_URL, "http://0.0.0.0:5000"))
