
from communex.client import CommuneClient, Keypair
from communex.types import SubnetParamsWithEmission, ModuleInfoWithOptionalBalance
from communex.misc import get_map_modules, get_map_subnets_params
from communex._common import get_node_url

from comx.interface import ComxInterface
from loguru import logger
import random
import time


class ComxClient(ComxInterface):
    """
    A client implementation of the ComxInterface for communication with Communex.

    Attributes:
        client (CommuneClient): 
            The client used to interact with the Communex network.

    Methods:
        __init__(client: CommuneClient):
            Initializes the ComxClient with a given CommuneClient.
        get_map_modules(
            netuid: int = 0, 
            include_balances: bool = False
        ) -> dict[str, ModuleInfoWithOptionalBalance]:
            Retrieves a mapping of registered modules for a given subnet netuid.
        get_subnet_params(
            block_hash: str | None = None, 
            key: int = 0
        ) -> SubnetParamsWithEmission | None:
            Retrieves the parameters of a subnet given a block hash and key.
        get_current_block() -> int:
            Retrieves the current block number.
    """

    def __init__(self, client: CommuneClient):
        """
        Initializes the ComxClient with a given CommuneClient.

        Args:
            client (CommuneClient): 
                The client used to interact with the Communex network.
        """
        self.client = client

    def get_map_modules(
        self,
        netuid: int = 0,
        include_balances: bool = False
    ) -> dict[str, ModuleInfoWithOptionalBalance]:
        """
        Retrieves a mapping of registered modules for a given subnet netuid.

        Args:
            netuid (int, optional): 
                The subnet ID for which to retrieve module information. Defaults to 0.
            include_balances (bool, optional): 
                Whether to include balance information in the returned module information. 
                Defaults to False.

        Returns:
            dict[str, ModuleInfoWithOptionalBalance]: 
                A dictionary mapping module keys to their corresponding information, 
                potentially including balance information.
        """
        return get_map_modules(self.client, netuid, include_balances)

    def get_subnet_params(
        self,
        block_hash: str | None = None,
        key: int = 0
    ) -> SubnetParamsWithEmission | None:
        """
        Retrieves the parameters of a subnet given a block hash and key. Not specifying a 
        block_hash will return the current subnet parameters.

        Args:
            block_hash (str | None, optional): 
                The block hash for which to retrieve subnet parameters. If None, the latest 
                parameters are retrieved.
            key (int, optional): 
                An additional key parameter for the query. Defaults to 0.

        Returns:
            SubnetParamsWithEmission | None: 
                The parameters of the subnet, or None if no parameters are found for the given 
                block hash and key.
        """
        subnets = get_map_subnets_params(self.client, block_hash)
        subnet = subnets.get(key, None)
        return subnet

    def get_current_block(self) -> int:
        """
        Retrieves the current block number.

        Returns:
            int: The current block number.
        """
        current_block = self.client.get_block()["header"]["number"]
        return int(current_block)

    def vote(
        self,
        key: Keypair,
        uids: list[int],
        weights: list[int],
        netuid: int = 0,
        use_testnet: bool = False
    ):
        """
        Args:
            key: The keypair used for signing the vote transaction.
            uids: A list of module UIDs to vote on.
            weights: A list of weights corresponding to each UID.
            netuid: The network identifier.
            use_testnet: Whether testnet is being used
        """
        try:
            self.client.vote(key=key, uids=uids, weights=weights, netuid=netuid)
        except Exception as e:
            logger.error(f"WARNING: Failed to set weights with exception: {e}. Will retry.")
            sleepy_time = random.uniform(1, 2)
            time.sleep(sleepy_time)
            # TODO: Remove need to create new commune client every retry
            self.client = CommuneClient(get_node_url(use_testnet=use_testnet))
            self.client.vote(key=key, uids=uids, weights=weights, netuid=netuid)
