

class MinerModule:
    """
    This class represents a miner registered on the network.

    Attributes:
        uid (int): The uid of the miner on the subnet.
        ss58 (str): The SS58 of the miner occupying the uid slot.
        address (str): The url the miner is serving on.

    Methods:
        __init__(uid: int, ss58: str, address: str, score: int):
            Initializes the ScoreMinerModule instance.
        to_dict(self) -> dict:
            Returns this ScoredMinerModule as a dict.
        def __repr__(self):
            Defines and returns the string representation of a 
            ScoredMinerModule
    """

    def __init__(self, uid: int, ss58: str, address: str):
        """
        Initializes the MinerModule instance.

        Args:
            uid (int): The uid of the miner on the subnet.
            ss58 (str): The SS58 of the miner occupying the uid slot.
            address (str): The url the miner is serving on.
        """
        self.uid = uid
        self.ss58 = ss58
        self.address = address

    def to_dict(self) -> dict:
        """
        Returns this MinerModule as a dict.

        Returns:
            dict: A dictionary representation of the MinerModule instance.
        """
        return {
            "uid": self.uid,
            "ss58": self.ss58,
            "address": self.address
        }

    def get_split_ip_port(self) -> list[str]:
        ip_port = self.address.split(':')
        return ip_port

    def __repr__(self) -> str:
        """
        Defines and returns the string representation of a MinerModule.

        Returns:
            str: The string representation of the MinerModule instance.
        """
        return f"MinerModule(UID={self.uid}, SS58={self.ss58}, Address={self.address})"


class ScoredMinerModule(MinerModule):
    """
    This class represents a miner registered on the network that should be
    (or will be) scored. It inherits from MinerModule.

    Attributes:
        uid (int): The uid of the miner on the subnet.
        ss58 (str): The SS58 of the miner occupying the uid slot.
        address (str): The url the miner is serving on.
        score (int): The score of the miner determined by the validator.

    Methods:
        __init__(uid: int, ss58: str, address: str, score: int):
            Initializes the ScoreMinerModule instance.
        to_dict(self) -> dict:
            Returns this ScoredMinerModule as a dict.
        def __repr__(self):
            Defines and returns the string representation of a 
            ScoredMinerModule
    """

    def __init__(self, uid: int, ss58: str, address: str, score: int):
        """
        Initializes the ScoredMinerModule instance.

        Args:
            uid (int): The uid of the miner on the subnet.
            ss58 (str): The SS58 of the miner occupying the uid slot.
            address (str): The url the miner is serving on.
            score (int): The score of the miner determined by the validator.
        """
        super().__init__(uid=uid, ss58=ss58, address=address)
        self.score = score

    def to_dict(self) -> dict:
        """
        Returns this ScoredMinerModule as a dict.

        Returns:
            dict: A dictionary representation of the ScoredMinerModule instance.
        """
        base_dict = super().to_dict()
        base_dict["score"] = self.score
        return base_dict

    def __repr__(self) -> str:
        """
        Defines and returns the string representation of a ScoredMinerModule.

        Returns:
            str: The string representation of the ScoredMinerModule instance.
        """
        return (
            f"ScoredMinerModule(UID={self.uid}, "
            f"SS58={self.ss58}, Address={self.address}, Score={self.score})"
        )
