

from comx.miner.module import ScoredMinerModule

class MinerRegistry:
    """
    This class manages the registration and lookup of miners by their UID and SS58 address.
    As of now it does NOT make sure these maps remained synced. For example, if you want to
    update an entry for a miners who's uid changed using the set method, it will update the
    miner in the _ss58_dict, but it will insert a new miner into the _uid_dict. This would 
    result in one entry representing the miner _ss58_dict, and two entries representing the
    miner in _uid_dict, due to the old uid never getting deleted. You must manually manage 
    this scenario for now.

    Attributes:
        _uid_dict (dict[int, ScoredMinerModule]): 
            Dictionary mapping UIDs to ScoredMinerModule instances.
        _ss58_dict (dict[str, ScoredMinerModule]): 
            Dictionary mapping SS58 addresses to ScoredMinerModule instances.
        
    Methods:
        __init__():
            Initializes the MinerRegistry with empty dictionaries for UIDs and SS58 addresses.
        set(miner: ScoredMinerModule):
            Registers or updates a miner in the registry. If the SS58 or UID has changed, the
            miner should be deleted before calling this method.
        get_by_uid(uid: int) -> ScoredMinerModule | None:
            Retrieves a miner by their UID.
        get_by_ss58(ss58: str) -> ScoredMinerModule | None:
            Retrieves a miner by their SS58 address.
        delete_by_uid(uid: int):
            Deletes a miner from the registry by their UID. Miners are deleted from
            both dictionaries.
        delete_by_ss58(ss58: str):
            Deletes a miner from the registry by their SS58 address. Miners are deleted
            from both dictionaries.
        _delete(miner: ScoredMinerModule):
            Deletes a miner from the internal dictionaries.
        get_all_by_uid() -> dict[int, ScoredMinerModule]:
            Retrieves all miners as a dictionary indexed by UID.
        get_all_by_ss58() -> dict[str, ScoredMinerModule]:
            Retrieves all miners as a dictionary indexed by SS58 address.
        to_uid_dict() -> dict[int, dict]:
            Converts each of the miners to a dictionary and returns those indexed by UID 
            to a dictionary format.
        to_ss58_dict() -> dict[str, dict]:
            Converts each of the miners to a dictionary and returns thsoe indexed by SS58 
            address to a dictionary format.
    """
    def __init__(self) -> None:
        """
        Initializes the MinerRegistry with empty dictionaries for UIDs and SS58 addresses.
        """
        self._uid_dict: dict[int, ScoredMinerModule] = {}
        self._ss58_dict: dict[str, ScoredMinerModule] = {}

    def set(self, miner: ScoredMinerModule):
        """
        Registers or updates a miner in the registry. If the SS58 or UID has changed, the
        miner should be deleted before calling this method.

        Args:
            miner (ScoredMinerModule): The miner to be registered or updated.
        """
        self._uid_dict[miner.uid] = miner
        self._ss58_dict[miner.ss58] = miner

    def get_by_uid(self, uid: int) -> ScoredMinerModule | None:
        """
        Retrieves a miner by their UID.

        Args:
            uid (int): The UID of the miner to retrieve.

        Returns:
            ScoredMinerModule | None: The miner with the specified UID, or None if not found.
        """
        if uid not in self._uid_dict:
            return None
        return self._uid_dict[uid]

    def get_by_ss58(self, ss58: str) -> ScoredMinerModule | None:
        """
        Retrieves a miner by their SS58 address.

        Args:
            ss58 (str): The SS58 address of the miner to retrieve.

        Returns:
            ScoredMinerModule | None: The miner with the specified SS58 address, 
            or None if not found.
        """
        if ss58 not in self._ss58_dict:
            return None
        return self._ss58_dict[ss58]

    def delete_by_uid(self, uid: int):
        """
        Deletes a miner from the registry by their UID. Miners are deleted from both 
        dictionaries.

        Args:
            uid (int): The UID of the miner to delete.
        """
        self._delete(self.get_by_uid(uid))

    def delete_by_ss58(self, ss58: str):
        """
        Deletes a miner from the registry by their SS58 address. Miners are deleted from 
        both dictionaries.

        Args:
            ss58 (str): The SS58 address of the miner to delete.
        """
        self._delete(self.get_by_ss58(ss58))

    def _delete(self, miner: ScoredMinerModule):
        """
        Deletes a miner from the internal dictionaries.

        Args:
            miner (ScoredMinerModule): The miner to delete.
        """
        if miner is None:
            return
        
        del self._uid_dict[miner.uid]
        del self._ss58_dict[miner.ss58]

    def get_all_by_uid(self) -> dict[int, ScoredMinerModule]:
        """
        Retrieves all miners as a dictionary indexed by UID.

        Returns:
            dict[int, ScoredMinerModule]: A copy of the dictionary of miners indexed by UID.
        """
        return self._uid_dict.copy() 
    
    def get_all_by_ss58(self) -> dict[str, ScoredMinerModule]:
        """
        Retrieves all miners as a dictionary indexed by SS58 address.

        Returns:
            dict[str, ScoredMinerModule]: A copy of the dictionary of miners indexed by SS58 
            address.
        """
        return self._ss58_dict.copy()
    
    def to_uid_dict(self) -> dict[int, dict]:
        """
        Converts each of the miners to a dictionary and returns those indexed by UID 
        to a dictionary format.

        Returns:
            dict[int, dict]: A dictionary where each UID is mapped to the dictionary 
            representation of the corresponding miner.
        """
        miners = self.get_all_by_uid()
        return {uid: miner.to_dict() for uid, miner in miners.items()}
    
    def to_ss58_dict(self) -> dict[str, dict]:
        """
        Converts each of the miners to a dictionary and returns those indexed by SS58 
        address to a dictionary format.

        Returns:
            dict[str, dict]: A dictionary where each SS58 address is mapped to the dictionary 
            representation of the corresponding miner.
        """
        miners = self.get_all_by_ss58()
        return {ss58: miner.to_dict() for ss58, miner in miners.items()}
    