
import os
from abc import ABC, abstractmethod

from loguru import logger

from .io import IOInterface
from comx.miner.registry import MinerRegistry
from comx.miner.module import ScoredMinerModule


class WeightIOInterface:

    @abstractmethod
    def validate_weights_file(self):
        pass

    @abstractmethod
    def write_weights(self, miner_registry: MinerRegistry):
        pass

    @abstractmethod
    def read_weights(self) -> MinerRegistry | None:
        pass


class WeightIO(WeightIOInterface):
    def __init__(self, io: IOInterface, dir_path: str, file_name: str):
        self.io = io
        self.dir_path: str = dir_path
        self.file_name: str = file_name
        self.file_path: str = os.path.join(self.dir_path, self.file_name)

    def validate_weights_file(self):
        if not self.io.path_exists(self.dir_path):
            self.io.make_dir(self.dir_path)
            logger.info(f"Created directory: {self.dir_path}")

        if not self.io.path_exists(self.file_path):
            self.io.write_json_file(self.file_path, {})
            logger.info(f"Created file: {self.file_path}")

    def write_weights(self, miner_registry: MinerRegistry):
        self.io.write_json_file(self.file_path, miner_registry.to_uid_dict())

    def read_weights(self) -> MinerRegistry | None:
        if not self.io.path_exists(self.file_path):
            return None

        json_data = self.io.read_json_file(self.file_path)
        if json_data is None:
            return None

        miner_registry = MinerRegistry()
        for _, miner_data in json_data.items():
            miner_registry.set(ScoredMinerModule(
                uid=miner_data["uid"],
                ss58=miner_data["ss58"],
                address=miner_data["address"],
                score=miner_data["score"]
            ))
        return miner_registry
