import os
import json
from typing import Any
from abc import ABC, abstractmethod


class IOInterface(ABC):

    @abstractmethod
    def path_exists(self, path: str) -> bool:
        pass

    @abstractmethod
    def make_dir(self, path: str, mode: int = 511, exist_ok: bool = False):
        pass

    @abstractmethod
    def write_json_file(self, path: str, body: Any):
        pass

    @abstractmethod
    def read_json_file(self, path: str) -> Any | None:
        pass


class IO(IOInterface):

    def path_exists(self, path: str) -> bool:
        return os.path.exists(path)

    def make_dir(self, path: str, mode: int = 511, exist_ok: bool = False):
        os.makedirs(path, mode=mode, exist_ok=exist_ok)

    def write_json_file(self, path: str, body: Any):
        # Ensure the directory exists
        dir_name = os.path.dirname(path)
        if dir_name:
            self.make_dir(dir_name, exist_ok=True)

        # Create an empty file if it does not exist
        if not self.path_exists(path):
            open(path, 'a').close()

        with open(path, 'w') as file:
            json.dump(body, file, indent=4)

    def read_json_file(self, path: str) -> Any | None:
        # Ensure the file exists
        if not self.path_exists(path):
            open(path, 'a').close()

        with open(path, 'r') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = None

        return data
