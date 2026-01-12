import json
from typing import Iterator, Dict


class JSONLDataset:
    def __init__(
        self,
        path: str,
        key: str,
    ):
        self.path = path
        self.key = key


    def __iter__(self) -> Iterator[str]:
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                yield obj[self.key]