import json
from typing import Iterator, Dict


class JSONLDataset:
    def __init__(
        self,
        path: str,
        prompt_key: str,
        completion_key: str,
    ):
        self.path = path
        self.prompt_key = prompt_key
        self.completion_key = completion_key


    def __iter__(self) -> Iterator[str]:
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                yield (obj[self.prompt_key], obj[self.completion_key])