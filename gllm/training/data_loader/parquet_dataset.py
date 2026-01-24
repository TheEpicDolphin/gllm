import json
import pyarrow.dataset as ds
from typing import Iterator, Dict


class ParquetDataset:
    def __init__(
        self,
        path: str,
    ):
        self.path = path
        

    def __iter__(self) -> Iterator[str]:
        dataset = ds.dataset(self.path, format="parquet")
        scanner = ds.Scanner.from_dataset(dataset, batch_size=10_000)
        for batch in scanner.to_batches():
            columns = batch.columns
            num_rows = batch.num_rows
            for i in range(num_rows):
                yield tuple(col[i].as_py() for col in columns)