from typing import List


class DataLoader:
    def __init__(
        self,
        dataset,
        batch_size: int,
        max_num_samples: int = -1,
        drop_last: bool = False
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_num_samples = max_num_samples
        self.drop_last = False
    
    
    def __iter__(self):
        num_samples = 0
        batch: List[str] = []
        for sample in self.dataset:
            batch.append(sample)
            num_samples += 1
            if len(batch) == self.batch_size:
                yield batch
                batch.clear()
            if num_samples == self.max_num_samples:
                break

        if len(batch) > 0 and not self.drop_last:
            yield batch