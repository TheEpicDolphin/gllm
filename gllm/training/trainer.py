import argparse

from gllm.model.model import Model
from gllm.training.cross_entropy_loss import CrossEntropyLoss
from gllm.training.sgd import SGD
from gllm.training.data_loader.data_loader import DataLoader
from gllm.training.data_loader.jsonl_dataset import JSONLDataset


class Trainer:
    def post_train(
        model: Model,
        dataloader: DataLoader,
        loss_fn,
        optimizer,
        num_epochs: int,
        device: str,
    ):
        for epoch in range(num_epochs):
            for step, batched_text_samples in enumerate(dataloader):
                batched_token_ids = model.tokenize(batched_text_samples)
                
                # Pad requests with padding token ids.
                max_len = max(len(row) for row in batched_token_ids)
                for row in batched_token_ids:
                    num_padding = max_len - len(row)
                    row.extend([model.pad_token_id] * num_padding)
                
                # Forward pass.
                inputs = batched_token_ids[:, :-1]
                logits = model.forward(inputs)
                
                # Compute loss.
                target_token_ids = batched_token_ids[:, 1:]
                loss = loss_fn.forward(logits, target_token_ids)
                
                # Backward pass.
                dL_dy = loss_fn.backward()
                model.backward(dL_dy)
                
                # Update weights.
                optimizer.step()
                optimizer.zero_grad()
                
                if step % 100 == 0:
                    print(f"epoch {epoch}, step {step}, loss={loss.item()}")
                    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--dataset-path", required=True, type=str)
    parser.add_argument("--row-key", default="text", required=False, type=str)
    parser.add_argument("--batch-size", default=1, required=False, type=int)
    parser.add_argument("--max-num-samples", default=-1, required=False, type=int)
    parser.add_argument("--optimizer", required=True, type=str)
    parser.add_argument("--loss-fn", required=True, type=str)
    parser.add_argument("--num_epochs", required=True, type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    trainer = Trainer()
    
    # Create model.
    model = Model(
        hf_model=args.model,
        gen_params=,
        device=args.device,
    )
    
    # Create dataloader.
    dataset_path = args.dataset_path
    assert dataset_path.exists()
    ext = dataset_path.suffix
    if ext == '.jsonl':
        dataset = JSONLDataset(
            dataset_path,
            args.row_key
        )
    else:
        raise NotImplementedError(f"Dataset extension: {ext} is not currently supported.")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        max_num_samples=args.max_num_samples,
    )
    
    # Optimizer.
    if args.optimizer == "sgd":
        optimizer = SGD(
            params=model.parameters(),
            lr=1e-3,
        )
    else:
        raise NotImplementedError(f"Optimizer: {args.optimizer} is not currently supported.")
    
    # Loss function.
    if args.loss_fn == "cross-entropy":
        loss_fn = CrossEntropyLoss()
    else:
        raise NotImplementedError(f"Loss function: {args.loss_fn} is not currently supported.")

    # Run post-training loop.
    trainer.post_train(
        model,
        dataloader
        optimizer,
        loss_fn,
        args.num_epochs,
    )


if __name__ == "__main__":
    main()