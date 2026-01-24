import argparse

from safetensors.torch import save_file

from gllm.model.layers.attention import AttentionMetadata
from gllm.model.model import Model
from gllm.training.loss.cross_entropy_loss import CrossEntropyLoss
from gllm.training.optimizers.sgd import SGD
from gllm.training.data_loader.data_loader import DataLoader
from gllm.training.data_loader.parquet_dataset import ParquetDataset


class Trainer:
    def train(
        model: Model,
        train_data_loader: DataLoader,
        val_data_loader: DataLoader,
        loss_fn,
        optimizer,
        num_epochs: int,
        validate_every: int,
        checkpoint_dir: str,
    ):
        # Enable training.
        model.training = True
        
        min_val_loss = float("inf")
        for epoch in range(num_epochs):
            for step, train_batch in enumerate(train_data_loader):
                batched_prompt_ids, batched_completion_ids = zip(*train_batch)
                
                # Join prompt and completion token ids together.
                batched_token_ids = []
                completion_mask = []
                seq_lens = [len(prompt) + len(completion) for prompt, completion in zip(batched_prompt_ids, batched_completion_ids)]
                max_seq_len = max(seq_lens)
                for seq_len, prompt_ids, completion_ids in zip(seq_lens, batched_prompt_ids, batched_completion_ids):
                    # Pad sequences so that they all have the same length, T.
                    prompt_len = len(prompt_ids)
                    completion_len = len(completion_ids)
                    pad_len = max_seq_len - seq_len
                    pad_token_ids = [model.pad_token_id] * pad_len
                    batched_token_ids.append(prompt_ids + completion_ids + pad_token_ids)
                    completion_mask.append([False] * prompt_len + [True] * completion_len + [False] * pad_len)
                batched_token_ids_tensor = torch.tensor(batched_token_ids, device=model.device)
                completion_mask_tensor = torch.tensor(completion_mask, device=model.device)
                seq_lens_tensor = torch.tensor(seq_lens, device=model.device)
                
                # Calculate token positions.
                B, _ = batched_token_ids_tensor.shape
                # [T]
                positions = torch.arange(max_seq_len, device=model.device)
                # [B, T]
                positions = positions.unsqueeze(0).expand(B, -1)
                
                # Calculate causal attention bias.
                # [B, T, T]
                bias = torch.full(
                    (B, max_seq_len, max_seq_len),
                    float("-inf"),
                    dtype=model.dtype,
                    device=model.device,
                )
                bias.triu_(diagonal=1)
                
                # Forward pass.
                # [B, T]
                input_ids = batched_token_ids_tensor[:, :-1]
                attention_metadata = AttentionMetadata(
                    positions=positions,
                    query_lens=seq_lens_tensor,
                    seq_lens=seq_lens_tensor,
                    bias=bias,
                    # No KV caching for training.
                    block_table=None,
                    slot_mapping=None,
                    query_slot_mapping=None,
                )
                # [B, T]
                logits = model.forward(
                    input_ids,
                    attention_metadata,
                )
                
                # Compute loss.
                # [B, T]
                target_ids = batched_token_ids_tensor[:, 1:]
                completion_mask = completion_mask_tensor[:, 1:]
                train_loss = loss_fn.forward(
                    logits,
                    target_ids,
                    completion_mask,
                )
                
                # Backward pass.
                dL_dy = loss_fn.backward()
                model.backward(dL_dy)
                
                # Update weights.
                optimizer.step()
                optimizer.zero_grad()
                    
                if step % validate_every == 0:
                    # Validate.
                    for val_batch in enumerate(val_data_loader):
                        # TODO.
                        
                    print(f"epoch: {epoch}, step: {step}, training loss: {train_loss.item()}, validation loss: {val_loss}")
                    if val_loss < min_val_loss:
                        # Lowest validation loss so far. Create checkpoint.
                        min_val_loss = val_loss
                        weights_dict = {}
                        model.save_tensors(weights_dict)
                        save_file(weights_dict, "model.safetensors")
                        
        # Disable training.
        model.training = False
        

def create_dataloader(
    path: str,
    batch_size: int,
    max_num_samples: int
):
    # Create training dataloader.
    training_dataset_path = args.training_dataset_path
    assert path.exists()
    ext = path.suffix
    if ext == '.parquet':
        dataset = ParquetDataset(
            path,
        )
    else:
        raise NotImplementedError(f"Dataset extension: {ext} is not currently supported.")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        max_num_samples=max_num_samples,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--training-dataset-path", required=True, type=str)
    parser.add_argument("--validation-dataset-path", required=True, type=str)
    parser.add_argument("--batch-size", default=1, required=False, type=int)
    parser.add_argument("--optimizer", required=True, type=str)
    parser.add_argument("--loss-fn", required=True, type=str)
    parser.add_argument("--num_epochs", required=True, type=int)
    parser.add_argument("--checkpoint-dir", required=True, type=int)
    parser.add_argument("--validate_every", default=2000, required=False, type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # Create model.
    model = Model(
        model_path=args.model,
        max_seq_len=1024,
        device=args.device,
    )
    
    # Create training dataloader.
    train_data_loader = create_data_loader(
        args.training_dataset_path,
        args.batch_size,
    )
    
    # Create validation dataloader.
    val_data_loader = create_dataloader(
        args.validation_dataset_path,
        args.batch_size,
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

    # Run training loop.
    trainer = Trainer()
    trainer.train(
        model,
        train_data_loader,
        val_data_loader,
        optimizer,
        loss_fn,
        args.num_epochs,
        args.validate_every,
    )


if __name__ == "__main__":
    main()