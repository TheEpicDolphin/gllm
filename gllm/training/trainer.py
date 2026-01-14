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
    ):
        # Enable training.
        model.training = True
        
        for epoch in range(num_epochs):
            for step, batched_samples in enumerate(dataloader):
                batched_prompts, batched_completions = zip(*batched_samples)
                batched_prompt_ids = model.tokenize(batched_prompts)
                batched_completion_ids = model.tokenize(batched_completions)
                
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
                positions = torch.arange(max_seq_len, device=model.device)
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
                loss = loss_fn.forward(
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
                
                if step % 100 == 0:
                    print(f"epoch {epoch}, step {step}, loss={loss.item()}")
        
        # Disable training.
        model.training = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--dataset-path", required=True, type=str)
    parser.add_argument("--prompt-key", default="prompt", required=False, type=str)
    parser.add_argument("--completion-key", default="completion", required=False, type=str)
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
            args.prompt_key,
            args.completion_key,
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