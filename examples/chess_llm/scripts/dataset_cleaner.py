import argparse
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from tokenizers import Tokenizer


RESULT_SCORE_MAP = {
    # White wins.
    "1-0": 0.0,
    # Draw.
    "1/2-1/2": 0.5,
    # Black wins.
    "0-1": 1.0,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-file", required=True, type=str)
    parser.add_argument("--input-dataset-file", required=True, type=str)
    parser.add_argument("--output-dataset-file", required=True, type=str)
    args = parser.parse_args()
    
    tokenizer = Tokenizer.from_file(args.tokenizer_file)
    dataset = ds.dataset(args.input_dataset_file, format="parquet")
    scanner = ds.Scanner.from_dataset(
        dataset,
        columns=["Moves", "Result"],
        batch_size=50_000,
    )
    writer = None
    try:
        for batch in scanner.to_batches():
            # Get only moves & results from each row in the batch.
            moves_list = batch.column(batch.schema.get_field_index("Moves")).to_pylist()
            results_list = batch.column(batch.schema.get_field_index("Result")).to_pylist()

            # Process moves and results.
            out_prompt_moves = []
            out_completion_moves = []
            out_results = []
            for moves, result in zip(moves_list, results_list):
                out_prompt_moves.append([])
                out_completion_moves.append(tokenizer.encode_tokens(moves, add_special_tokens=True))
                out_result.append(RESULT_SCORE_MAP[result])

            # Build arrow arrays.
            out_batch = pa.record_batch(
                [
                    pa.array(out_prompt_moves),
                    pa.array(out_completion_moves),
                    pa.array(out_result),
                ],
                names=["Prompt Moves", "Completion Moves", "Result"]
            )
            
            # Initialize writer once (schema is known after first batch).
            if writer is None:
                writer = pq.ParquetWriter(
                    args.output_dataset_file,
                    out_batch.schema,
                    compression="zstd",
                    use_dictionary=True
                )
            writer.write_batch(out_batch)
    finally:
        if writer is not None:
            writer.close()
    

if __name__ == "__main__":
    main()