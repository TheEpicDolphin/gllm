import argparse
import os

from tokenizers import AddedToken, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"
SPECIAL_TOKENS = ["[UNK]", "[PAD]", BOS_TOKEN, EOS_TOKEN]
NUM_RANKS = 8


def is_valid_knight_move(from_square, to_square):
    x1, y1 = from_square
    x2, y2 = to_square
    dx = x2 - x1
    dy = y2 - y1
    return dx * dx + dy * dy == 5


def is_valid_queen_move(from_square, to_square):
    # Queen's moves are a superset of pawn, bishop, rook, and king.
    x1, y1 = from_square
    x2, y2 = to_square
    dx = x2 - x1
    dy = y2 - y1
    return ((dx == 0) or (dy == 0)) ^ (abs(dx) == abs(dy))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=str)
    args = parser.parse_args()
    
    # Build 2D list of all chess squares.
    chess_squares = [[f"{f}{r}" for f in "abcdefgh"] for r in range(NUM_RANKS, 0, -1)]
    
    # Construct vocabulary of valid chess moves.
    vocab = {tok:id for id, tok in enumerate(SPECIAL_TOKENS)}
    for y1 in range(0, NUM_RANKS):
        for x1 in range(0, NUM_RANKS):
            for y2 in range(0, NUM_RANKS):
                for x2 in range(0, NUM_RANKS):
                    from_square = (x1, y1)
                    to_square = (x2, y2)
                    move = f"{chess_squares[y1][x1]}{chess_squares[y2][x2]}"
                    if (is_valid_knight_move(from_square, to_square)
                        or is_valid_queen_move(from_square, to_square)):
                        move = f"{chess_squares[y1][x1]}{chess_squares[y2][x2]}"
                        vocab[move] = len(vocab)
    # Add pawn promotion moves.
    for f in "abcdefgh":
        for p in "qrbn":
            bp_promotion_move = f"{f}{2}{f}{1}{p}"
            vocab[bp_promotion_move] = len(vocab)
            wp_promotion_move = f"{f}{NUM_RANKS-1}{f}{NUM_RANKS}{p}"
            vocab[wp_promotion_move] = len(vocab)
            
    # Chess moves (in UCI notation) are treated as words.
    tokenizer = Tokenizer(
        WordLevel(
            vocab=vocab,
            unk_token="[UNK]",
        )
    )
    # Moves are separated by whitespace.
    tokenizer.pre_tokenizer = Whitespace()
    # Mark special tokens.
    tokenizer.add_special_tokens([AddedToken(t, special=True) for t in SPECIAL_TOKENS])
    # Add template to configure how BOS and EOS tokens are added.
    bos_id = tokenizer.token_to_id(BOS_TOKEN)
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    tokenizer.post_processor = TemplateProcessing(
        single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
        special_tokens=[(BOS_TOKEN, bos_id), (EOS_TOKEN, eos_id)],
    )

    # Save tokenizer to output directory.
    tokenizer.save(os.path.join(args.output_dir, "tokenizer.json"))
    

if __name__ == "__main__":
    main()