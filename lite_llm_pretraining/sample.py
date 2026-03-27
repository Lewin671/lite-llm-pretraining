import argparse
import sys
from pathlib import Path

from lite_llm_pretraining.common import load_checkpoint, sample_text, sample_text_stream


def parse_args():
    parser = argparse.ArgumentParser(description="Sample from a saved MLX checkpoint.")
    parser.add_argument(
        "--checkpoint_dir",
        default="checkpoints/tinyshakespeare-byte-smoke/best",
        help="Directory containing weights.npz and model_config.json.",
    )
    parser.add_argument(
        "--prompt",
        default="ROMEO:\n",
        help="UTF-8 prompt to seed generation.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=200,
        help="Number of new tokens to decode.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Print generated text incrementally as tokens are decoded.",
    )
    return parser.parse_args()


def sample_from_checkpoint(checkpoint_dir: Path, prompt: str, max_new_tokens: int):
    model, _, state = load_checkpoint(checkpoint_dir)
    output = sample_text(model, prompt, max_new_tokens)
    return output, state


def stream_from_checkpoint(checkpoint_dir: Path, prompt: str, max_new_tokens: int):
    model, _, state = load_checkpoint(checkpoint_dir)
    return sample_text_stream(model, prompt, max_new_tokens), state


def main():
    args = parse_args()
    if args.stream:
        stream, state = stream_from_checkpoint(
            Path(args.checkpoint_dir), args.prompt, args.max_new_tokens
        )
        print(f"loaded checkpoint step={state.get('step', 'unknown')}")
        for piece in stream:
            sys.stdout.write(piece)
            sys.stdout.flush()
        print()
    else:
        output, state = sample_from_checkpoint(
            Path(args.checkpoint_dir), args.prompt, args.max_new_tokens
        )
        print(f"loaded checkpoint step={state.get('step', 'unknown')}")
        print(output)


if __name__ == "__main__":
    main()
