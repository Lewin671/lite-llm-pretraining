import argparse
from pathlib import Path

from lite_llm_pretraining.common import load_checkpoint, sample_text


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
    return parser.parse_args()


def main():
    args = parse_args()
    model, _, state = load_checkpoint(Path(args.checkpoint_dir))
    output = sample_text(model, args.prompt, args.max_new_tokens)
    print(f"loaded checkpoint step={state.get('step', 'unknown')}")
    print(output)


if __name__ == "__main__":
    main()
