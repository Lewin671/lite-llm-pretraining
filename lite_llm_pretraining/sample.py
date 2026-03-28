import argparse
import sys
from pathlib import Path

from lite_llm_pretraining.model import CheckpointLanguageModel
from lite_llm_pretraining.story_inference import (
    PLAIN_STORY_TEMPLATE,
    build_story_prompt,
    resolve_inference_profile,
)


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
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "raw", "story"],
        help="Prompt handling mode. auto uses checkpoint metadata for TinyStories.",
    )
    return parser.parse_args()


def resolve_prompt(checkpoint_dir: Path, prompt: str, mode: str):
    if mode == "raw":
        return prompt
    if mode == "story":
        profile = resolve_inference_profile(checkpoint_dir)
        return build_story_prompt(
            prompt,
            profile.get("prompt_template", PLAIN_STORY_TEMPLATE),
        )

    profile = resolve_inference_profile(checkpoint_dir)
    if profile.get("mode") == "story":
        return build_story_prompt(
            prompt,
            profile.get("prompt_template", PLAIN_STORY_TEMPLATE),
        )
    return prompt


def sample_from_checkpoint(
    checkpoint_dir: Path,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    mode: str = "auto",
):
    model = CheckpointLanguageModel(checkpoint_dir)
    model_prompt = resolve_prompt(checkpoint_dir, prompt, mode)
    output = model.generate(model_prompt, max_new_tokens, temperature=temperature)
    return output, model.state


def stream_from_checkpoint(
    checkpoint_dir: Path,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    mode: str = "auto",
):
    model = CheckpointLanguageModel(checkpoint_dir)
    model_prompt = resolve_prompt(checkpoint_dir, prompt, mode)
    return (
        model.stream_generate(model_prompt, max_new_tokens, temperature=temperature),
        model.state,
    )


def main():
    args = parse_args()
    if args.stream:
        stream, state = stream_from_checkpoint(
            Path(args.checkpoint_dir),
            args.prompt,
            args.max_new_tokens,
            args.temperature,
            mode=args.mode,
        )
        print(f"loaded checkpoint step={state.get('step', 'unknown')}")
        sys.stdout.write(args.prompt)
        sys.stdout.flush()
        for piece in stream:
            sys.stdout.write(piece)
            sys.stdout.flush()
        print()
    else:
        output, state = sample_from_checkpoint(
            Path(args.checkpoint_dir),
            args.prompt,
            args.max_new_tokens,
            args.temperature,
            mode=args.mode,
        )
        print(f"loaded checkpoint step={state.get('step', 'unknown')}")
        print(f"{args.prompt}{output}")


if __name__ == "__main__":
    main()
