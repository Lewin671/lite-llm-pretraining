import argparse
import sys
from pathlib import Path

from lite_llm_pretraining.model import CheckpointLanguageModel
from lite_llm_pretraining.story_inference import (
    build_prompt_from_profile,
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
        "--context",
        default=None,
        help="Optional context used by qa mode.",
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
        "--top_k",
        type=int,
        default=None,
        help="Optional top-k sampling cutoff.",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Optional repetition penalty greater than 1.0.",
    )
    parser.add_argument(
        "--repetition_window",
        type=int,
        default=None,
        help="Optional lookback window used by repetition penalty.",
    )
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "raw", "story", "qa"],
        help="Prompt handling mode. auto uses checkpoint metadata for task-specific templates.",
    )
    return parser.parse_args()


def resolve_prompt(checkpoint_dir: Path, prompt: str, mode: str, context: str | None = None):
    if mode == "raw":
        return prompt
    profile = resolve_inference_profile(checkpoint_dir)
    if mode in {"story", "qa"}:
        profile = {**profile, "mode": mode}
        return build_prompt_from_profile(prompt, profile, context=context)
    if profile.get("mode") in {"story", "qa"}:
        return build_prompt_from_profile(prompt, profile, context=context)
    return prompt


def sample_from_checkpoint(
    checkpoint_dir: Path,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    mode: str = "auto",
    context: str | None = None,
    top_k: int | None = None,
    repetition_penalty: float = 1.0,
    repetition_window: int | None = None,
):
    model = CheckpointLanguageModel(checkpoint_dir)
    model_prompt = resolve_prompt(checkpoint_dir, prompt, mode, context=context)
    output = model.generate(
        model_prompt,
        max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        repetition_window=repetition_window,
    )
    return output, model.state


def stream_from_checkpoint(
    checkpoint_dir: Path,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    mode: str = "auto",
    context: str | None = None,
    top_k: int | None = None,
    repetition_penalty: float = 1.0,
    repetition_window: int | None = None,
):
    model = CheckpointLanguageModel(checkpoint_dir)
    model_prompt = resolve_prompt(checkpoint_dir, prompt, mode, context=context)
    return (
        model.stream_generate(
            model_prompt,
            max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
        ),
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
            context=args.context,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            repetition_window=args.repetition_window,
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
            context=args.context,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            repetition_window=args.repetition_window,
        )
        print(f"loaded checkpoint step={state.get('step', 'unknown')}")
        print(f"{args.prompt}{output}")


if __name__ == "__main__":
    main()
