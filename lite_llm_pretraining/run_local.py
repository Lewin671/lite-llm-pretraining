import argparse
from pathlib import Path

from lite_llm_pretraining.common import load_json, save_json
from lite_llm_pretraining.prepare_tiny_shakespeare import prepare_dataset
from lite_llm_pretraining.sample import sample_from_checkpoint
from lite_llm_pretraining.train import train_from_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full local MLX workflow: prepare, train, checkpoint, sample."
    )
    parser.add_argument(
        "--config",
        default="configs/tinyshakespeare-byte-smoke.json",
        help="Path to the run config JSON.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Optional prompt override for the final sample.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Optional override for the final sample length.",
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.9,
        help="Train split used when preparing Tiny Shakespeare.",
    )
    parser.add_argument(
        "--force_prepare",
        action="store_true",
        help="Rebuild the local dataset even if files already exist.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    config = load_json(config_path)
    data_dir = Path(config["data_dir"])
    meta_path = data_dir / "meta.json"

    if args.force_prepare or not meta_path.exists():
        meta = prepare_dataset(data_dir, train_split=args.train_split)
        print(f"prepared dataset at {data_dir}")
        print(f"train tokens: {meta['train_tokens']}, val tokens: {meta['val_tokens']}")
    else:
        meta = load_json(meta_path)
        print(f"using existing dataset at {data_dir}")
        print(f"train tokens: {meta['train_tokens']}, val tokens: {meta['val_tokens']}")

    train_result = train_from_config(config_path)
    checkpoint_dir = Path(train_result["best_checkpoint_dir"])
    prompt = args.prompt or config["sample_prompt"]
    max_new_tokens = args.max_new_tokens or config["train"]["sample_tokens"]
    final_sample, sample_state = sample_from_checkpoint(
        checkpoint_dir, prompt, max_new_tokens
    )

    out_dir = Path(train_result["out_dir"])
    final_sample_path = out_dir / "final_sample.txt"
    final_sample_path.write_text(final_sample, encoding="utf-8")

    summary = {
        "config": str(config_path),
        "data_dir": str(data_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "final_sample_path": str(final_sample_path),
        "loaded_step": sample_state.get("step"),
        **train_result,
    }
    save_json(out_dir / "local_run_summary.json", summary)

    print(f"final sample saved to {final_sample_path}")
    print(
        f"local run complete: step={summary['loaded_step']}, "
        f"best_val_loss={summary['best_val_loss']:.4f}"
    )


if __name__ == "__main__":
    main()
