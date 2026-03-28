import argparse
import json
from pathlib import Path

import mlx.nn as nn
import mlx.optimizers as optim
import mlx.core as mx

from lite_llm_pretraining.common import (
    TransformerLM,
    count_parameters,
    estimate_loss,
    example_start_positions,
    get_batch,
    learning_rate_at,
    load_json,
    load_loss_mask,
    load_memmap,
    loss_fn,
    perplexity,
    sample_text,
    save_checkpoint,
    save_json,
    set_seed,
    token_dtype_from_meta,
)
from lite_llm_pretraining.tokenizer import load_tokenizer_from_meta
from lite_llm_pretraining.story_inference import (
    PLAIN_STORY_TEMPLATE,
    build_story_prompt,
    resolve_inference_profile_from_config,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a minimal MLX language model.")
    parser.add_argument(
        "--config",
        default="configs/tinyshakespeare-byte-smoke.json",
        help="Path to the run config JSON.",
    )
    return parser.parse_args()


def append_metrics(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def train_from_config(config_path: Path):
    config = load_json(config_path)
    set_seed(config["seed"])

    data_dir = Path(config["data_dir"])
    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "run_config.json", config)

    meta = load_json(data_dir / "meta.json")
    tokenizer = load_tokenizer_from_meta(meta, data_dir)
    model_config = {
        **config["model"],
        "vocab_size": tokenizer.vocab_size,
    }
    train_config = config["train"]
    sample_temperature = train_config.get("sample_temperature", 1.0)
    sample_top_k = train_config.get("sample_top_k")
    sample_repetition_penalty = train_config.get("sample_repetition_penalty", 1.0)
    sample_repetition_window = train_config.get("sample_repetition_window")
    inference_profile = resolve_inference_profile_from_config(config, path_hint=str(out_dir))

    model = TransformerLM(**model_config)
    optimizer = optim.AdamW(
        learning_rate=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
    )
    mx.eval(model.parameters(), optimizer.state)

    token_dtype = token_dtype_from_meta(meta)
    train_data = load_memmap(data_dir, "train", token_dtype=token_dtype)
    val_data = load_memmap(data_dir, "val", token_dtype=token_dtype)
    train_loss_mask = load_loss_mask(data_dir, "train", meta)
    val_loss_mask = load_loss_mask(data_dir, "val", meta)
    if not train_config.get("use_loss_mask", True):
        train_loss_mask = None
        val_loss_mask = None
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    batch_size = train_config["batch_size"]
    context_size = model_config["context_size"]
    batch_sampling_mode = train_config.get("batch_sampling_mode", "random")
    train_start_positions = None
    val_start_positions = None
    if batch_sampling_mode == "example_start":
        eos_token_id = meta.get("tokenizer", {}).get("eos_token_id")
        if eos_token_id is None:
            raise ValueError("batch_sampling_mode=example_start requires tokenizer.eos_token_id")
        train_start_positions = example_start_positions(
            train_data,
            eos_token_id,
            context_size,
        )
        val_start_positions = example_start_positions(
            val_data,
            eos_token_id,
            context_size,
        )
    best_val_loss = float("inf")
    metrics_path = out_dir / "metrics.jsonl"
    metrics_path.unlink(missing_ok=True)

    print(f"run: {config['run_name']}")
    print(f"params: {count_parameters(model):,}")
    print(f"train tokens: {meta['train_tokens']}, val tokens: {meta['val_tokens']}")
    if train_start_positions is not None:
        print(
            "example-aligned windows: "
            f"train={len(train_start_positions)}, val={len(val_start_positions)}"
        )
    last_val_loss = None

    for step in range(1, train_config["max_steps"] + 1):
        current_lr = learning_rate_at(
            step,
            train_config["learning_rate"],
            train_config["warmup_steps"],
            min_lr=train_config.get("min_learning_rate"),
            decay_steps=train_config.get("lr_decay_steps"),
        )
        optimizer.learning_rate = current_lr

        x, y, loss_mask = get_batch(
            train_data,
            batch_size,
            context_size,
            loss_mask_data=train_loss_mask,
            start_positions=train_start_positions,
        )
        train_loss, grads = loss_and_grad_fn(model, x, y, loss_mask)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, train_loss)
        train_loss_value = train_loss.item()

        if step % train_config["log_interval"] == 0 or step == 1:
            print(
                f"step {step:04d} "
                f"train_loss={train_loss_value:.4f} "
                f"lr={current_lr:.6f}"
            )

        if step % train_config["eval_interval"] == 0 or step == 1:
            val_loss = estimate_loss(
                model,
                val_data,
                batch_size,
                context_size,
                train_config["eval_batches"],
                loss_mask_data=val_loss_mask,
                start_positions=val_start_positions,
            )
            metric = {
                "step": step,
                "train_loss": round(train_loss_value, 6),
                "val_loss": round(val_loss, 6),
                "val_ppl": round(perplexity(val_loss), 6),
                "learning_rate": current_lr,
            }
            append_metrics(metrics_path, metric)
            print(
                f"step {step:04d} "
                f"val_loss={val_loss:.4f} "
                f"val_ppl={perplexity(val_loss):.2f}"
            )
            last_val_loss = val_loss

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    out_dir / "best",
                    model,
                    step,
                    {"best_val_loss": best_val_loss},
                    tokenizer=tokenizer,
                )

        if step % train_config["checkpoint_interval"] == 0 or step == train_config["max_steps"]:
            save_checkpoint(
                out_dir / "latest",
                model,
                step,
                {"best_val_loss": best_val_loss},
                tokenizer=tokenizer,
            )

        if step % train_config["sample_interval"] == 0 or step == train_config["max_steps"]:
            sample_prompt = config["sample_prompt"]
            if inference_profile.get("mode") == "story":
                sample_prompt = build_story_prompt(
                    config["sample_prompt"],
                    inference_profile.get("prompt_template", PLAIN_STORY_TEMPLATE),
                )
            sample = sample_text(
                model,
                sample_prompt,
                train_config["sample_tokens"],
                temperature=sample_temperature,
                tokenizer=tokenizer,
                top_k=sample_top_k,
                repetition_penalty=sample_repetition_penalty,
                repetition_window=sample_repetition_window,
            )
            sample_path = out_dir / "samples" / f"step-{step:04d}.txt"
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            sample_path.write_text(sample, encoding="utf-8")
            print(f"sample saved to {sample_path}")

    return {
        "run_name": config["run_name"],
        "out_dir": str(out_dir),
        "best_val_loss": best_val_loss,
        "last_val_loss": last_val_loss,
        "final_step": train_config["max_steps"],
        "best_checkpoint_dir": str(out_dir / "best"),
        "latest_checkpoint_dir": str(out_dir / "latest"),
        "metrics_path": str(metrics_path),
    }


def main():
    args = parse_args()
    train_from_config(Path(args.config))


if __name__ == "__main__":
    main()
