import argparse
import json
import tempfile
from pathlib import Path

from lite_llm_pretraining.common import load_json, save_json
from lite_llm_pretraining.run_local import prepare_from_config
from lite_llm_pretraining.sample import sample_from_checkpoint
from lite_llm_pretraining.train import train_from_config
from lite_llm_pretraining.validate_checkpoint import validate_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one tracked TinyStories sweep attempt from a base config."
    )
    parser.add_argument("--base_config", required=True, help="Base config JSON path.")
    parser.add_argument(
        "--attempt_id",
        required=True,
        help="Stable attempt identifier, such as A09.",
    )
    parser.add_argument(
        "--overrides_json",
        default="{}",
        help="Deep-merged JSON object of config overrides.",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Short note describing the hypothesis for this attempt.",
    )
    parser.add_argument(
        "--artifact_dir",
        default="progress/artifacts/tinystories-sweep",
        help="Directory for tracked attempt summaries.",
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.9,
        help="Train split used when a raw text dataset must be prepared.",
    )
    parser.add_argument(
        "--force_prepare",
        action="store_true",
        help="Rebuild the dataset even if meta.json already exists.",
    )
    return parser.parse_args()


def deep_merge(base, overrides):
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def sample_flags(text: str):
    return {
        "contains_unknown_marker": "⁇" in text or "<unk>" in text,
        "contains_eot_marker": "<|endoftext|>" in text,
    }


def run_attempt(
    base_config_path: Path,
    attempt_id: str,
    overrides_json: str,
    notes: str,
    artifact_dir: Path,
    train_split: float,
    force_prepare: bool,
):
    base_config = load_json(base_config_path)
    overrides = json.loads(overrides_json)
    config = deep_merge(base_config, overrides)
    config["attempt_id"] = attempt_id

    run_name = config.get("run_name", attempt_id.lower())
    config["run_name"] = run_name
    config.setdefault("out_dir", f"checkpoints/{run_name}")

    data_dir = Path(config["data_dir"])
    meta_path = data_dir / "meta.json"
    if force_prepare or not meta_path.exists():
        prepare_from_config(config, data_dir, train_split=train_split)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=f"-{attempt_id.lower()}.json",
        prefix="lite-llm-pretraining-",
        delete=False,
    ) as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temp_config_path = Path(handle.name)

    train_result = train_from_config(temp_config_path)
    checkpoint_dir = Path(train_result["best_checkpoint_dir"])
    sample_temperature = config["train"].get("sample_temperature", 1.0)
    sample_text, sample_state = sample_from_checkpoint(
        checkpoint_dir,
        config["sample_prompt"],
        config["train"]["sample_tokens"],
        temperature=sample_temperature,
    )

    validation_config = config.get("validation", {})
    validation_report = validate_checkpoint(
        checkpoint_dir,
        data_dir=data_dir,
        prompts=validation_config.get("prompts"),
        max_new_tokens=validation_config.get(
            "max_new_tokens", config["train"]["sample_tokens"]
        ),
        temperature=validation_config.get("temperature", sample_temperature),
        eval_batches=validation_config.get("eval_batches", 10),
    )

    unknown_markers = sum(
        sample["metrics"].get("unknown_marker_count", 0)
        for sample in validation_report["samples"]
    )
    artifact = {
        "attempt_id": attempt_id,
        "notes": notes,
        "base_config": str(base_config_path),
        "temp_config": str(temp_config_path),
        "run_name": run_name,
        "data_dir": str(data_dir),
        "out_dir": config["out_dir"],
        "checkpoint_dir": str(checkpoint_dir),
        "prepare": config.get("prepare", {}),
        "model": config["model"],
        "train": config["train"],
        "validation": validation_report.get("validation"),
        "validation_summary": validation_report["summary"],
        "validation_temperature": validation_config.get(
            "temperature", sample_temperature
        ),
        "sample_temperature": sample_temperature,
        "sample_prompt": config["sample_prompt"],
        "sample_excerpt": sample_text[:500],
        "sample_state": sample_state,
        "sample_flags": sample_flags(sample_text),
        "total_unknown_markers": unknown_markers,
        "train_result": train_result,
    }
    artifact_path = artifact_dir / f"{attempt_id.lower()}.json"
    save_json(artifact_path, artifact)
    return artifact_path, artifact


def main():
    args = parse_args()
    artifact_path, artifact = run_attempt(
        base_config_path=Path(args.base_config),
        attempt_id=args.attempt_id,
        overrides_json=args.overrides_json,
        notes=args.notes,
        artifact_dir=Path(args.artifact_dir),
        train_split=args.train_split,
        force_prepare=args.force_prepare,
    )
    print(f"attempt saved to {artifact_path}")
    print(
        f"{artifact['attempt_id']} "
        f"best_val_loss={artifact['train_result']['best_val_loss']:.4f} "
        f"passed={artifact['validation_summary']['passed_samples']}/"
        f"{artifact['validation_summary']['total_samples']} "
        f"unknown_markers={artifact['total_unknown_markers']}"
    )


if __name__ == "__main__":
    main()
