import argparse
from pathlib import Path

from lite_llm_pretraining.common import load_json, save_json
from lite_llm_pretraining.prepare_dolly_qa import prepare_dataset as prepare_dolly_qa
from lite_llm_pretraining.prepare_open_trivia_qa import (
    prepare_dataset as prepare_open_trivia_qa,
)
from lite_llm_pretraining.prepare_tiny_shakespeare import prepare_dataset
from lite_llm_pretraining.prepare_tinystories import prepare_dataset as prepare_tinystories
from lite_llm_pretraining.prepare_tinystories_sentencepiece import (
    prepare_dataset as prepare_tinystories_sentencepiece,
)
from lite_llm_pretraining.prepare_webquestions_qa import (
    prepare_dataset as prepare_webquestions_qa,
)
from lite_llm_pretraining.sample import sample_from_checkpoint
from lite_llm_pretraining.train import train_from_config
from lite_llm_pretraining.validate_checkpoint import validate_checkpoint


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
        help="Train split used when preparing split-based raw text datasets.",
    )
    parser.add_argument(
        "--force_prepare",
        action="store_true",
        help="Rebuild the local dataset even if files already exist.",
    )
    return parser.parse_args()


def prepare_from_config(config, data_dir: Path, train_split: float):
    prepare_config = config.get("prepare", {})
    prepare_name = prepare_config.get("name", "tinyshakespeare")

    if prepare_name == "tinyshakespeare":
        split = prepare_config.get("train_split", train_split)
        return prepare_dataset(data_dir, train_split=split)

    if prepare_name == "tinystories":
        train_url = prepare_config.get("train_url")
        val_url = prepare_config.get("val_url")
        return prepare_tinystories(
            data_dir,
            train_url=train_url,
            val_url=val_url,
            preserve_eot_marker=prepare_config.get("preserve_eot_marker", False),
        )

    if prepare_name == "tinystories_sentencepiece":
        return prepare_tinystories_sentencepiece(
            data_dir,
            byte_data_dir=Path(
                prepare_config.get(
                    "byte_data_dir",
                    prepare_config.get(
                        "source_data_dir", "data/tinystories-byte-clean"
                    ),
                )
            ),
            vocab_size=prepare_config.get("vocab_size", 2048),
            model_type=prepare_config.get("model_type", "bpe"),
            byte_fallback=prepare_config.get("byte_fallback", False),
            input_sentence_size=prepare_config.get("input_sentence_size", 200000),
            max_sentence_length=prepare_config.get("max_sentence_length", 16384),
            shuffle_input_sentence=prepare_config.get("shuffle_input_sentence", True),
            tokenizer_model_path=(
                Path(prepare_config["tokenizer_model_path"])
                if prepare_config.get("tokenizer_model_path")
                else None
            ),
            story_format=prepare_config.get("story_format", "plain"),
            prompt_sentence_count=prepare_config.get("prompt_sentence_count", 1),
            continuation_sentence_limit=prepare_config.get("continuation_sentence_limit"),
            prompt_label=prepare_config.get("prompt_label", "Prompt"),
            continuation_label=prepare_config.get("continuation_label", "Continuation"),
            instruction_text=prepare_config.get("instruction_text", ""),
            use_loss_mask=prepare_config.get("use_loss_mask"),
            prompt_loss_weight=prepare_config.get("prompt_loss_weight", 0.0),
            continuation_head_token_count=prepare_config.get(
                "continuation_head_token_count",
                0,
            ),
            continuation_head_loss_weight=prepare_config.get(
                "continuation_head_loss_weight",
                1.0,
            ),
        )

    if prepare_name == "dolly_qa":
        split = prepare_config.get("train_split", train_split)
        return prepare_dolly_qa(
            data_dir,
            source_url=prepare_config.get("source_url"),
            train_split=split,
            split_seed=prepare_config.get("split_seed", 42),
            vocab_size=prepare_config.get("vocab_size", 4096),
            model_type=prepare_config.get("model_type", "unigram"),
            byte_fallback=prepare_config.get("byte_fallback", True),
            input_sentence_size=prepare_config.get("input_sentence_size", 200000),
            max_sentence_length=prepare_config.get("max_sentence_length", 16384),
            shuffle_input_sentence=prepare_config.get("shuffle_input_sentence", True),
            tokenizer_model_path=(
                Path(prepare_config["tokenizer_model_path"])
                if prepare_config.get("tokenizer_model_path")
                else None
            ),
            question_label=prepare_config.get("question_label", "Question"),
            context_label=prepare_config.get("context_label", "Context"),
            answer_label=prepare_config.get("answer_label", "Answer"),
            instruction_text=prepare_config.get("instruction_text", ""),
            prompt_loss_weight=prepare_config.get("prompt_loss_weight", 0.0),
            continuation_head_token_count=prepare_config.get(
                "continuation_head_token_count",
                0,
            ),
            continuation_head_loss_weight=prepare_config.get(
                "continuation_head_loss_weight",
                1.0,
            ),
            context_word_limit=prepare_config.get("context_word_limit", 96),
            allowed_categories=prepare_config.get("allowed_categories"),
            min_answer_words=prepare_config.get("min_answer_words", 0),
            max_answer_words=prepare_config.get("max_answer_words"),
            max_question_words=prepare_config.get("max_question_words"),
            require_single_line_answer=prepare_config.get(
                "require_single_line_answer", False
            ),
            factoid_only=prepare_config.get("factoid_only", False),
            normalize_factoid_answers=prepare_config.get(
                "normalize_factoid_answers", False
            ),
            max_normalized_answer_words=prepare_config.get(
                "max_normalized_answer_words"
            ),
        )

    if prepare_name == "open_trivia_qa":
        split = prepare_config.get("train_split", train_split)
        return prepare_open_trivia_qa(
            data_dir,
            repo_url=prepare_config.get("repo_url"),
            repo_dir=(
                Path(prepare_config["repo_dir"])
                if prepare_config.get("repo_dir")
                else None
            ),
            train_split=split,
            split_seed=prepare_config.get("split_seed", 42),
            vocab_size=prepare_config.get("vocab_size", 2048),
            model_type=prepare_config.get("model_type", "unigram"),
            byte_fallback=prepare_config.get("byte_fallback", True),
            input_sentence_size=prepare_config.get("input_sentence_size", 200000),
            max_sentence_length=prepare_config.get("max_sentence_length", 16384),
            shuffle_input_sentence=prepare_config.get("shuffle_input_sentence", True),
            tokenizer_model_path=(
                Path(prepare_config["tokenizer_model_path"])
                if prepare_config.get("tokenizer_model_path")
                else None
            ),
            question_label=prepare_config.get("question_label", "Question"),
            context_label=prepare_config.get("context_label", "Context"),
            answer_label=prepare_config.get("answer_label", "Answer"),
            instruction_text=prepare_config.get("instruction_text", ""),
            prompt_loss_weight=prepare_config.get("prompt_loss_weight", 0.0),
            continuation_head_token_count=prepare_config.get(
                "continuation_head_token_count",
                0,
            ),
            continuation_head_loss_weight=prepare_config.get(
                "continuation_head_loss_weight",
                1.0,
            ),
            min_answer_words=prepare_config.get("min_answer_words", 1),
            max_answer_words=prepare_config.get("max_answer_words", 4),
            max_question_words=prepare_config.get("max_question_words", 28),
            require_single_line_answer=prepare_config.get(
                "require_single_line_answer", False
            ),
            factoid_only=prepare_config.get("factoid_only", False),
            normalize_factoid_answers=prepare_config.get(
                "normalize_factoid_answers", False
            ),
            max_normalized_answer_words=prepare_config.get(
                "max_normalized_answer_words"
            ),
            selected_categories=prepare_config.get("selected_categories"),
            question_prefixes=prepare_config.get("question_prefixes"),
            require_question_style=prepare_config.get("require_question_style", False),
        )

    if prepare_name == "webquestions_qa":
        return prepare_webquestions_qa(
            data_dir,
            repo_url=prepare_config.get("repo_url"),
            repo_dir=(
                Path(prepare_config["repo_dir"])
                if prepare_config.get("repo_dir")
                else None
            ),
            train_split_name=prepare_config.get("train_split_name", "trainmodel"),
            val_split_name=prepare_config.get("val_split_name", "val"),
            vocab_size=prepare_config.get("vocab_size", 2048),
            model_type=prepare_config.get("model_type", "unigram"),
            byte_fallback=prepare_config.get("byte_fallback", True),
            input_sentence_size=prepare_config.get("input_sentence_size", 200000),
            max_sentence_length=prepare_config.get("max_sentence_length", 16384),
            shuffle_input_sentence=prepare_config.get("shuffle_input_sentence", True),
            tokenizer_model_path=(
                Path(prepare_config["tokenizer_model_path"])
                if prepare_config.get("tokenizer_model_path")
                else None
            ),
            question_label=prepare_config.get("question_label", "Question"),
            context_label=prepare_config.get("context_label", "Context"),
            answer_label=prepare_config.get("answer_label", "Answer"),
            instruction_text=prepare_config.get("instruction_text", ""),
            prompt_loss_weight=prepare_config.get("prompt_loss_weight", 0.0),
            continuation_head_token_count=prepare_config.get(
                "continuation_head_token_count",
                0,
            ),
            continuation_head_loss_weight=prepare_config.get(
                "continuation_head_loss_weight",
                1.0,
            ),
            min_answer_words=prepare_config.get("min_answer_words", 1),
            max_answer_words=prepare_config.get("max_answer_words", 4),
            max_question_words=prepare_config.get("max_question_words", 24),
            require_single_line_answer=prepare_config.get(
                "require_single_line_answer", False
            ),
            factoid_only=prepare_config.get("factoid_only", False),
            normalize_factoid_answers=prepare_config.get(
                "normalize_factoid_answers", False
            ),
            max_normalized_answer_words=prepare_config.get(
                "max_normalized_answer_words"
            ),
            single_answer_only=prepare_config.get("single_answer_only", False),
        )

    raise ValueError(f"unsupported prepare dataset: {prepare_name}")


def main():
    args = parse_args()
    config_path = Path(args.config)
    config = load_json(config_path)
    data_dir = Path(config["data_dir"])
    meta_path = data_dir / "meta.json"

    if args.force_prepare or not meta_path.exists():
        meta = prepare_from_config(config, data_dir, train_split=args.train_split)
        print(f"prepared dataset at {data_dir}")
        print(f"train tokens: {meta['train_tokens']}, val tokens: {meta['val_tokens']}")
    else:
        meta = load_json(meta_path)
        print(f"using existing dataset at {data_dir}")
        print(f"train tokens: {meta['train_tokens']}, val tokens: {meta['val_tokens']}")

    train_result = train_from_config(config_path)
    checkpoint_dir = Path(
        train_result.get("best_suite_checkpoint_dir")
        or train_result["best_checkpoint_dir"]
    )
    prompt = args.prompt or config["sample_prompt"]
    max_new_tokens = args.max_new_tokens or config["train"]["sample_tokens"]
    sample_temperature = config["train"].get("sample_temperature", 1.0)
    sample_top_k = config["train"].get("sample_top_k")
    sample_repetition_penalty = config["train"].get("sample_repetition_penalty", 1.0)
    sample_repetition_window = config["train"].get("sample_repetition_window")
    final_sample, sample_state = sample_from_checkpoint(
        checkpoint_dir,
        prompt,
        max_new_tokens,
        temperature=sample_temperature,
        context=config.get("sample_context"),
        top_k=sample_top_k,
        repetition_penalty=sample_repetition_penalty,
        repetition_window=sample_repetition_window,
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
        "sample_temperature": sample_temperature,
        "sample_top_k": sample_top_k,
        "sample_repetition_penalty": sample_repetition_penalty,
        "sample_repetition_window": sample_repetition_window,
        **train_result,
    }

    validation_config = config.get("validation")
    if validation_config:
        prompts = validation_config.get("prompts")
        validation_report = validate_checkpoint(
            checkpoint_dir,
            data_dir=data_dir,
            prompts=prompts,
            max_new_tokens=validation_config.get(
                "max_new_tokens", min(160, max_new_tokens)
            ),
            temperature=validation_config.get("temperature", 0.8),
            top_k=validation_config.get("top_k"),
            repetition_penalty=validation_config.get("repetition_penalty", 1.0),
            repetition_window=validation_config.get("repetition_window"),
            eval_batches=validation_config.get("eval_batches", 10),
        )
        validation_path = out_dir / "validation_report.json"
        save_json(validation_path, validation_report)
        summary["validation_report_path"] = str(validation_path)
        summary["validation_passed"] = (
            validation_report["summary"]["passed_samples"]
            == validation_report["summary"]["total_samples"]
        )

    save_json(out_dir / "local_run_summary.json", summary)

    print(f"final sample saved to {final_sample_path}")
    if validation_config:
        print(f"validation report saved to {summary['validation_report_path']}")
    print(
        f"local run complete: step={summary['loaded_step']}, "
        f"best_val_loss={summary['best_val_loss']:.4f}"
    )


if __name__ == "__main__":
    main()
