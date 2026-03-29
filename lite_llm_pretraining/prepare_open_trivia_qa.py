import argparse
import json
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import sentencepiece as spm

from lite_llm_pretraining.common import save_json
from lite_llm_pretraining.prepare_dolly_qa import (
    encode_split,
    filter_examples,
    prompt_prefix_text,
    save_examples_jsonl,
    transform_examples,
    write_training_corpus,
)
from lite_llm_pretraining.prepare_tinystories_sentencepiece import train_sentencepiece


DEFAULT_REPO_URL = "https://github.com/uberspot/OpenTriviaQA"
DEFAULT_SELECTED_CATEGORIES = [
    "general",
    "world",
    "science-technology",
    "history",
    "geography",
    "humanities",
]
DEFAULT_QUESTION_PREFIXES = ["what", "which", "who", "where", "when", "how"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare OpenTriviaQA into local Q/A training data."
    )
    parser.add_argument("--out_dir", default="data/open-trivia-qa-spm")
    parser.add_argument("--repo_url", default=DEFAULT_REPO_URL)
    parser.add_argument("--repo_dir", default=None)
    parser.add_argument("--train_split", type=float, default=0.95)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--vocab_size", type=int, default=2048)
    parser.add_argument(
        "--model_type",
        default="unigram",
        choices=["bpe", "unigram"],
    )
    parser.add_argument("--byte_fallback", action="store_true")
    parser.add_argument("--input_sentence_size", type=int, default=200000)
    parser.add_argument("--max_sentence_length", type=int, default=16384)
    parser.add_argument("--disable_shuffle_input_sentence", action="store_true")
    parser.add_argument("--tokenizer_model_path", default=None)
    parser.add_argument("--question_label", default="Question")
    parser.add_argument("--context_label", default="Context")
    parser.add_argument("--answer_label", default="Answer")
    parser.add_argument("--instruction_text", default="")
    parser.add_argument("--prompt_loss_weight", type=float, default=0.0)
    parser.add_argument("--continuation_head_token_count", type=int, default=0)
    parser.add_argument("--continuation_head_loss_weight", type=float, default=1.0)
    parser.add_argument("--min_answer_words", type=int, default=1)
    parser.add_argument("--max_answer_words", type=int, default=4)
    parser.add_argument("--max_question_words", type=int, default=28)
    parser.add_argument("--require_single_line_answer", action="store_true")
    parser.add_argument("--factoid_only", action="store_true")
    parser.add_argument("--normalize_factoid_answers", action="store_true")
    parser.add_argument("--max_normalized_answer_words", type=int, default=None)
    parser.add_argument(
        "--selected_categories_json",
        default=json.dumps(DEFAULT_SELECTED_CATEGORIES),
    )
    parser.add_argument(
        "--question_prefixes_json",
        default=json.dumps(DEFAULT_QUESTION_PREFIXES),
    )
    parser.add_argument("--require_question_style", action="store_true")
    return parser.parse_args()


def clone_repo_if_missing(repo_url: str, repo_dir: Path):
    if repo_dir.exists():
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth=1", repo_url, str(repo_dir)],
        check=True,
    )


def load_category_names(payload: str):
    value = json.loads(payload)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("category/prefix payload must be a JSON array of strings")
    return value


def parse_open_trivia_examples(
    repo_dir: Path,
    selected_categories: list[str] | None = None,
    question_prefixes: list[str] | None = None,
    require_question_style: bool = False,
):
    categories_dir = repo_dir / "categories"
    if not categories_dir.exists():
        raise FileNotFoundError(f"missing categories dir: {categories_dir}")

    selected = set(selected_categories or [])
    prefixes = tuple((question_prefixes or []))
    examples = []
    for path in sorted(categories_dir.iterdir()):
        if not path.is_file():
            continue
        if selected and path.name not in selected:
            continue
        question = None
        answer = None
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for line in lines + ["#END"]:
            if line.startswith("#Q "):
                if question and answer:
                    question_text = " ".join(question.split()).strip()
                    answer_text = " ".join(answer.split()).strip()
                    if question_text and answer_text:
                        lower_question = question_text.lower()
                        if require_question_style:
                            if prefixes and not lower_question.startswith(prefixes):
                                question = line[3:].strip()
                                answer = None
                                continue
                            if "?" not in question_text and not lower_question.startswith(prefixes):
                                question = line[3:].strip()
                                answer = None
                                continue
                        examples.append(
                            {
                                "instruction": question_text,
                                "context": "",
                                "response": answer_text,
                                "category": path.name,
                            }
                        )
                question = line[3:].strip()
                answer = None
            elif line.startswith("^ "):
                answer = line[2:].strip()
    return examples


def prepare_dataset(
    out_dir: Path,
    repo_url: str = DEFAULT_REPO_URL,
    repo_dir: Path | None = None,
    train_split: float = 0.95,
    split_seed: int = 42,
    vocab_size: int = 2048,
    model_type: str = "unigram",
    byte_fallback: bool = True,
    input_sentence_size: int = 200000,
    max_sentence_length: int = 16384,
    shuffle_input_sentence: bool = True,
    tokenizer_model_path: Path | None = None,
    question_label: str = "Question",
    context_label: str = "Context",
    answer_label: str = "Answer",
    instruction_text: str = "",
    prompt_loss_weight: float = 0.0,
    continuation_head_token_count: int = 0,
    continuation_head_loss_weight: float = 1.0,
    min_answer_words: int = 1,
    max_answer_words: int | None = 4,
    max_question_words: int | None = 28,
    require_single_line_answer: bool = False,
    factoid_only: bool = False,
    normalize_factoid_answers: bool = False,
    max_normalized_answer_words: int | None = None,
    selected_categories: list[str] | None = None,
    question_prefixes: list[str] | None = None,
    require_question_style: bool = False,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = repo_dir or (out_dir / "OpenTriviaQA")
    clone_repo_if_missing(repo_url, repo_dir)

    examples = parse_open_trivia_examples(
        repo_dir=repo_dir,
        selected_categories=selected_categories,
        question_prefixes=question_prefixes,
        require_question_style=require_question_style,
    )
    examples = filter_examples(
        examples,
        min_answer_words=min_answer_words,
        max_answer_words=max_answer_words,
        max_question_words=max_question_words,
        require_single_line_answer=require_single_line_answer,
    )
    examples = transform_examples(
        examples,
        factoid_only=factoid_only,
        normalize_factoid_answers=normalize_factoid_answers,
        max_normalized_answer_words=max_normalized_answer_words,
    )
    if len(examples) < 2:
        raise ValueError("filtered OpenTriviaQA dataset must contain at least 2 examples")

    rng = np.random.default_rng(split_seed)
    indices = np.arange(len(examples))
    rng.shuffle(indices)
    shuffled_examples = [examples[idx] for idx in indices]
    split_index = max(1, int(len(shuffled_examples) * train_split))
    train_examples = shuffled_examples[:split_index]
    val_examples = shuffled_examples[split_index:]
    if not val_examples:
        val_examples = train_examples[-1:]
        train_examples = train_examples[:-1]

    save_examples_jsonl(out_dir / "train_examples.jsonl", train_examples)
    save_examples_jsonl(out_dir / "val_examples.jsonl", val_examples)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix="-opentrivia-train.txt",
        prefix="lite-llm-pretraining-",
        delete=False,
    ) as handle:
        formatted_train_path = Path(handle.name)
    write_training_corpus(
        train_examples,
        formatted_train_path,
        question_label=question_label,
        context_label=context_label,
        answer_label=answer_label,
        instruction_text=instruction_text,
        context_word_limit=0,
    )
    try:
        if tokenizer_model_path is None:
            model_path, vocab_path = train_sentencepiece(
                formatted_train_path,
                out_dir,
                vocab_size=vocab_size,
                model_type=model_type,
                byte_fallback=byte_fallback,
                input_sentence_size=input_sentence_size,
                max_sentence_length=max_sentence_length,
                shuffle_input_sentence=shuffle_input_sentence,
            )
        else:
            tokenizer_model_path = Path(tokenizer_model_path)
            if not tokenizer_model_path.exists():
                raise FileNotFoundError(
                    f"tokenizer model does not exist: {tokenizer_model_path}"
                )
            model_path = out_dir / "tokenizer.model"
            vocab_path = out_dir / "tokenizer.vocab"
            shutil.copyfile(tokenizer_model_path, model_path)
            source_vocab_path = tokenizer_model_path.with_suffix(".vocab")
            if source_vocab_path.exists():
                shutil.copyfile(source_vocab_path, vocab_path)
    finally:
        formatted_train_path.unlink(missing_ok=True)

    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    train_tokens = encode_split(
        train_examples,
        out_dir / "train.bin",
        out_dir / "train_loss_mask.bin",
        processor,
        question_label=question_label,
        context_label=context_label,
        answer_label=answer_label,
        instruction_text=instruction_text,
        prompt_loss_weight=prompt_loss_weight,
        continuation_head_token_count=continuation_head_token_count,
        continuation_head_loss_weight=continuation_head_loss_weight,
        context_word_limit=0,
    )
    val_tokens = encode_split(
        val_examples,
        out_dir / "val.bin",
        out_dir / "val_loss_mask.bin",
        processor,
        question_label=question_label,
        context_label=context_label,
        answer_label=answer_label,
        instruction_text=instruction_text,
        prompt_loss_weight=prompt_loss_weight,
        continuation_head_token_count=continuation_head_token_count,
        continuation_head_loss_weight=continuation_head_loss_weight,
        context_word_limit=0,
    )

    category_counts = Counter(example["category"] or "unknown" for example in shuffled_examples)
    meta = {
        "dataset": "open-trivia-qa",
        "task_format": "qa",
        "source_repo_url": repo_url,
        "source_repo_dir": str(repo_dir),
        "num_examples": len(shuffled_examples),
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "category_counts": dict(sorted(category_counts.items())),
        "selected_categories": selected_categories,
        "question_prefixes": question_prefixes,
        "require_question_style": require_question_style,
        "min_answer_words": min_answer_words,
        "max_answer_words": max_answer_words,
        "max_question_words": max_question_words,
        "require_single_line_answer": require_single_line_answer,
        "factoid_only": factoid_only,
        "normalize_factoid_answers": normalize_factoid_answers,
        "max_normalized_answer_words": max_normalized_answer_words,
        "question_label": question_label,
        "context_label": context_label,
        "answer_label": answer_label,
        "instruction_text": instruction_text,
        "prompt_loss_weight": prompt_loss_weight,
        "continuation_head_token_count": continuation_head_token_count,
        "continuation_head_loss_weight": continuation_head_loss_weight,
        "tokenizer_model_path": str(tokenizer_model_path) if tokenizer_model_path else None,
        "tokenizer": {
            "name": "sentencepiece",
            "model_file": model_path.name,
            "vocab_size": processor.vocab_size(),
            "eos_token_id": processor.eos_id(),
        },
        "vocab_size": processor.vocab_size(),
        "token_dtype": "uint16",
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "has_loss_mask": True,
        "loss_mask_dtype": "float16",
        "loss_mask_mode": (
            "prompt_weighted_head_weighted_answer"
            if prompt_loss_weight > 0 and continuation_head_token_count > 0
            else (
                "head_weighted_answer"
                if continuation_head_token_count > 0
                else "answer_only"
            )
        ),
    }
    save_json(out_dir / "meta.json", meta)
    return meta


def main():
    args = parse_args()
    selected_categories = load_category_names(args.selected_categories_json)
    question_prefixes = load_category_names(args.question_prefixes_json)
    meta = prepare_dataset(
        out_dir=Path(args.out_dir),
        repo_url=args.repo_url,
        repo_dir=Path(args.repo_dir) if args.repo_dir else None,
        train_split=args.train_split,
        split_seed=args.split_seed,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        byte_fallback=args.byte_fallback,
        input_sentence_size=args.input_sentence_size,
        max_sentence_length=args.max_sentence_length,
        shuffle_input_sentence=not args.disable_shuffle_input_sentence,
        tokenizer_model_path=Path(args.tokenizer_model_path) if args.tokenizer_model_path else None,
        question_label=args.question_label,
        context_label=args.context_label,
        answer_label=args.answer_label,
        instruction_text=args.instruction_text,
        prompt_loss_weight=args.prompt_loss_weight,
        continuation_head_token_count=args.continuation_head_token_count,
        continuation_head_loss_weight=args.continuation_head_loss_weight,
        min_answer_words=args.min_answer_words,
        max_answer_words=args.max_answer_words,
        max_question_words=args.max_question_words,
        require_single_line_answer=args.require_single_line_answer,
        factoid_only=args.factoid_only,
        normalize_factoid_answers=args.normalize_factoid_answers,
        max_normalized_answer_words=args.max_normalized_answer_words,
        selected_categories=selected_categories,
        question_prefixes=question_prefixes,
        require_question_style=args.require_question_style,
    )
    print(f"saved dataset to {args.out_dir}")
    print(f"train examples: {meta['train_examples']}, val examples: {meta['val_examples']}")
    print(f"train tokens: {meta['train_tokens']}, val tokens: {meta['val_tokens']}")


if __name__ == "__main__":
    main()
