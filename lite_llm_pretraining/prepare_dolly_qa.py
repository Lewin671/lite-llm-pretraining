import argparse
import json
import shutil
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import sentencepiece as spm

from lite_llm_pretraining.common import save_json
from lite_llm_pretraining.prepare_tinystories_sentencepiece import train_sentencepiece


DEFAULT_DOLLY_URL = (
    "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/"
    "databricks-dolly-15k.jsonl"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare Databricks Dolly 15k for Question -> Answer training."
    )
    parser.add_argument("--out_dir", default="data/dolly-qa-spm")
    parser.add_argument("--source_url", default=DEFAULT_DOLLY_URL)
    parser.add_argument("--train_split", type=float, default=0.95)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--vocab_size", type=int, default=4096)
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
    parser.add_argument("--context_word_limit", type=int, default=96)
    parser.add_argument(
        "--allowed_categories_json",
        default=None,
        help="Optional JSON array of Dolly categories to keep.",
    )
    return parser.parse_args()


def download_if_missing(url: str, out_path: Path):
    if out_path.exists():
        return
    request = Request(url, headers={"User-Agent": "lite-llm-pretraining/1.0"})
    with urlopen(request) as response, out_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def load_examples(source_path: Path, allowed_categories: list[str] | None = None):
    examples = []
    allowed = set(allowed_categories or [])
    decoder = json.JSONDecoder()
    text = source_path.read_text(encoding="utf-8")
    cursor = 0
    text_length = len(text)
    while cursor < text_length:
        while cursor < text_length and text[cursor].isspace():
            cursor += 1
        if cursor >= text_length:
            break
        item, cursor = decoder.raw_decode(text, cursor)
        if allowed and item.get("category") not in allowed:
            continue
        examples.append(
            {
                "instruction": item["instruction"].strip(),
                "context": item.get("context", "").strip(),
                "response": item["response"].strip(),
                "category": item.get("category", "").strip(),
            }
        )
    return examples


def trim_context(context: str, context_word_limit: int):
    if context_word_limit <= 0:
        return context.strip()
    words = context.split()
    return " ".join(words[:context_word_limit]).strip()


def format_example(
    example: dict,
    question_label: str,
    context_label: str,
    answer_label: str,
    instruction_text: str = "",
    context_word_limit: int = 96,
):
    question = example["instruction"].strip()
    context = trim_context(example.get("context", ""), context_word_limit)
    answer = example["response"].strip()
    parts = []
    if instruction_text.strip():
        parts.append(instruction_text.strip())
    parts.append(f"{question_label}: {question}")
    if context:
        parts.append(f"{context_label}: {context}")
    parts.append(f"{answer_label}: {answer}")
    return "\n".join(parts)


def prompt_prefix_text(
    example: dict,
    question_label: str,
    context_label: str,
    answer_label: str,
    instruction_text: str = "",
    context_word_limit: int = 96,
):
    question = example["instruction"].strip()
    context = trim_context(example.get("context", ""), context_word_limit)
    parts = []
    if instruction_text.strip():
        parts.append(instruction_text.strip())
    parts.append(f"{question_label}: {question}")
    if context:
        parts.append(f"{context_label}: {context}")
    parts.append(f"{answer_label}:")
    return "\n".join(parts)


def write_training_corpus(
    examples,
    out_path: Path,
    question_label: str,
    context_label: str,
    answer_label: str,
    instruction_text: str,
    context_word_limit: int,
):
    with out_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(
                format_example(
                    example,
                    question_label=question_label,
                    context_label=context_label,
                    answer_label=answer_label,
                    instruction_text=instruction_text,
                    context_word_limit=context_word_limit,
                )
            )
            handle.write("\n\n")


def encode_example_with_loss_mask(
    processor: spm.SentencePieceProcessor,
    example: dict,
    question_label: str,
    context_label: str,
    answer_label: str,
    instruction_text: str,
    prompt_loss_weight: float,
    continuation_head_token_count: int,
    continuation_head_loss_weight: float,
    context_word_limit: int,
):
    eos_id = processor.eos_id()
    formatted = format_example(
        example,
        question_label=question_label,
        context_label=context_label,
        answer_label=answer_label,
        instruction_text=instruction_text,
        context_word_limit=context_word_limit,
    )
    prefix = prompt_prefix_text(
        example,
        question_label=question_label,
        context_label=context_label,
        answer_label=answer_label,
        instruction_text=instruction_text,
        context_word_limit=context_word_limit,
    )
    token_ids = processor.encode(formatted, out_type=int)
    prefix_ids = processor.encode(prefix, out_type=int)
    prompt_token_count = min(len(prefix_ids), len(token_ids))
    continuation_token_count = max(0, len(token_ids) - prompt_token_count)
    head_token_count = min(max(0, continuation_head_token_count), continuation_token_count)
    tail_token_count = continuation_token_count - head_token_count
    loss_mask = [prompt_loss_weight] * prompt_token_count
    loss_mask.extend([continuation_head_loss_weight] * head_token_count)
    loss_mask.extend([1.0] * tail_token_count)
    if eos_id >= 0:
        token_ids.append(eos_id)
        loss_mask.append(1.0)
    return token_ids, loss_mask


def encode_split(
    examples,
    out_path: Path,
    loss_mask_out_path: Path,
    processor: spm.SentencePieceProcessor,
    question_label: str,
    context_label: str,
    answer_label: str,
    instruction_text: str,
    prompt_loss_weight: float,
    continuation_head_token_count: int,
    continuation_head_loss_weight: float,
    context_word_limit: int,
):
    total_tokens = 0
    with out_path.open("wb") as token_handle, loss_mask_out_path.open("wb") as mask_handle:
        for example in examples:
            token_ids, loss_mask = encode_example_with_loss_mask(
                processor,
                example,
                question_label=question_label,
                context_label=context_label,
                answer_label=answer_label,
                instruction_text=instruction_text,
                prompt_loss_weight=prompt_loss_weight,
                continuation_head_token_count=continuation_head_token_count,
                continuation_head_loss_weight=continuation_head_loss_weight,
                context_word_limit=context_word_limit,
            )
            np.asarray(token_ids, dtype=np.uint16).tofile(token_handle)
            np.asarray(loss_mask, dtype=np.float16).tofile(mask_handle)
            total_tokens += len(token_ids)
    return total_tokens


def save_examples_jsonl(path: Path, examples):
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False))
            handle.write("\n")


def prepare_dataset(
    out_dir: Path,
    source_url: str = DEFAULT_DOLLY_URL,
    train_split: float = 0.95,
    split_seed: int = 42,
    vocab_size: int = 4096,
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
    context_word_limit: int = 96,
    allowed_categories: list[str] | None = None,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    source_path = out_dir / "databricks-dolly-15k.jsonl"
    download_if_missing(source_url, source_path)
    examples = load_examples(source_path, allowed_categories=allowed_categories)
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
        suffix="-dolly-qa-train.txt",
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
        context_word_limit=context_word_limit,
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
        context_word_limit=context_word_limit,
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
        context_word_limit=context_word_limit,
    )

    category_counts = {}
    for example in shuffled_examples:
        category = example["category"] or "unknown"
        category_counts[category] = category_counts.get(category, 0) + 1

    meta = {
        "dataset": "databricks-dolly-15k",
        "task_format": "qa",
        "source_url": source_url,
        "num_examples": len(shuffled_examples),
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "category_counts": category_counts,
        "allowed_categories": allowed_categories,
        "question_label": question_label,
        "context_label": context_label,
        "answer_label": answer_label,
        "instruction_text": instruction_text,
        "prompt_loss_weight": prompt_loss_weight,
        "continuation_head_token_count": continuation_head_token_count,
        "continuation_head_loss_weight": continuation_head_loss_weight,
        "context_word_limit": context_word_limit,
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
    allowed_categories = None
    if args.allowed_categories_json:
        payload = json.loads(args.allowed_categories_json)
        if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
            raise ValueError("--allowed_categories_json must be a JSON array of strings")
        allowed_categories = payload
    meta = prepare_dataset(
        out_dir=Path(args.out_dir),
        source_url=args.source_url,
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
        context_word_limit=args.context_word_limit,
        allowed_categories=allowed_categories,
    )
    print(f"saved dataset to {args.out_dir}")
    print(f"train examples: {meta['train_examples']}, val examples: {meta['val_examples']}")
    print(f"train tokens: {meta['train_tokens']}, val tokens: {meta['val_tokens']}")


if __name__ == "__main__":
    main()
