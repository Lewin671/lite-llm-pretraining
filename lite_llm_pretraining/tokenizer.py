import shutil
import json
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ByteTokenizer:
    name: str = "utf8-bytes"
    vocab_size: int = 256
    eos_token_id: int | None = None

    def encode(self, text: str):
        return list(text.encode("utf-8"))

    def decode(self, token_ids):
        return bytes(int(token_id) for token_id in token_ids).decode(
            "utf-8", errors="ignore"
        )

    def config_dict(self):
        return {"name": self.name, "vocab_size": self.vocab_size}

    def default_prompt_tokens(self):
        return [10]

    def save_to_checkpoint(self, checkpoint_dir: Path):
        save_tokenizer_assets(checkpoint_dir, self.config_dict(), source_dir=None)


class SentencePieceTokenizer:
    def __init__(self, model_path: Path):
        import sentencepiece as spm

        self.model_path = Path(model_path)
        self.processor = spm.SentencePieceProcessor(model_file=str(self.model_path))
        self.name = "sentencepiece"
        self.vocab_size = self.processor.vocab_size()
        eos_id = self.processor.eos_id()
        self.eos_token_id = eos_id if eos_id >= 0 else None

    def encode(self, text: str):
        return self.processor.encode(text, out_type=int)

    def decode(self, token_ids):
        return self.processor.decode([int(token_id) for token_id in token_ids])

    def config_dict(self):
        return {
            "name": self.name,
            "model_file": self.model_path.name,
            "vocab_size": self.vocab_size,
            "eos_token_id": self.eos_token_id,
        }

    def default_prompt_tokens(self):
        if self.eos_token_id is not None:
            return [self.eos_token_id]
        return self.encode("\n") or [0]

    def save_to_checkpoint(self, checkpoint_dir: Path):
        save_tokenizer_assets(
            checkpoint_dir, self.config_dict(), source_dir=self.model_path.parent
        )


def tokenizer_spec_from_meta(meta):
    tokenizer_meta = meta.get("tokenizer", "utf8-bytes")
    if isinstance(tokenizer_meta, str):
        return {"name": tokenizer_meta, "vocab_size": meta.get("vocab_size", 256)}
    return tokenizer_meta


def load_tokenizer(tokenizer_meta, base_dir: Path | None = None):
    name = tokenizer_meta.get("name", "utf8-bytes")
    if name == "utf8-bytes":
        return ByteTokenizer(vocab_size=tokenizer_meta.get("vocab_size", 256))
    if name in {"sentencepiece", "sentencepiece-bpe"}:
        if base_dir is None:
            raise ValueError("base_dir is required to load sentencepiece tokenizer")
        model_file = tokenizer_meta.get("model_file", "tokenizer.model")
        return SentencePieceTokenizer(Path(base_dir) / model_file)
    raise ValueError(f"unsupported tokenizer: {name}")


def load_tokenizer_from_meta(meta, data_dir: Path):
    return load_tokenizer(tokenizer_spec_from_meta(meta), base_dir=data_dir)


def load_tokenizer_from_checkpoint(checkpoint_dir: Path):
    tokenizer_meta_path = checkpoint_dir / "tokenizer_meta.json"
    if not tokenizer_meta_path.exists():
        return ByteTokenizer()
    tokenizer_meta = json.loads(tokenizer_meta_path.read_text(encoding="utf-8"))
    return load_tokenizer(tokenizer_meta, base_dir=checkpoint_dir)


def save_tokenizer_assets(checkpoint_dir: Path, tokenizer_meta, source_dir: Path | None = None):
    tokenizer_meta = dict(tokenizer_meta)
    model_file = tokenizer_meta.get("model_file")
    if model_file and source_dir is not None:
        source_path = Path(source_dir) / model_file
        if source_path.exists():
            shutil.copy2(source_path, checkpoint_dir / Path(model_file).name)
            tokenizer_meta["model_file"] = Path(model_file).name
    tokenizer_meta_path = checkpoint_dir / "tokenizer_meta.json"
    tokenizer_meta_path.write_text(
        json.dumps(tokenizer_meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
