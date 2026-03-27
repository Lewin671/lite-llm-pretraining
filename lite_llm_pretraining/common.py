import codecs
import json
import math
import random
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten


DEFAULT_TOKEN_DTYPE = "uint16"


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_size: int,
        dim: int,
        num_layers: int,
        num_heads: int,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.dim = dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.gradient_checkpointing = gradient_checkpointing

        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(context_size, dim)
        self.transformer = nn.TransformerEncoder(
            num_layers,
            dim,
            num_heads,
            norm_first=True,
            checkpoint=gradient_checkpointing,
        )
        self.final_norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def __call__(self, x):
        length = x.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(length)
        positions = mx.arange(length, dtype=mx.int32)
        hidden = self.token_embedding(x) + self.position_embedding(positions)
        hidden = self.transformer(hidden, mask)
        hidden = self.final_norm(hidden)
        return self.lm_head(hidden)

    def config_dict(self):
        return {
            "vocab_size": self.vocab_size,
            "context_size": self.context_size,
            "dim": self.dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "gradient_checkpointing": self.gradient_checkpointing,
        }


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)


def token_dtype_from_meta(meta):
    return meta.get("token_dtype", DEFAULT_TOKEN_DTYPE)


def load_memmap(data_dir: Path, split: str, token_dtype: str = DEFAULT_TOKEN_DTYPE):
    return np.memmap(data_dir / f"{split}.bin", dtype=np.dtype(token_dtype), mode="r")


def get_batch(data, batch_size: int, context_size: int):
    max_start = len(data) - context_size - 1
    if max_start <= 0:
        raise ValueError(
            f"dataset is too small for context_size={context_size}: {len(data)} tokens"
        )
    starts = np.random.randint(0, max_start + 1, size=batch_size)
    x = np.stack([data[idx : idx + context_size] for idx in starts]).astype(np.int32)
    y = np.stack([data[idx + 1 : idx + context_size + 1] for idx in starts]).astype(
        np.int32
    )
    return mx.array(x), mx.array(y)


def loss_fn(model: TransformerLM, x, y):
    logits = model(x)
    return nn.losses.cross_entropy(logits, y, reduction="mean")


def estimate_loss(model: TransformerLM, data, batch_size: int, context_size: int, steps: int):
    model.eval()
    losses = []
    for _ in range(steps):
        x, y = get_batch(data, batch_size, context_size)
        loss = loss_fn(model, x, y)
        mx.eval(loss)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def learning_rate_at(step: int, base_lr: float, warmup_steps: int):
    if warmup_steps <= 0:
        return base_lr
    return min(1.0, step / warmup_steps) * base_lr


def count_parameters(model: TransformerLM):
    mx.eval(model.parameters())
    return sum(param.size for _, param in tree_flatten(model.parameters()))


def save_checkpoint(checkpoint_dir: Path, model: TransformerLM, step: int, extra_state):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(checkpoint_dir / "weights.npz"))
    save_json(checkpoint_dir / "model_config.json", model.config_dict())
    save_json(checkpoint_dir / "state.json", {"step": step, **extra_state})


def load_checkpoint(checkpoint_dir: Path):
    model_config = load_json(checkpoint_dir / "model_config.json")
    model = TransformerLM(**model_config)
    model.load_weights(str(checkpoint_dir / "weights.npz"))
    mx.eval(model.parameters())
    state_path = checkpoint_dir / "state.json"
    state = load_json(state_path) if state_path.exists() else {}
    return model, model_config, state


def encode_text(text: str):
    return list(text.encode("utf-8"))


def decode_tokens(tokens):
    return bytes(int(token) for token in tokens).decode("utf-8", errors="ignore")


def sample_text(
    model: TransformerLM,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    include_prompt: bool = True,
):
    return "".join(
        sample_text_stream(
            model,
            prompt,
            max_new_tokens,
            temperature=temperature,
            include_prompt=include_prompt,
        )
    )


def sample_text_stream(
    model: TransformerLM,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    include_prompt: bool = True,
):
    prompt_tokens = encode_text(prompt) or [10]
    tokens = list(prompt_tokens)
    decoder = codecs.getincrementaldecoder("utf-8")("ignore")
    model.eval()

    if prompt and include_prompt:
        yield prompt

    for _ in range(max_new_tokens):
        x = mx.array([tokens[-model.context_size :]], dtype=mx.int32)
        logits = model(x)[:, -1, :]
        if temperature != 1.0:
            logits = logits / temperature
        next_token = mx.random.categorical(logits)
        mx.eval(next_token)
        token_value = int(next_token.item())
        tokens.append(token_value)

        piece = decoder.decode(bytes([token_value]), final=False)
        if piece:
            yield piece

    tail = decoder.decode(b"", final=True)
    if tail:
        yield tail


def perplexity(loss_value: float):
    return math.exp(min(loss_value, 20.0))
