import json
import math
import random
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from lite_llm_pretraining.tokenizer import ByteTokenizer


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


def load_loss_mask(data_dir: Path, split: str, meta=None):
    meta = meta or {}
    if not meta.get("has_loss_mask"):
        return None
    mask_path = data_dir / f"{split}_loss_mask.bin"
    if not mask_path.exists():
        return None
    mask_dtype = meta.get("loss_mask_dtype", "uint8")
    return np.memmap(mask_path, dtype=np.dtype(mask_dtype), mode="r")


def example_start_positions(data, eos_token_id: int, context_size: int):
    eos_positions = np.flatnonzero(np.asarray(data) == eos_token_id)
    if eos_positions.size == 0:
        return np.asarray([0], dtype=np.int64)
    starts = np.concatenate(
        [np.asarray([0], dtype=np.int64), eos_positions[:-1].astype(np.int64) + 1]
    )
    valid = (eos_positions.astype(np.int64) - starts) >= context_size
    starts = starts[valid]
    if starts.size == 0:
        raise ValueError(
            "no example-aligned windows fit the requested context size; "
            f"context_size={context_size}"
        )
    return starts


def loss_window_start_positions(data, loss_mask_data, context_size: int):
    max_start = len(data) - context_size - 1
    if max_start <= 0:
        raise ValueError(
            f"dataset is too small for context_size={context_size}: {len(data)} tokens"
        )
    mask = np.asarray(loss_mask_data, dtype=np.float32)
    prefix = np.concatenate([np.asarray([0.0], dtype=np.float32), np.cumsum(mask)])
    starts = np.arange(max_start + 1, dtype=np.int64)
    window_sums = prefix[starts + context_size + 1] - prefix[starts + 1]
    valid = window_sums > 0
    starts = starts[valid]
    if starts.size == 0:
        raise ValueError(
            "no loss-bearing windows fit the requested context size; "
            f"context_size={context_size}"
        )
    return starts


def get_batch(
    data,
    batch_size: int,
    context_size: int,
    loss_mask_data=None,
    start_positions=None,
):
    max_start = len(data) - context_size - 1
    if max_start <= 0:
        raise ValueError(
            f"dataset is too small for context_size={context_size}: {len(data)} tokens"
        )
    if start_positions is not None:
        indices = np.random.randint(0, len(start_positions), size=batch_size)
        starts = start_positions[indices]
    else:
        starts = np.random.randint(0, max_start + 1, size=batch_size)
    x = np.stack([data[idx : idx + context_size] for idx in starts]).astype(np.int32)
    y = np.stack([data[idx + 1 : idx + context_size + 1] for idx in starts]).astype(
        np.int32
    )
    loss_mask = None
    if loss_mask_data is not None:
        loss_mask = np.stack(
            [
                loss_mask_data[idx + 1 : idx + context_size + 1]
                for idx in starts
            ]
        ).astype(np.float32)
    return mx.array(x), mx.array(y), mx.array(loss_mask) if loss_mask is not None else None


def loss_fn(model: TransformerLM, x, y, loss_mask=None):
    logits = model(x)
    if loss_mask is None:
        return nn.losses.cross_entropy(logits, y, reduction="mean")
    token_losses = nn.losses.cross_entropy(logits, y, reduction="none")
    weighted_loss = token_losses * loss_mask
    normalizer = mx.maximum(loss_mask.sum(), mx.array(1.0, dtype=loss_mask.dtype))
    return weighted_loss.sum() / normalizer


def estimate_loss(
    model: TransformerLM,
    data,
    batch_size: int,
    context_size: int,
    steps: int,
    loss_mask_data=None,
    start_positions=None,
):
    model.eval()
    losses = []
    for _ in range(steps):
        x, y, loss_mask = get_batch(
            data,
            batch_size,
            context_size,
            loss_mask_data=loss_mask_data,
            start_positions=start_positions,
        )
        loss = loss_fn(model, x, y, loss_mask)
        mx.eval(loss)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def learning_rate_at(
    step: int,
    base_lr: float,
    warmup_steps: int,
    min_lr: float | None = None,
    decay_steps: int | None = None,
):
    if warmup_steps > 0 and step <= warmup_steps:
        return min(1.0, step / warmup_steps) * base_lr

    if min_lr is None or decay_steps is None or decay_steps <= warmup_steps:
        return base_lr

    if step >= decay_steps:
        return min_lr

    decay_progress = (step - warmup_steps) / max(1, decay_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return min_lr + (base_lr - min_lr) * cosine


def count_parameters(model: TransformerLM):
    mx.eval(model.parameters())
    return sum(param.size for _, param in tree_flatten(model.parameters()))


def save_checkpoint(
    checkpoint_dir: Path,
    model: TransformerLM,
    step: int,
    extra_state,
    tokenizer=None,
):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(checkpoint_dir / "weights.npz"))
    save_json(checkpoint_dir / "model_config.json", model.config_dict())
    save_json(checkpoint_dir / "state.json", {"step": step, **extra_state})
    if tokenizer is not None:
        tokenizer.save_to_checkpoint(checkpoint_dir)


def load_checkpoint(checkpoint_dir: Path):
    model_config = load_json(checkpoint_dir / "model_config.json")
    model = TransformerLM(**model_config)
    model.load_weights(str(checkpoint_dir / "weights.npz"))
    mx.eval(model.parameters())
    state_path = checkpoint_dir / "state.json"
    state = load_json(state_path) if state_path.exists() else {}
    return model, model_config, state


def encode_text(text: str):
    return ByteTokenizer().encode(text)


def decode_tokens(tokens):
    return ByteTokenizer().decode(tokens)


def sample_text(
    model: TransformerLM,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    include_prompt: bool = True,
    tokenizer=None,
    top_k: int | None = None,
    repetition_penalty: float = 1.0,
    repetition_window: int | None = None,
):
    return "".join(
        sample_text_stream(
            model,
            prompt,
            max_new_tokens,
            temperature=temperature,
            include_prompt=include_prompt,
            tokenizer=tokenizer,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
        )
    )


def apply_decoding_constraints(
    logits,
    token_history,
    top_k: int | None = None,
    repetition_penalty: float = 1.0,
    repetition_window: int | None = None,
):
    logits_np = np.array(logits)

    if repetition_penalty and repetition_penalty > 1.0:
        recent_tokens = token_history[-repetition_window:] if repetition_window else token_history
        for token in set(recent_tokens):
            value = logits_np[0, token]
            logits_np[0, token] = (
                value * repetition_penalty if value < 0 else value / repetition_penalty
            )

    if top_k is not None and 0 < top_k < logits_np.shape[-1]:
        threshold = np.partition(logits_np[0], -top_k)[-top_k]
        logits_np[0, logits_np[0] < threshold] = -np.inf

    return mx.array(logits_np)


def sample_text_stream(
    model: TransformerLM,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    include_prompt: bool = True,
    tokenizer=None,
    top_k: int | None = None,
    repetition_penalty: float = 1.0,
    repetition_window: int | None = None,
):
    tokenizer = tokenizer or ByteTokenizer()
    prompt_tokens = tokenizer.encode(prompt) or tokenizer.default_prompt_tokens()
    tokens = list(prompt_tokens)
    generated_tokens = []
    decoded_generated = ""
    model.eval()

    if prompt and include_prompt:
        yield prompt

    for _ in range(max_new_tokens):
        x = mx.array([tokens[-model.context_size :]], dtype=mx.int32)
        logits = model(x)[:, -1, :]
        logits = apply_decoding_constraints(
            logits,
            tokens,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
        )
        if temperature > 0 and temperature != 1.0:
            logits = logits / temperature
        next_token = (
            mx.argmax(logits, axis=-1)
            if temperature <= 0
            else mx.random.categorical(logits)
        )
        mx.eval(next_token)
        token_value = int(next_token.item())
        tokens.append(token_value)
        generated_tokens.append(token_value)

        current_text = tokenizer.decode(generated_tokens)
        if current_text.startswith(decoded_generated):
            piece = current_text[len(decoded_generated) :]
        else:
            piece = current_text
        decoded_generated = current_text
        if piece:
            yield piece


def perplexity(loss_value: float):
    return math.exp(min(loss_value, 20.0))
