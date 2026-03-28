from pathlib import Path

from lite_llm_pretraining.common import load_checkpoint, sample_text, sample_text_stream
from lite_llm_pretraining.tokenizer import load_tokenizer_from_checkpoint


class CheckpointLanguageModel:
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.model, self.model_config, self.state = load_checkpoint(checkpoint_dir)
        self.tokenizer = load_tokenizer_from_checkpoint(checkpoint_dir)

    @property
    def loaded_step(self):
        return self.state.get("step", "unknown")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        repetition_penalty: float = 1.0,
        repetition_window: int | None = None,
    ):
        return sample_text(
            self.model,
            prompt,
            max_new_tokens,
            temperature=temperature,
            include_prompt=False,
            tokenizer=self.tokenizer,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
        )

    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        repetition_penalty: float = 1.0,
        repetition_window: int | None = None,
    ):
        return sample_text_stream(
            self.model,
            prompt,
            max_new_tokens,
            temperature=temperature,
            include_prompt=False,
            tokenizer=self.tokenizer,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
        )
