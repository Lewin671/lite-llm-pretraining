from pathlib import Path

from lite_llm_pretraining.common import load_checkpoint, sample_text, sample_text_stream


class CheckpointLanguageModel:
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.model, self.model_config, self.state = load_checkpoint(checkpoint_dir)

    @property
    def loaded_step(self):
        return self.state.get("step", "unknown")

    def generate(self, prompt: str, max_new_tokens: int, temperature: float = 1.0):
        return sample_text(
            self.model,
            prompt,
            max_new_tokens,
            temperature=temperature,
            include_prompt=False,
        )

    def stream_generate(
        self, prompt: str, max_new_tokens: int, temperature: float = 1.0
    ):
        return sample_text_stream(
            self.model,
            prompt,
            max_new_tokens,
            temperature=temperature,
            include_prompt=False,
        )

