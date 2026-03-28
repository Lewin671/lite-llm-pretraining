from pathlib import Path

from lite_llm_pretraining.common import load_json


PLAIN_STORY_TEMPLATE = "{prompt}"
PROMPT_CONTINUATION_TEMPLATE = "Prompt: {prompt}\nContinuation:"


def checkpoint_run_config_path(checkpoint_dir: Path):
    return checkpoint_dir.parent / "run_config.json"


def load_checkpoint_run_config(checkpoint_dir: Path):
    run_config_path = checkpoint_run_config_path(checkpoint_dir)
    if not run_config_path.exists():
        return {}
    return load_json(run_config_path)


def resolve_inference_profile_from_config(config, path_hint: str = ""):
    inference = config.get("inference", {})
    if inference:
        return {
            "mode": inference.get("mode", "chat"),
            "prompt_template": inference.get("prompt_template", PLAIN_STORY_TEMPLATE),
        }

    prepare = config.get("prepare", {})
    if prepare.get("story_format") == "prompt_continuation":
        return {
            "mode": "story",
            "prompt_template": PROMPT_CONTINUATION_TEMPLATE,
        }

    data_dir = str(config.get("data_dir", "")).lower()
    if "tinystories" in data_dir or "tinystories" in path_hint.lower():
        return {
            "mode": "story",
            "prompt_template": PLAIN_STORY_TEMPLATE,
        }

    return {
        "mode": "chat",
        "prompt_template": PLAIN_STORY_TEMPLATE,
    }


def resolve_inference_profile(checkpoint_dir: Path):
    config = load_checkpoint_run_config(checkpoint_dir)
    return resolve_inference_profile_from_config(config, path_hint=str(checkpoint_dir))


def build_story_prompt(prompt: str, prompt_template: str):
    return prompt_template.format(prompt=prompt.strip())
