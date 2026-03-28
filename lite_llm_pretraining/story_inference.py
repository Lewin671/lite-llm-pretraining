from pathlib import Path

from lite_llm_pretraining.common import load_json


PLAIN_STORY_TEMPLATE = "{prompt}"
PROMPT_CONTINUATION_TEMPLATE = "Prompt: {prompt}\nContinuation:"
QA_TEMPLATE = "Question: {question}\n{context_block}Answer:"


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
            "question_label": inference.get("question_label", "Question"),
            "context_label": inference.get("context_label", "Context"),
            "answer_label": inference.get("answer_label", "Answer"),
            "instruction_text": inference.get("instruction_text", ""),
            "answer_word_limit": inference.get("answer_word_limit"),
        }

    prepare = config.get("prepare", {})
    if prepare.get("name") == "dolly_qa" or prepare.get("task_format") == "qa":
        answer_word_limit = None
        if prepare.get("normalize_factoid_answers"):
            answer_word_limit = prepare.get("max_normalized_answer_words")
        return {
            "mode": "qa",
            "question_label": prepare.get("question_label", "Question"),
            "context_label": prepare.get("context_label", "Context"),
            "answer_label": prepare.get("answer_label", "Answer"),
            "instruction_text": prepare.get("instruction_text", "").strip(),
            "prompt_template": prepare.get("prompt_template", QA_TEMPLATE),
            "answer_word_limit": answer_word_limit,
        }
    if prepare.get("story_format") == "prompt_continuation":
        prompt_label = prepare.get("prompt_label", "Prompt")
        continuation_label = prepare.get("continuation_label", "Continuation")
        instruction_text = prepare.get("instruction_text", "").strip()
        template_lines = [f"{prompt_label}: {{prompt}}"]
        if instruction_text:
            template_lines.append(instruction_text)
        template_lines.append(f"{continuation_label}:")
        return {
            "mode": "story",
            "prompt_template": "\n".join(template_lines),
        }

    data_dir = str(config.get("data_dir", "")).lower()
    if "dolly" in data_dir or "qa" in path_hint.lower():
        return {
            "mode": "qa",
            "question_label": "Question",
            "context_label": "Context",
            "answer_label": "Answer",
            "instruction_text": "",
            "prompt_template": QA_TEMPLATE,
        }
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


def build_qa_prompt(
    question: str,
    prompt_template: str = QA_TEMPLATE,
    context: str | None = None,
    question_label: str = "Question",
    context_label: str = "Context",
    answer_label: str = "Answer",
    instruction_text: str = "",
):
    question = question.strip()
    context = (context or "").strip()
    rendered_template = prompt_template or QA_TEMPLATE
    context_block = ""
    if context:
        context_block = f"{context_label}: {context}\n"
    if "{" in rendered_template:
        return rendered_template.format(
            prompt=question,
            question=question,
            context=context,
            context_block=context_block,
            question_label=question_label,
            context_label=context_label,
            answer_label=answer_label,
            instruction_text=instruction_text.strip(),
        )

    parts = []
    if instruction_text.strip():
        parts.append(instruction_text.strip())
    parts.append(f"{question_label}: {question}")
    if context:
        parts.append(f"{context_label}: {context}")
    parts.append(f"{answer_label}:")
    return "\n".join(parts)


def build_prompt_from_profile(prompt: str, profile: dict, context: str | None = None):
    mode = profile.get("mode")
    if mode == "story":
        return build_story_prompt(
            prompt,
            profile.get("prompt_template", PLAIN_STORY_TEMPLATE),
        )
    if mode == "qa":
        return build_qa_prompt(
            prompt,
            prompt_template=profile.get("prompt_template", QA_TEMPLATE),
            context=context,
            question_label=profile.get("question_label", "Question"),
            context_label=profile.get("context_label", "Context"),
            answer_label=profile.get("answer_label", "Answer"),
            instruction_text=profile.get("instruction_text", ""),
        )
    return prompt


def extract_qa_answer(text: str, answer_word_limit: int | None = None):
    cleaned = text.strip()
    if not cleaned:
        return ""
    if cleaned.lower().startswith("answer:"):
        cleaned = cleaned.split(":", 1)[1].strip()
    first_line = cleaned.splitlines()[0].strip()
    if not first_line:
        return ""
    if first_line.lower().startswith(("question:", "context:", "user:", "assistant:")):
        return ""
    if answer_word_limit is not None and answer_word_limit > 0:
        words = first_line.split()
        first_line = " ".join(words[:answer_word_limit]).strip()
    return first_line
