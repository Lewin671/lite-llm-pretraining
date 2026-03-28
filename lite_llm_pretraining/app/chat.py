from dataclasses import dataclass

from lite_llm_pretraining.model import CheckpointLanguageModel
from lite_llm_pretraining.story_inference import (
    PLAIN_STORY_TEMPLATE,
    QA_TEMPLATE,
    build_qa_prompt,
    build_story_prompt,
)


@dataclass
class ChatMessage:
    role: str
    content: str


class ChatApplication:
    def __init__(
        self,
        model: CheckpointLanguageModel,
        system_prompt: str | None = None,
        assistant_prefix: str = "Assistant",
        user_prefix: str = "User",
    ):
        self.model = model
        self.system_prompt = system_prompt or (
            "You are a tiny local language model running in a terminal chat."
        )
        self.assistant_prefix = assistant_prefix
        self.user_prefix = user_prefix
        self.messages: list[ChatMessage] = []

    def clear(self):
        self.messages.clear()

    def add_message(self, role: str, content: str):
        self.messages.append(ChatMessage(role=role, content=content))

    def history_lines(self):
        lines = [f"System: {self.system_prompt}", ""]
        for message in self.messages:
            if message.role == "user":
                prefix = self.user_prefix
            else:
                prefix = self.assistant_prefix
            lines.append(f"{prefix}: {message.content}")
            lines.append("")
        return lines

    def build_prompt(self):
        parts = [f"System: {self.system_prompt}", ""]
        for message in self.messages:
            label = self.user_prefix if message.role == "user" else self.assistant_prefix
            parts.append(f"{label}: {message.content}")
            parts.append("")
        parts.append(f"{self.assistant_prefix}: ")
        return "\n".join(parts)

    def generate_reply(
        self,
        user_text: str,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        repetition_penalty: float = 1.0,
        repetition_window: int | None = None,
    ):
        self.add_message("user", user_text)
        self.add_message("assistant", "")
        assistant_message = self.messages[-1]
        prompt = self.build_prompt()
        reply = self.model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
        )
        assistant_message.content = reply.strip()
        return assistant_message.content

    def stream_reply(
        self,
        user_text: str,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        repetition_penalty: float = 1.0,
        repetition_window: int | None = None,
    ):
        self.add_message("user", user_text)
        self.add_message("assistant", "")
        assistant_message = self.messages[-1]
        prompt = self.build_prompt()
        for piece in self.model.stream_generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
        ):
            assistant_message.content += piece
            yield piece
        assistant_message.content = assistant_message.content.strip()


class StoryApplication:
    def __init__(
        self,
        model: CheckpointLanguageModel,
        prompt_prefix: str = "Prompt",
        continuation_prefix: str = "Continuation",
        prompt_template: str = PLAIN_STORY_TEMPLATE,
    ):
        self.model = model
        self.prompt_prefix = prompt_prefix
        self.continuation_prefix = continuation_prefix
        self.prompt_template = prompt_template
        self.messages: list[ChatMessage] = []

    def clear(self):
        self.messages.clear()

    def add_message(self, role: str, content: str):
        self.messages.append(ChatMessage(role=role, content=content))

    def history_lines(self):
        lines = []
        for message in self.messages:
            prefix = (
                self.prompt_prefix
                if message.role == "prompt"
                else self.continuation_prefix
            )
            lines.append(f"{prefix}: {message.content}")
            lines.append("")
        return lines

    def build_prompt(self, prompt_text: str):
        return build_story_prompt(prompt_text, self.prompt_template)

    def generate_reply(
        self,
        user_text: str,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        repetition_penalty: float = 1.0,
        repetition_window: int | None = None,
    ):
        self.add_message("prompt", user_text)
        reply = self.model.generate(
            self.build_prompt(user_text),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
        )
        self.add_message("continuation", reply.strip())
        return reply.strip()

    def stream_reply(
        self,
        user_text: str,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        repetition_penalty: float = 1.0,
        repetition_window: int | None = None,
    ):
        self.add_message("prompt", user_text)
        self.add_message("continuation", "")
        continuation = self.messages[-1]
        for piece in self.model.stream_generate(
            self.build_prompt(user_text),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
        ):
            continuation.content += piece
            yield piece
        continuation.content = continuation.content.strip()


class QAApplication:
    def __init__(
        self,
        model: CheckpointLanguageModel,
        question_prefix: str = "Question",
        answer_prefix: str = "Answer",
        context_label: str = "Context",
        prompt_template: str = QA_TEMPLATE,
        instruction_text: str = "",
        context_separator: str = " || ",
    ):
        self.model = model
        self.question_prefix = question_prefix
        self.answer_prefix = answer_prefix
        self.context_label = context_label
        self.prompt_template = prompt_template
        self.instruction_text = instruction_text
        self.context_separator = context_separator
        self.messages: list[ChatMessage] = []

    def clear(self):
        self.messages.clear()

    def add_message(self, role: str, content: str):
        self.messages.append(ChatMessage(role=role, content=content))

    def history_lines(self):
        lines = []
        for message in self.messages:
            prefix = self.question_prefix if message.role == "question" else self.answer_prefix
            lines.append(f"{prefix}: {message.content}")
            lines.append("")
        return lines

    def split_question_and_context(self, user_text: str):
        question_text = user_text.strip()
        context_text = ""
        if self.context_separator in question_text:
            question_text, context_text = question_text.split(self.context_separator, 1)
        return question_text.strip(), context_text.strip()

    def build_prompt(self, question_text: str, context_text: str = ""):
        return build_qa_prompt(
            question_text,
            prompt_template=self.prompt_template,
            context=context_text,
            question_label=self.question_prefix,
            context_label=self.context_label,
            answer_label=self.answer_prefix,
            instruction_text=self.instruction_text,
        )

    def generate_reply(
        self,
        user_text: str,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        repetition_penalty: float = 1.0,
        repetition_window: int | None = None,
    ):
        question_text, context_text = self.split_question_and_context(user_text)
        history_question = question_text
        if context_text:
            history_question = f"{question_text}\n{self.context_label}: {context_text}"
        self.add_message("question", history_question)
        reply = self.model.generate(
            self.build_prompt(question_text, context_text=context_text),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
        )
        self.add_message("answer", reply.strip())
        return reply.strip()

    def stream_reply(
        self,
        user_text: str,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        repetition_penalty: float = 1.0,
        repetition_window: int | None = None,
    ):
        question_text, context_text = self.split_question_and_context(user_text)
        history_question = question_text
        if context_text:
            history_question = f"{question_text}\n{self.context_label}: {context_text}"
        self.add_message("question", history_question)
        self.add_message("answer", "")
        answer = self.messages[-1]
        for piece in self.model.stream_generate(
            self.build_prompt(question_text, context_text=context_text),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
        ):
            answer.content += piece
            yield piece
        answer.content = answer.content.strip()
