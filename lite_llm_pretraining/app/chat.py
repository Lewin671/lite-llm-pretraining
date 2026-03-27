from dataclasses import dataclass

from lite_llm_pretraining.model import CheckpointLanguageModel


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

    def generate_reply(self, user_text: str, max_new_tokens: int, temperature: float = 1.0):
        self.add_message("user", user_text)
        self.add_message("assistant", "")
        assistant_message = self.messages[-1]
        prompt = self.build_prompt()
        reply = self.model.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
        assistant_message.content = reply.strip()
        return assistant_message.content

    def stream_reply(self, user_text: str, max_new_tokens: int, temperature: float = 1.0):
        self.add_message("user", user_text)
        self.add_message("assistant", "")
        assistant_message = self.messages[-1]
        prompt = self.build_prompt()
        for piece in self.model.stream_generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        ):
            assistant_message.content += piece
            yield piece
        assistant_message.content = assistant_message.content.strip()
