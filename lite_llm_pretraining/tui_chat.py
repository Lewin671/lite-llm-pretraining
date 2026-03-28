import argparse
import curses
import textwrap
from pathlib import Path

from lite_llm_pretraining.app import ChatApplication, QAApplication, StoryApplication
from lite_llm_pretraining.model import CheckpointLanguageModel
from lite_llm_pretraining.story_inference import (
    PLAIN_STORY_TEMPLATE,
    QA_TEMPLATE,
    resolve_inference_profile,
)


CHAT_HELP_TEXT = "Enter send | /clear reset | /quit exit"
STORY_HELP_TEXT = "Enter a story opening | /clear reset | /quit exit"
QA_HELP_TEXT = "Enter question or question || context | /clear reset | /quit exit"


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive TUI chat for a local checkpoint.")
    parser.add_argument(
        "--checkpoint_dir",
        default="checkpoints/tinyshakespeare-byte-smoke/best",
        help="Directory containing weights.npz and model_config.json.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=160,
        help="Maximum number of generated tokens per assistant turn.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Optional top-k sampling cutoff.",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Optional repetition penalty greater than 1.0.",
    )
    parser.add_argument(
        "--repetition_window",
        type=int,
        default=None,
        help="Optional lookback window used by repetition penalty.",
    )
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "chat", "story", "qa"],
        help="Prompt format. auto uses task-specific mode from checkpoint metadata.",
    )
    return parser.parse_args()


class ChatTUI:
    def __init__(
        self,
        stdscr,
        app: ChatApplication,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        repetition_penalty: float,
        repetition_window: int | None,
    ):
        self.stdscr = stdscr
        self.app = app
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.repetition_window = repetition_window
        self.input_buffer = ""
        self.status = self._default_status()

    def _default_status(self):
        if isinstance(self.app, StoryApplication):
            return STORY_HELP_TEXT
        if isinstance(self.app, QAApplication):
            return QA_HELP_TEXT
        return CHAT_HELP_TEXT

    def _wrap_block(self, text: str, width: int):
        if not text:
            return [""]
        lines = []
        for raw_line in text.splitlines():
            wrapped = textwrap.wrap(raw_line, width=width, replace_whitespace=False)
            lines.extend(wrapped or [""])
        return lines or [""]

    def _render(self):
        self.stdscr.erase()
        height, width = self.stdscr.getmaxyx()
        transcript_height = max(3, height - 4)
        content_width = max(10, width - 2)

        lines = [f"Checkpoint step: {self.app.model.loaded_step}", ""]
        lines.extend(self.app.history_lines())
        transcript_lines = []
        for line in lines:
            transcript_lines.extend(self._wrap_block(line, content_width))
        visible = transcript_lines[-transcript_height:]

        for index, line in enumerate(visible):
            self.stdscr.addnstr(index, 0, line, width - 1)

        self.stdscr.hline(transcript_height, 0, "-", width)
        self.stdscr.addnstr(transcript_height + 1, 0, self.status, width - 1)
        input_line = f"> {self.input_buffer}"
        self.stdscr.addnstr(transcript_height + 2, 0, input_line, width - 1)
        cursor_x = min(len(input_line), width - 1)
        self.stdscr.move(transcript_height + 2, cursor_x)
        self.stdscr.refresh()

    def _run_command(self, command: str):
        if command == "/quit":
            return False
        if command == "/clear":
            self.app.clear()
            self.status = "conversation cleared"
            return True
        self.status = f"unknown command: {command}"
        return True

    def _submit(self):
        user_text = self.input_buffer.strip()
        self.input_buffer = ""
        if not user_text:
            self.status = self._default_status()
            return True
        if user_text.startswith("/"):
            return self._run_command(user_text)

        self.status = "assistant is generating..."
        self._render()
        for _piece in self.app.stream_reply(
            user_text,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            repetition_window=self.repetition_window,
        ):
            self._render()

        self.status = self._default_status()
        return True

    def run(self):
        curses.curs_set(1)
        self.stdscr.keypad(True)
        while True:
            self._render()
            key = self.stdscr.get_wch()
            if key in ("\n", "\r"):
                if not self._submit():
                    break
            elif key in (curses.KEY_BACKSPACE, "\b", "\x7f"):
                self.input_buffer = self.input_buffer[:-1]
            elif key == curses.KEY_RESIZE:
                continue
            elif isinstance(key, str) and key.isprintable():
                self.input_buffer += key


def resolve_tui_profile(checkpoint_dir: Path, requested_mode: str):
    profile = resolve_inference_profile(checkpoint_dir)
    if requested_mode == "chat":
        return {"mode": "chat", "prompt_template": PLAIN_STORY_TEMPLATE}
    if requested_mode == "story":
        return {
            "mode": "story",
            "prompt_template": profile.get("prompt_template", PLAIN_STORY_TEMPLATE),
        }
    if requested_mode == "qa":
        return {
            "mode": "qa",
            "prompt_template": profile.get("prompt_template", QA_TEMPLATE),
            "question_label": profile.get("question_label", "Question"),
            "context_label": profile.get("context_label", "Context"),
            "answer_label": profile.get("answer_label", "Answer"),
            "instruction_text": profile.get("instruction_text", ""),
            "answer_word_limit": profile.get("answer_word_limit"),
        }
    return profile


def run_tui(stdscr, args):
    checkpoint_dir = Path(args.checkpoint_dir)
    model = CheckpointLanguageModel(checkpoint_dir)
    profile = resolve_tui_profile(checkpoint_dir, args.mode)
    if profile.get("mode") == "story":
        app = StoryApplication(
            model,
            prompt_template=profile.get("prompt_template", PLAIN_STORY_TEMPLATE),
        )
    elif profile.get("mode") == "qa":
        app = QAApplication(
            model,
            question_prefix=profile.get("question_label", "Question"),
            answer_prefix=profile.get("answer_label", "Answer"),
            context_label=profile.get("context_label", "Context"),
            prompt_template=profile.get("prompt_template", QA_TEMPLATE),
            instruction_text=profile.get("instruction_text", ""),
            answer_word_limit=profile.get("answer_word_limit"),
        )
    else:
        app = ChatApplication(model)
    ChatTUI(
        stdscr,
        app,
        args.max_new_tokens,
        args.temperature,
        args.top_k,
        args.repetition_penalty,
        args.repetition_window,
    ).run()


def main():
    args = parse_args()
    curses.wrapper(run_tui, args)


if __name__ == "__main__":
    main()
