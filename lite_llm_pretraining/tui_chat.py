import argparse
import curses
import textwrap
from pathlib import Path

from lite_llm_pretraining.app import ChatApplication, StoryApplication
from lite_llm_pretraining.common import load_json
from lite_llm_pretraining.model import CheckpointLanguageModel


CHAT_HELP_TEXT = "Enter send | /clear reset | /quit exit"
STORY_HELP_TEXT = "Enter a story opening | /clear reset | /quit exit"


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
        "--mode",
        default="auto",
        choices=["auto", "chat", "story"],
        help="Prompt format. auto uses story mode for TinyStories checkpoints.",
    )
    return parser.parse_args()


class ChatTUI:
    def __init__(self, stdscr, app: ChatApplication, max_new_tokens: int, temperature: float):
        self.stdscr = stdscr
        self.app = app
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.input_buffer = ""
        self.status = STORY_HELP_TEXT if isinstance(app, StoryApplication) else CHAT_HELP_TEXT

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
            self.status = (
                STORY_HELP_TEXT if isinstance(self.app, StoryApplication) else CHAT_HELP_TEXT
            )
            return True
        if user_text.startswith("/"):
            return self._run_command(user_text)

        self.status = "assistant is generating..."
        self._render()
        for _piece in self.app.stream_reply(
            user_text,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        ):
            self._render()

        self.status = (
            STORY_HELP_TEXT if isinstance(self.app, StoryApplication) else CHAT_HELP_TEXT
        )
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


def infer_mode(checkpoint_dir: Path, requested_mode: str):
    if requested_mode != "auto":
        return requested_mode
    run_config_path = checkpoint_dir.parent / "run_config.json"
    if run_config_path.exists():
        config = load_json(run_config_path)
        if "tinystories" in str(config.get("data_dir", "")).lower():
            return "story"
    return "story" if "tinystories" in str(checkpoint_dir).lower() else "chat"


def run_tui(stdscr, args):
    checkpoint_dir = Path(args.checkpoint_dir)
    model = CheckpointLanguageModel(checkpoint_dir)
    mode = infer_mode(checkpoint_dir, args.mode)
    if mode == "story":
        app = StoryApplication(model)
    else:
        app = ChatApplication(model)
    ChatTUI(stdscr, app, args.max_new_tokens, args.temperature).run()


def main():
    args = parse_args()
    curses.wrapper(run_tui, args)


if __name__ == "__main__":
    main()
