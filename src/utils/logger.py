"""
Terminal output for the QA Agent — one shared Rich ``Console`` for the whole app.

- **Workflow logs:** ``get_logger(...)`` → ``log.info``, ``log.start``, ``log.end``, etc.
- **UI (panels, tables, prompts):** import ``console`` and use ``console.print(...)``

    from utils.logger import console, get_logger

    log = get_logger("extract_files")
    log.start()

    console.print(Panel.fit("QA Agent"))
"""

import sys
from datetime import datetime

from rich.console import Console
from rich.text import Text

# Reconfigure standard output streams to UTF-8 to prevent UnicodeEncodeError on Windows terminals
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        try:
            stream.reconfigure(encoding='utf-8')
        except Exception:
            pass

console = Console(width=160, soft_wrap=False)

_LEVEL_STYLE = {
    "DEBUG": "dim",
    "INFO": "cyan",
    "START": "bold green",
    "END": "bold blue",
    "WARN": "bold yellow",
    "ERROR": "bold red",
}


class Logger:
    def __init__(self, name: str) -> None:
        self.name = name

    def _fmt(self, msg: str, *args) -> str:
        return msg % args if args else msg

    def _log_line(self, level: str, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        source = self.name[:32].ljust(32)
        style = _LEVEL_STYLE.get(level, "white")

        line = Text()
        line.append(f"[{timestamp}] ", style="dim")
        line.append(f"{level:<7} ", style=style)
        line.append(source, style="bold")
        line.append("  ", style="")
        line.append(message)

        console.print(line, soft_wrap=False, overflow="ignore", crop=False)

    def section(self, title: str) -> None:
        """Visual separator between workflow phases."""
        bar = f"=== {title} " + "=" * max(0, 60 - len(title))
        console.print(f"\n[bold]{bar}[/bold]", soft_wrap=False)

    def info(self, msg: str, *args) -> None:
        self._log_line("INFO", self._fmt(msg, *args))

    def start(self, msg: str = "Execution started", *args) -> None:
        self._log_line("START", self._fmt(msg, *args))

    def end(self, msg: str = "Execution completed", *args) -> None:
        self._log_line("END", self._fmt(msg, *args))

    def warn(self, msg: str, *args) -> None:
        self._log_line("WARN", self._fmt(msg, *args))

    def warning(self, msg: str, *args) -> None:
        self.warn(msg, *args)

    def error(self, msg: str, *args) -> None:
        self._log_line("ERROR", self._fmt(msg, *args))


def get_logger(name: str) -> Logger:
    return Logger(name)
