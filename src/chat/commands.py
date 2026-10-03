"""Парсер команд внутри интерактивного чата."""

from __future__ import annotations

from typing import Optional, Tuple

CHAT_COMMANDS = {
    "/state show": "показать текущий task state",
    "/state edit": "показать путь к task_state.json (откройте в редакторе)",
    "/state clear": "очистить task state (не историю!)",
    "/history show [N]": "последние N сообщений (по умолчанию 10)",
    "/history clear": "очистить историю сессии",
    "/exit": "выйти из чата (сессия сохраняется)",
    "/help": "показать команды чата",
}


def build_chat_help() -> str:
    lines = ["Команды чата:", ""]
    for cmd, desc in CHAT_COMMANDS.items():
        lines.append(f"  {cmd:<22} {desc}")
    lines.append("")
    lines.append("Любой другой текст отправляется ассистенту.")
    return "\n".join(lines)


def parse_chat_command(line: str) -> Optional[Tuple[str, list]]:
    """Вернуть (имя_команды, аргументы), если строка — команда чата; иначе None (сообщение)."""
    line = line.strip()
    if not line.startswith("/"):
        return None
    parts = line.split()
    cmd = parts[0]
    args = parts[1:]

    if cmd == "/exit":
        return ("exit", args)
    if cmd == "/help":
        return ("help", args)
    if cmd == "/state":
        sub = args[0] if args else "show"
        return (f"state_{sub}", args[1:])
    if cmd == "/history":
        sub = args[0] if args else "show"
        return (f"history_{sub}", args[1:])
    return ("unknown", [cmd])


__all__ = ["CHAT_COMMANDS", "build_chat_help", "parse_chat_command"]
