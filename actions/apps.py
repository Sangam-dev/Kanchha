from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from shutil import which


@dataclass(slots=True)
class AppLaunchResult:
    success: bool
    message: str


APP_ALIASES: dict[str, tuple[str, ...]] = {
    "browser": ("chrome.exe", "chrome", "google-chrome", "firefox", "chromium", "chromium-browser"),
    "chrome": ("chrome.exe", "chrome", "google-chrome", "google-chrome-stable", "chromium", "chromium-browser"),
    "edge": ("msedge.exe", "msedge", "microsoft-edge", "edge"),
    "microsoft edge": ("msedge.exe", "msedge", "microsoft-edge", "edge"),
    "firefox": ("firefox.exe", "firefox"),
    "notepad": ("notepad.exe", "notepad", "gedit", "kate", "xed"),
    "calculator": ("calc.exe", "gnome-calculator", "kcalc", "qalculate-gtk", "calculator"),
    "paint": ("mspaint.exe", "mspaint", "pinta", "kolourpaint", "drawing"),
    "excel": ("excel.exe", "excel", "libreoffice", "localc"),
    "command prompt": ("cmd.exe", "cmd", "x-terminal-emulator", "gnome-terminal", "konsole"),
    "terminal": ("wt.exe", "powershell.exe", "x-terminal-emulator", "gnome-terminal", "konsole", "xterm"),
    "files": ("explorer.exe", "nautilus", "dolphin", "thunar", "pcmanfm", "xdg-open"),
    "file manager": ("explorer.exe", "nautilus", "dolphin", "thunar", "pcmanfm", "xdg-open"),
    "file explorer": ("explorer.exe", "nautilus", "dolphin", "thunar", "xdg-open"),
    "explorer": ("explorer.exe", "nautilus", "dolphin", "thunar", "xdg-open"),
    "settings": ("ms-settings:", "gnome-control-center", "systemsettings", "xfce4-settings-manager", "cinnamon-settings"),
    "system settings": ("ms-settings:", "gnome-control-center", "systemsettings", "xfce4-settings-manager", "cinnamon-settings"),
    "spotify": ("spotify.exe", "spotify"),
    "discord": ("discord.exe", "discord"),
    "vscode": ("code.exe", "code"),
    "visual studio code": ("code.exe", "code"),
}

KNOWN_PATHS: dict[str, tuple[Path, ...]] = {
    "chrome": (
        Path(os.environ.get("PROGRAMFILES", "")) / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Chrome/Application/chrome.exe",
    ),
    "edge": (
        Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("PROGRAMFILES", "")) / "Microsoft/Edge/Application/msedge.exe",
    ),
}


def _normalize_app_name(app_name: str) -> str:
    return " ".join(app_name.lower().strip().split())


def _candidate_commands(app_name: str) -> tuple[str, ...]:
    normalized = _normalize_app_name(app_name)
    return APP_ALIASES.get(normalized, (normalized,))


def _candidate_paths(app_name: str) -> tuple[Path, ...]:
    normalized = _normalize_app_name(app_name)
    paths = list(KNOWN_PATHS.get(normalized, ()))
    for alias, alias_commands in APP_ALIASES.items():
        if normalized == alias or normalized in alias_commands:
            paths.extend(KNOWN_PATHS.get(alias, ()))
    return tuple(dict.fromkeys(paths))


def _launch_command(command: str) -> None:
    if Path(command).name == "xdg-open":
        subprocess.Popen([command, str(Path.home())], close_fds=True)
        return
    subprocess.Popen([command], close_fds=True)


def _open_with_os(app_name: str) -> bool:
    system = platform.system().lower()
    try:
        if system == "windows":
            os.startfile(app_name)  # type: ignore[attr-defined]
            return True
        if system == "darwin":
            subprocess.Popen(["open", "-a", app_name], close_fds=True)
            return True

        opener = which("gtk-launch")
        if opener:
            for desktop_id in _desktop_ids(app_name):
                try:
                    completed = subprocess.run(
                        [opener, desktop_id],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=2,
                        check=False,
                    )
                except (OSError, subprocess.TimeoutExpired):
                    continue
                if completed.returncode == 0:
                    return True

        opener = which("xdg-open")
        if opener and Path(app_name).exists():
            subprocess.Popen([opener, app_name], close_fds=True)
            return True
    except OSError:
        return False

    return False


def _desktop_ids(app_name: str) -> tuple[str, ...]:
    normalized = _normalize_app_name(app_name)
    hyphenated = normalized.replace(" ", "-")
    compact = normalized.replace(" ", "")
    return tuple(
        dict.fromkeys(
            (
                normalized,
                hyphenated,
                compact,
                f"{normalized}.desktop",
                f"{hyphenated}.desktop",
                f"{compact}.desktop",
            )
        )
    )


def open_app(app_name: str) -> AppLaunchResult:
    normalized = _normalize_app_name(app_name)
    if not normalized:
        return AppLaunchResult(False, "Tell me which app to open.")

    for command in _candidate_commands(normalized):
        if command.endswith(":") and platform.system().lower() == "windows":
            os.startfile(command)  # type: ignore[attr-defined]
            return AppLaunchResult(True, f"Opening {normalized}.")

        executable = which(command)
        if executable:
            try:
                _launch_command(executable)
                return AppLaunchResult(True, f"Opening {normalized}.")
            except OSError as exc:
                return AppLaunchResult(False, f"I found {normalized}, but the OS could not open it: {exc}.")

    for path in _candidate_paths(normalized):
        if path.exists():
            try:
                subprocess.Popen([str(path)], close_fds=True)
                return AppLaunchResult(True, f"Opening {normalized}.")
            except OSError as exc:
                return AppLaunchResult(False, f"I found {normalized}, but the OS could not open it: {exc}.")

    if _open_with_os(normalized):
        return AppLaunchResult(True, f"Opening {normalized}.")

    return AppLaunchResult(
        False,
        f"I couldn't find {normalized}. Try the exact app name or add it to PATH.",
    )
