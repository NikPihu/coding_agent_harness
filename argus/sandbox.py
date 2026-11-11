"""Kernel-enforced limits on what bash can touch.

One policy - read anything, write only inside the project, no network - and a
different enforcement mechanism per OS. The idea ports; the mechanism never does.

The macOS profile is adapted from openai/codex (Apache-2.0), simplified.
https://github.com/openai/codex

Windows has no equivalent we can switch on from Python, so there the command
runs unsandboxed through Git Bash. Models write POSIX shell (ls, grep, cat),
which cmd.exe does not understand. permissions.py is the only guard left.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT = Path.cwd().resolve()

PROFILE = f"""(version 1)
(deny default)
(allow process-exec process-fork signal)
(allow file-read*)
(allow sysctl-read)
(deny network*)
(allow file-write* (subpath "{PROJECT}") (literal "/dev/null"))
(deny file-write* (subpath "{PROJECT}/.git"))
"""


def git_bash():
    """Git for Windows' bash. Not `which bash`: on Windows that is often WSL's
    System32\\bash.exe, which runs in a different filesystem."""
    candidates = [
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git" / "bin" / "bash.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Git" / "bin" / "bash.exe",
    ]
    git = shutil.which("git")
    if git:  # ...\Git\cmd\git.exe -> ...\Git\bin\bash.exe
        candidates.append(Path(git).resolve().parent.parent / "bin" / "bash.exe")
    for path in candidates:
        if path.is_file():
            return str(path)
    return None


def wrap(command):
    """Wrap a shell command in an OS sandbox. None means we have no sandbox."""
    if sys.platform == "darwin":
        profile = Path(tempfile.gettempdir()) / "argus.sb"
        profile.write_text(PROFILE)
        return ["sandbox-exec", "-f", str(profile), "/bin/sh", "-c", command]

    if sys.platform.startswith("linux") and shutil.which("bwrap"):
        return [
            "bwrap",
            "--ro-bind", "/", "/",
            "--bind", str(PROJECT), str(PROJECT),
            "--dev", "/dev", "--proc", "/proc",
            "--unshare-net", "--die-with-parent",
            "/bin/sh", "-c", command,
        ]

    if sys.platform == "win32" and git_bash():
        # A shell, not a sandbox. The script goes in on stdin (see run): Windows
        # re-encodes argv to the ANSI code page, which mangles anything non-ASCII.
        return [git_bash(), "-s"]

    return None  # Linux without bubblewrap, or Windows without Git


def name():
    if sys.platform == "darwin":
        return "seatbelt"
    if sys.platform.startswith("linux") and shutil.which("bwrap"):
        return "bubblewrap"
    if sys.platform == "win32" and git_bash():
        return "none (git bash)"
    return "none"


def run(command, timeout=60):
    """Run a command, sandboxed when the OS lets us."""
    sandboxed = wrap(command)
    via_stdin = sys.platform == "win32" and sandboxed is not None
    return subprocess.run(
        sandboxed or command,
        shell=sandboxed is None,
        input=command if via_stdin else None,
        stdin=None if via_stdin else subprocess.DEVNULL,  # never hang waiting for input
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
