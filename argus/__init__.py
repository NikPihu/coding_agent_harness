"""Argus, a minimal coding agent."""

import sys

# Windows consoles and pipes often default to cp1252, which has no box-drawing
# characters: the first panel the UI draws would crash with UnicodeEncodeError.
# Speak UTF-8, and never die over a character the console cannot show.
if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
