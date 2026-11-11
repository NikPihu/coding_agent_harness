"""Files and images you hand the model with your message.

Three ways in:
  @path/to/file      text files are pasted inline, images are attached
  a dropped file     dragging a file into the terminal types its path, so any
                     existing image path in the message is attached too
  /paste             grabs an image (a screenshot) from the clipboard

Images go out in the OpenAI "content parts" format, which every
OpenAI-compatible server with a vision model understands:
  [{"type": "text", ...}, {"type": "image_url", "image_url": {"url": "data:..."}}]
"""

import base64
import io
import re
import tempfile
from pathlib import Path

IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
MAX_IMAGE_SIDE = 1568  # bigger costs more tokens and buys nothing
MAX_TEXT_CHARS = 100_000

# @"quoted path"  |  @path  |  "quoted path"  |  bare token
TOKEN = re.compile(r'@"([^"]+)"|@(\S+)|"([^"]+)"|(\S+)')

PENDING = []  # images from /paste, sent with the next message


class AttachmentError(Exception):
    pass


def _image_url(path):
    """A data: URL, shrunk to a sensible size and re-encoded."""
    from PIL import Image

    with Image.open(path) as image:
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        has_alpha = image.mode in ("RGBA", "LA", "P")
        image = image.convert("RGBA" if has_alpha else "RGB")
        buffer = io.BytesIO()
        kind = "png" if has_alpha else "jpeg"
        image.save(buffer, format=kind.upper(), **({} if has_alpha else {"quality": 88}))
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{kind};base64,{data}"


def _read_text(path):
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise AttachmentError(f"{path} is not a text file or image - cannot attach it")
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS] + f"\n[... cut at {MAX_TEXT_CHARS} chars]"
    return text


def _existing(raw):
    raw = raw.strip().strip("'")
    if not raw or len(raw) > 400:
        return None
    try:
        path = Path(raw).expanduser()
        return path if path.is_file() else None
    except OSError:
        return None


def build(message):
    """Turn what you typed into message content.

    Returns (content, attached) where content is a plain string when there
    are no images, and attached is a list of short labels to show you.
    """
    texts, images, attached = [], [], []

    def replace(match):
        at_quoted, at_bare, quoted, bare = match.groups()
        raw = at_quoted or at_bare or quoted or bare
        path = _existing(raw)
        mentioned = bool(at_quoted or at_bare)

        if path is None:
            return match.group(0)  # not a file (an @handle, a typo): leave the text alone

        if path.suffix.lower() in IMAGE_TYPES:
            images.append(_image_url(path))
            attached.append(f"image {path.name}")
            # Say it is attached, or the model goes hunting for the file on disk.
            return f"[image {path.name} - attached to this message, look at it directly]"

        if not mentioned:
            return match.group(0)  # a bare path to a text file is just text
        texts.append(f'<file path="{path}">\n{_read_text(path)}\n</file>')
        attached.append(f"file {path.name}")
        return str(path)

    text = TOKEN.sub(replace, message)

    for pasted in PENDING:
        images.append(_image_url(pasted))
        attached.append(f"image {pasted.name}")
    PENDING.clear()

    if texts:
        text = text + "\n\n" + "\n\n".join(texts)
    if not images:
        return text, attached
    return [{"type": "text", "text": text}] + [
        {"type": "image_url", "image_url": {"url": url}} for url in images
    ], attached


def paste():
    """Grab an image from the clipboard and queue it. Returns a label."""
    from PIL import Image, ImageGrab

    grabbed = ImageGrab.grabclipboard()
    if isinstance(grabbed, Image.Image):
        path = Path(tempfile.gettempdir()) / f"argus-paste-{len(PENDING) + 1}.png"
        grabbed.save(path)
        PENDING.append(path)
        return "clipboard image"
    if isinstance(grabbed, list):  # files copied in Explorer
        added = [Path(p) for p in grabbed if Path(p).suffix.lower() in IMAGE_TYPES]
        PENDING.extend(added)
        if added:
            return ", ".join(p.name for p in added)
    raise AttachmentError("no image on the clipboard - copy a screenshot or image first")
