"""A message's content is either a string or a list of parts (text + images).

Everything that only needs the words - titles, the compaction transcript,
token estimates - goes through text_of() so it never has to care which.
"""

IMAGE_TOKENS = 1_000  # rough cost of one image, for estimates


def text_of(content):
    """The text of a message, with each image as a short placeholder."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    pieces = []
    for part in content:
        if part.get("type") == "text":
            pieces.append(part.get("text", ""))
        elif part.get("type") == "image_url":
            pieces.append("[image]")
    return "\n".join(pieces)


def image_count(content):
    if isinstance(content, list):
        return sum(1 for part in content if part.get("type") == "image_url")
    return 0
