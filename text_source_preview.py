"""Read-only subtype selection for text source previews."""

from chat_parser import ChatParser


def read_text_source_page(file_path, platform, offset, limit):
    platform = (platform or "").strip().lower()
    messages = []
    # Legacy sources without subtype metadata are detected from their content.
    # An explicit non-WhatsApp subtype must never use WhatsApp formatting.
    if platform in ("", "whatsapp"):
        messages, _ = ChatParser().parse_whatsapp(file_path)

    if messages:
        total = len(messages)
        return {
            "subtype": "whatsapp",
            "platform": "whatsapp",
            "total": total,
            "has_more": offset + limit < total,
            "messages": [m.model_dump(mode="json") for m in messages[offset:offset + limit]],
            "lines": [],
        }

    # Preserve the original text, including indentation and blank lines.
    for encoding in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            with open(file_path, encoding=encoding) as source:
                content = source.read()
            break
        except UnicodeDecodeError:
            continue
    lines = content.splitlines() if content.strip() else []
    total = len(lines)
    return {
        "subtype": "plain_text",
        "platform": platform or "generic",
        "total": total,
        "has_more": offset + limit < total,
        "messages": [],
        "lines": [
            {"line_number": offset + i + 1, "content": line}
            for i, line in enumerate(lines[offset:offset + limit])
        ],
    }
