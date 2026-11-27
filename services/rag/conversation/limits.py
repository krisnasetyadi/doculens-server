"""History limits shared by chat prompts and Lens Guide continuation (MS-237)."""

# Count chats as questions plus their answers. The message cap also bounds a
# client that pads one question with many assistant turns.
MAX_MEMORY_CHATS = 5
MAX_MEMORY_MESSAGES = MAX_MEMORY_CHATS * 4
