"""One-time cleanup (MS-913): delete the chat sources nobody owns.

Chats are private to whoever added them. Rows made before that was enforced
have no owner, so since MS-913 nobody sees or searches them. This deletes
them for good:

  * Telegram connections whose user_id is not a DocuLens user (they were
    saved with a fixed id from the environment). Each is logged out of
    Telegram and every chat it synced is deleted with it, the same as
    disconnecting it in Sources.
  * Chat collections (WhatsApp uploads, Telegram chats) with no owner_id, or
    one that is not a DocuLens user (a deleted user, or a fixed id such as
    "test-user"): their row, their files in Supabase Storage and the local
    copies.

Run manually after deploying MS-913, against the database in .env:
    python scripts/delete_ownerless_chats.py            # lists what would go
    python scripts/delete_ownerless_chats.py --delete   # deletes it

Safe to re-run: what is already gone is no longer found. Restart the server
afterwards so no deleted index stays loaded in memory (none is searched
either way).
"""
import argparse
import asyncio
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db
from config import config
from services.storage import supabase as storage


def _rows(sql, params=()):
    conn = db.get_conn("storage")
    if not conn:
        raise SystemExit("Database unavailable: check DATABASE_URL in .env")
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall() if cur.description else []
    finally:
        conn.close()


def ownerless_connections():
    return _rows(
        """
        SELECT * FROM telegram_connections AS c
        WHERE NOT EXISTS (SELECT 1 FROM users AS u WHERE u.user_id = c.user_id)
        ORDER BY created_at
        """
    )


def synced_chats(connection_id):
    return _rows(
        """
        SELECT s.chat_collection_id AS collection_id, c.owner_id
        FROM telegram_selected_chats AS s
        LEFT JOIN collections AS c ON c.collection_id = s.chat_collection_id
        WHERE s.connection_id = %s AND s.chat_collection_id IS NOT NULL AND s.chat_collection_id <> ''
        """,
        (connection_id,),
    )


def ownerless_chats():
    return _rows(
        """
        SELECT c.collection_id, c.owner_id, c.file_names, c.metadata->>'platform' AS platform, c.created_at
        FROM collections AS c
        WHERE c.kind = 'chat'
          AND NOT EXISTS (SELECT 1 FROM users AS u WHERE u.user_id = c.owner_id)
        ORDER BY c.created_at
        """
    )


def delete_chat(collection_id, owner_id):
    """Row, Storage files and local copies, as deleting it in Sources does."""
    storage.delete_chat_collection_from_db(collection_id, owner_id)
    for folder in (config.chat_upload_folder, config.chat_index_folder):
        shutil.rmtree(os.path.join(folder, collection_id), ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--delete", action="store_true", help="delete them (without it, only list them)")
    args = parser.parse_args()

    connections = ownerless_connections()
    chats = ownerless_chats()
    print(f"Telegram connections with no DocuLens owner: {len(connections)}")
    for row in connections:
        print(f"  {row['connection_id']}  {row['label']}  ({len(synced_chats(row['connection_id']))} synced chat(s))")
    print(f"Chat collections with no DocuLens owner: {len(chats)}")
    for row in chats:
        name = (row["file_names"] or ["?"])[0]
        print(f"  {row['collection_id']}  {row['platform'] or '?'}  {name}  (owner: {row['owner_id'] or 'none'})")

    if not args.delete:
        print("\nNothing deleted. Run again with --delete to delete the above.")
        return

    from services import telegram

    for row in connections:
        asyncio.run(telegram._log_out_saved_session(row))
        for chat in synced_chats(row["connection_id"]):
            delete_chat(chat["collection_id"], chat["owner_id"])
        _rows("DELETE FROM telegram_connections WHERE connection_id = %s", (row["connection_id"],))
        print(f"deleted connection {row['connection_id']}")
    # Listed again: a connection's synced chats may have been among them.
    for row in ownerless_chats():
        delete_chat(row["collection_id"], row["owner_id"])
        print(f"deleted chat {row['collection_id']}")


if __name__ == "__main__":
    main()
