# TO_REMOVE
# This file is a one-off script / dev utility and is no longer needed.
# Safe to delete after confirming no active references.
# -------------------------------------------------------------------
"""
parse_chat_corpus.py
====================
Preprocessing pipeline untuk corpus Layer 3 (Teams chat MOFIDS).

Input  : cuplikan, cuplikan2, cuplikan-personal-message  (Teams copy-paste format)
Output : data/processed_chats/
           chat_group_mofids.txt          (Grup besar 2022)
           chat_group_mofids2.txt         (Grup sedang 2025)
           chat_personal_mofids.txt       (Personal Dev_B-PO_1 2022)

Format output (sama dengan adaro_analyst_chat.txt):
  --- GRUP: <label> ---
  --- Anggota: <daftar pseudonim (role)> ---
  --- Konteks: <deskripsi singkat> ---

  [DD/MM/YYYY, HH:MM:SS] Pseudonim: isi pesan

Jalankan:
  python parse_chat_corpus.py
  python parse_chat_corpus.py --dry-run
"""

import re
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "data" / "processed_chats"

CHAT_FILES = [
    {
        "path":    BASE_DIR / "cuplikan",
        "source":  "chat_group_mofids",
        "label":   "Grup MOFIDS 2022",
        "konteks": "Diskusi tim pengembang aplikasi MOFIDS (IDX), Juli-September 2022. "
                   "Mencakup bug fixing, UAT, demo preparation, dan koordinasi deployment.",
    },
    {
        "path":    BASE_DIR / "cuplikan2",
        "source":  "chat_group_mofids2",
        "label":   "Grup MOFIDS 2025",
        "konteks": "Diskusi tim pengembang aplikasi MOFIDS, Februari 2025. "
                   "Mencakup pengembangan fitur amend, quotation table redesign, dan isu trade custody.",
    },
    {
        "path":    BASE_DIR / "cuplikan-personal-message",
        "source":  "chat_personal_mofids",
        "label":   "Pesan Personal Dev_B - PO_1 (2022)",
        "konteks": "Percakapan personal antara Dev_B (developer) dan PO_1 (product owner) "
                   "mengenai validasi desimal offering price dan offering digit di MOFIDS.",
    },
]

# --- Anonimisasi --------------------------------------------------------------

NAME_MASK: Dict[str, str] = {
    "patresia ratu wetti sitanggang": "PO_1",
    "patresia":                       "PO_1",
    "mesakh dwi putra":               "PO_2",
    "mesakh":                         "PO_2",
    "ivena chindy claudia":           "PO_3",
    "ivena":                          "PO_3",
    "bondan chaya nugraha":           "PM_1",
    "bondan chahya nugraha":          "PM_1",
    "bondan":                         "PM_1",
    "ardy maulana":                   "Dev_A",
    "ardy":                           "Dev_A",
    "krisna dwi setyaadi":            "Dev_B",
    "krisna":                         "Dev_B",
    "sheldy rivaldi":                 "Dev_C",
    "sheldy":                         "Dev_C",
    "ezra hutapea":                   "Dev_D",
    "ezra":                           "Dev_D",
    "dhifa irawan":                   "Dev_E",
    "dhifa":                          "Dev_E",
    "julio lemena":                   "Dev_F",
    "julio":                          "Dev_F",
    "leslie aula":                    "Dev_G",
    "leslie":                         "Dev_G",
    "sandy agustinus suherman":       "DevOps_1",
    "sandy":                          "DevOps_1",
    "moonlay":                        "DevOps_1",
}

ROLE_LABEL: Dict[str, str] = {
    "PO_1": "Product Owner", "PO_2": "Product Owner", "PO_3": "Product Owner",
    "PM_1": "Project Manager",
    "Dev_A": "Developer", "Dev_B": "Developer", "Dev_C": "Developer",
    "Dev_D": "Developer", "Dev_E": "Developer", "Dev_F": "Developer",
    "Dev_G": "Developer",
    "DevOps_1": "DevOps",
}


def resolve_sender(raw: str) -> str:
    lower = raw.strip().lower()
    for real, pseudo in sorted(NAME_MASK.items(), key=lambda x: -len(x[0])):
        if real in lower:
            return pseudo
    return "Unknown"


def mask_fullnames(text: str) -> str:
    """Mask nama lengkap (2+ kata) dalam konten. Sapaan tunggal dibiarkan."""
    result = text
    for real, pseudo in sorted(NAME_MASK.items(), key=lambda x: -len(x[0])):
        if " " in real:
            result = re.sub(re.escape(real), pseudo, result, flags=re.IGNORECASE)
    return result


# --- Regex --------------------------------------------------------------------

# Baris UI noise Teams
TEAMS_UI_NOISE = re.compile(
    r"^("
    r"2 new notifications|New missed calls|Activity in other accounts.*|"
    r"Has context menu|Chat$|Unread$|Channels$|Chats$|Meeting chats$|"
    r"Press Ctrl\+F to find.*|Unread messageLast message.*|See more.*|"
    r"Community.*|Temporarily shown.*|Badged chat.*|Meet now|"
    r"Call (started|ended) at.*|\d+ new notification.*|, \+\d+$|"
    r"\d+$"
    r")",
    re.IGNORECASE,
)

# Baris preview Teams: "<teks pendek>... by <Nama Pengirim>"
# Format: konten dipenggal lalu " by <Nama>" di akhir
PREVIEW_LINE = re.compile(r"^.+\sby\s.+$")

# Timestamp Teams: "7/27/2022 10:34 AM" atau "2/3/2025 7:12 AM"
TEAMS_DATETIME = re.compile(
    r"^(\d{1,2}/\d{1,2}/\d{4})\s+(\d{1,2}:\d{2})\s*([AP]M)$",
    re.IGNORECASE,
)

# "Begin quote, ..."
BEGIN_QUOTE = re.compile(r"^Begin quote,", re.IGNORECASE)


def parse_teams_datetime(s: str) -> Optional[datetime]:
    s = s.strip()
    for fmt in ("%m/%d/%Y %I:%M %p", "%m/%d/%Y %I:%M%p"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def strip_image_prefix(text: str) -> str:
    """Hapus prefiks 'image' atau 'Selected photo' dari konten."""
    text = re.sub(r"^image\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^Selected photo\s*", "[foto]", text, flags=re.IGNORECASE)
    return text.strip()


# --- Parser -------------------------------------------------------------------

class TeamsMessage:
    __slots__ = ("timestamp", "sender", "content", "source")

    def __init__(self, timestamp: datetime, sender: str, content: str, source: str):
        self.timestamp = timestamp
        self.sender    = sender
        self.content   = content
        self.source    = source


def parse_teams_file(filepath: Path, source: str) -> List[TeamsMessage]:
    """
    Parser untuk Teams copy-paste format.

    Setiap blok pesan punya struktur:
      <baris preview>... by <Nama>    <- BUANG (summary/notif Teams)
      <Nama Pengirim>                 <- sender (urutan bisa terbalik dg timestamp)
      <M/D/YYYY H:MM AM/PM>          <- timestamp
      (blank)
      <konten baris 1>               <- AMBIL
      <konten baris 2>               <- AMBIL
      (blank atau blok berikutnya)

    Pada beberapa file urutan Nama dan Timestamp bisa terbalik.
    """
    messages: List[TeamsMessage] = []

    try:
        raw = filepath.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.error(f"Gagal membaca {filepath}: {e}")
        return messages

    lines = [l.rstrip() for l in raw.splitlines()]

    # State: SCAN ? temukan blok baru
    #        HEADER ? sedang kumpulkan nama+timestamp
    #        CONTENT ? kumpulkan konten pesan
    state = "SCAN"

    pending_sender:  Optional[str]      = None
    pending_dt:      Optional[datetime] = None
    current_content: List[str]          = []

    def flush():
        nonlocal pending_sender, pending_dt, current_content
        if pending_sender and pending_dt and current_content:
            raw_text = " ".join(current_content).strip()
            raw_text = strip_image_prefix(raw_text)
            raw_text = mask_fullnames(raw_text)
            if raw_text:
                messages.append(
                    TeamsMessage(pending_dt, pending_sender, raw_text, source)
                )
        pending_sender  = None
        pending_dt      = None
        current_content = []

    for line in lines:
        stripped = line.strip()

        # -- Selalu buang UI noise dan Begin quote --
        if not stripped:
            if state == "CONTENT":
                # Blank di tengah konten = akhir pesan
                flush()
                state = "SCAN"
            elif state == "HEADER":
                # Blank setelah header = konten dimulai
                if pending_sender and pending_dt:
                    state = "CONTENT"
                # jika header belum lengkap, tetap tunggu
            continue

        if TEAMS_UI_NOISE.match(stripped):
            continue

        if BEGIN_QUOTE.match(stripped):
            continue

        # -- Deteksi baris preview "... by <Nama>" --
        # Baris ini menandai AWAL blok pesan baru
        if PREVIEW_LINE.match(stripped):
            # Flush pesan sebelumnya jika ada
            if state == "CONTENT":
                flush()
            elif state == "HEADER":
                # Header tidak lengkap, buang
                pending_sender = None
                pending_dt     = None

            # Nama pengirim ada di token setelah " by " terakhir
            by_match = re.search(r"\sby\s(.+)$", stripped)
            if by_match:
                raw_name = by_match.group(1).strip()
                # "Unknown User" ? tetap Unknown, akan di-override oleh baris nama berikutnya
                pending_sender = resolve_sender(raw_name)
            else:
                pending_sender = "Unknown"

            pending_dt      = None
            current_content = []
            state = "HEADER"
            continue

        # -- Di dalam HEADER: cari timestamp atau konfirmasi nama --
        if state == "HEADER":
            dt_match = TEAMS_DATETIME.match(stripped)
            if dt_match:
                dt_str = f"{dt_match.group(1)} {dt_match.group(2)} {dt_match.group(3)}"
                pending_dt = parse_teams_datetime(dt_str)
                continue

            # Cek apakah ini baris nama pengirim (lebih akurat dari nama di preview)
            lower = stripped.lower()
            matched_name = any(real in lower for real in NAME_MASK)
            # Heuristik: nama = PascalCase, =5 kata, bukan kalimat
            looks_like_name = (
                matched_name
                or (
                    re.match(r"^[A-Z][a-zA-Z]+(\s[A-Z][a-zA-Z]+)*$", stripped)
                    and len(stripped.split()) <= 5
                )
            )
            if looks_like_name:
                # Override sender dengan nama yang lebih lengkap
                pending_sender = resolve_sender(stripped)
                continue

            # Baris lain di HEADER yang tidak dikenali ? masuk konten
            # (beberapa file tidak ada blank setelah timestamp)
            if pending_sender and pending_dt:
                state = "CONTENT"
                text = strip_image_prefix(stripped)
                if text:
                    current_content.append(text)
            continue

        # -- Di dalam CONTENT: kumpulkan baris --
        if state == "CONTENT":
            text = strip_image_prefix(stripped)
            if text:
                current_content.append(text)
            continue

        # -- SCAN: belum ada blok aktif, abaikan --

    # Flush terakhir
    if state == "CONTENT":
        flush()

    logger.info(f"  {filepath.name}: {len(messages)} pesan diparsing")
    return messages


# --- Output Writer ------------------------------------------------------------

def collect_participants(messages: List[TeamsMessage]) -> str:
    seen: Dict[str, str] = {}
    for msg in messages:
        s = msg.sender
        if s not in seen and s != "Unknown":
            seen[s] = ROLE_LABEL.get(s, "Unknown Role")
    return ", ".join(f"{p} ({r})" for p, r in seen.items()) or "Unknown"


def write_clean_txt(messages: List[TeamsMessage], cfg: Dict, output_path: Path) -> None:
    participants = collect_participants(messages)
    lines = [
        f"--- GRUP: {cfg['label']} ---",
        f"--- Anggota: {participants} ---",
        f"--- Konteks: {cfg['konteks']} ---",
        "",
    ]
    for msg in messages:
        ts = msg.timestamp.strftime("%d/%m/%Y, %H:%M:%S")
        lines.append(f"[{ts}] {msg.sender}: {msg.content}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"  -> {output_path} ({len(messages)} pesan)")


# --- Main ---------------------------------------------------------------------

def main(dry_run: bool = False):
    total_messages = 0

    print("\n" + "=" * 60)
    print("  MOFIDS Chat Corpus Parser")
    print("=" * 60)

    for cfg in CHAT_FILES:
        path  = cfg["path"]
        label = cfg["label"]

        if not path.exists():
            logger.warning(f"File tidak ditemukan: {path}")
            continue

        print(f"\n[{label}]")
        print(f"  Input : {path}")

        messages = parse_teams_file(path, cfg["source"])
        total_messages += len(messages)
        print(f"  Pesan : {len(messages)}")

        if not dry_run and messages:
            out_file = OUTPUT_DIR / f"{cfg['source']}.txt"
            write_clean_txt(messages, cfg, out_file)
            print(f"  Output: {out_file}")
            print("  Sample (5 pertama):")
            for msg in messages[:5]:
                ts = msg.timestamp.strftime("%d/%m/%Y, %H:%M")
                preview = msg.content[:90].replace("\n", " ")
                print(f"    [{ts}] {msg.sender}: {preview}")

    print("\n" + "=" * 60)
    print(f"  Total pesan  : {total_messages}")
    if dry_run:
        print("  [DRY RUN] File tidak ditulis.")
    else:
        print(f"  Output dir   : {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse Teams chat -> plain .txt corpus")
    parser.add_argument("--dry-run", action="store_true", help="Statistik saja, tidak tulis file")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
