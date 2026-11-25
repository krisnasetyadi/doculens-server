"""MS-1114: reading a requirement list from a spreadsheet (an SoA, a control
register, a template of required forms), so an analysis can start from it.

The story:
- read_sheets: every sheet's non-empty rows, as Excel numbers them;
- guess_mapping: which sheet, which row holds the headers, and which columns
  hold the number, the name and "applicable", from the header words alone
  (the AI proposes a mapping too, services/analysis_setup.py; this is the
  fallback and the check on it);
- apply_mapping: every row under the header becomes one requirement, read
  as written, with a pointer to its row. Nothing here guesses a requirement,
  so a 93-row SoA gives exactly its 93 requirements.
"""
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from openpyxl.utils import get_column_letter

from services.rag.gap_locations import requirement_key, split_code

logger = logging.getLogger(__name__)

# How much of a workbook is read: enough for any real requirement table,
# bounded so an export of a huge sheet can't stall the request.
MAX_ROWS_PER_SHEET = 3000
MAX_COLUMNS = 40
# Rows shown per sheet when the AI proposes a mapping, and to the user.
PREVIEW_ROWS = 8

# Header words per role, lower case, matched as whole words.
_HEADER_WORDS = {
    "code": ("no", "no.", "nomor", "number", "num", "#", "id", "code", "kode", "ref", "reference", "clause", "klausul",
             "control id", "control no", "kontrol no", "item"),
    "name": ("name", "nama", "title", "judul", "control", "kontrol", "requirement", "persyaratan", "syarat",
             "description", "deskripsi", "uraian", "control name", "nama kontrol", "form", "formulir"),
    "applicable": ("applicable", "applicability", "berlaku", "included", "include", "status", "selected",
                   "dipilih", "diterapkan", "implemented", "in scope", "scope", "y/n", "yes/no"),
}

_YES = {"yes", "y", "ya", "true", "1", "applicable", "included", "berlaku", "selected", "in scope", "v", "x", "✓", "✔"}
_NO = {"no", "n", "tidak", "tdk", "false", "0", "not applicable", "n/a", "na", "excluded", "tidak berlaku",
       "out of scope", "not selected", "-", "✗", "✘"}


def _text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip() if value is not None else ""


def read_sheets(xlsx_path: str) -> List[Dict[str, Any]]:
    """[{"name", "rows": [(excel_row_number, [cell text, ...])]}] for each
    sheet, empty rows left out. [] when the file can't be read."""
    try:
        import openpyxl
        workbook = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    except Exception as e:
        logger.warning("read_sheets failed for %s: %s", xlsx_path, e)
        return []
    sheets = []
    try:
        for sheet in workbook.worksheets:
            rows: List[Tuple[int, List[str]]] = []
            for number, cells in enumerate(sheet.iter_rows(values_only=True, max_col=MAX_COLUMNS), start=1):
                if number > MAX_ROWS_PER_SHEET:
                    break
                texts = [_text(c) for c in cells]
                while texts and not texts[-1]:
                    texts.pop()  # read-only rows are padded to max_col
                if texts:
                    rows.append((number, texts))
            if rows:
                sheets.append({"name": sheet.title, "rows": rows})
    finally:
        workbook.close()
    return sheets


def sheet_preview(sheets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Each sheet's first rows, with column letters: what the AI and the user
    see when the mapping is proposed."""
    preview = []
    for sheet in sheets:
        rows = sheet["rows"][:PREVIEW_ROWS]
        width = max((len(cells) for _, cells in rows), default=0)
        preview.append({
            "name": sheet["name"],
            "row_count": len(sheet["rows"]),
            "columns": [get_column_letter(i + 1) for i in range(width)],
            "rows": [{"row": number, "cells": cells} for number, cells in rows],
        })
    return preview


def _role_of(header: str) -> Optional[str]:
    words = header.lower().strip(" :")
    for role in ("applicable", "code", "name"):
        if words in _HEADER_WORDS[role] or any(re.search(rf"(^|\W){re.escape(w)}($|\W)", words) for w in _HEADER_WORDS[role] if len(w) > 2):
            return role
    return None


def guess_mapping(sheets: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The likeliest {"sheet", "header_row", "code", "name", "applicable"}
    (columns as letters; code and applicable may be None) from header words:
    the sheet and row whose cells name the most roles, then the longest
    sheet. None when no row names a name column."""
    best: Optional[Tuple[int, int, Dict[str, Any]]] = None
    for sheet in sheets:
        for number, cells in sheet["rows"][:20]:
            roles: Dict[str, str] = {}
            for i, cell in enumerate(cells):
                role = _role_of(cell) if cell else None
                if role and role not in roles:
                    roles[role] = get_column_letter(i + 1)
            if "name" not in roles:
                continue
            score = (len(roles), len(sheet["rows"]))
            if best is None or score > best[:2]:
                best = (score[0], score[1], {
                    "sheet": sheet["name"], "header_row": number,
                    "code": roles.get("code"), "name": roles["name"], "applicable": roles.get("applicable"),
                })
    return best[2] if best else None


def valid_mapping(sheets: List[Dict[str, Any]], mapping: Dict[str, Any]) -> bool:
    """True when the mapping names a sheet and header row that exist and a
    name column within that sheet's width."""
    sheet = next((s for s in sheets if s["name"] == mapping.get("sheet")), None)
    if not sheet or not mapping.get("name"):
        return False
    if not any(number == mapping.get("header_row") for number, _ in sheet["rows"]):
        return False
    width = max(len(cells) for _, cells in sheet["rows"])
    letters = {get_column_letter(i + 1) for i in range(width)}
    return all(mapping.get(role) in letters for role in ("code", "name", "applicable") if mapping.get(role))


def applicable(value: str) -> Optional[bool]:
    """True/False for a recognised yes/no cell, None when it says neither."""
    text = value.lower().strip()
    if not text:
        return None
    if text in _YES or text.startswith(("yes", "ya ", "applicable")):
        return True
    if text in _NO or text.startswith(("no ", "not ", "tidak", "excluded")):
        return False
    return None


def apply_mapping(sheets: List[Dict[str, Any]], mapping: Dict[str, Any], file_label: str) -> List[Dict[str, Any]]:
    """Every row under the header of the mapped sheet with a name (or a
    number), read as written: one requirement per row, keyed like any rule
    set's (gap_locations.build_requirements), with `applicable` (None when
    the sheet doesn't say) and `source`, a pointer to its row."""
    sheet = next((s for s in sheets if s["name"] == mapping["sheet"]), None)
    if not sheet:
        return []
    index = {role: _column_index(mapping.get(role)) for role in ("code", "name", "applicable")}
    many_sheets = len(sheets) > 1
    taken: set = set()
    requirements = []
    for number, cells in sheet["rows"]:
        if number <= mapping["header_row"]:
            continue
        code = _cell(cells, index["code"]) or None
        name = _cell(cells, index["name"])
        if not name and not code:
            continue
        if not code:
            code, name = split_code(name)  # "A.5.1 Policies" in one column
        place = f"{sheet['name']}, row {number}" if many_sheets else f"row {number}"
        requirements.append({
            "key": requirement_key(code, name or code, taken),
            "code": code,
            "label": name or code,
            "expected_document": None,
            "applicable": applicable(_cell(cells, index["applicable"])) if index["applicable"] is not None else None,
            "source": {"label": f"{file_label} · {place}", "sheet": sheet["name"], "row": number},
        })
    return requirements


def _column_index(letter: Optional[str]) -> Optional[int]:
    if not letter:
        return None
    from openpyxl.utils import column_index_from_string
    return column_index_from_string(letter) - 1


def _cell(cells: List[str], index: Optional[int]) -> str:
    return cells[index] if index is not None and index < len(cells) else ""
