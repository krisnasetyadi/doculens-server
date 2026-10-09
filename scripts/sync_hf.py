"""MS-657: keep hf-doculens-api (the deployed HF Space) identical to this
repo, without relying on anyone remembering to edit both.

pdf-reader is the source. Every file git tracks here is mirrored to the
Space repo, except the ones listed in SOURCE_ONLY (research notebooks,
docs, sample data, perf records). Files only the Space needs are listed in
TARGET_ONLY and left alone. A new module is picked up automatically once
it is tracked (git add) here.

    python scripts/sync_hf.py --check    # exit 1 and list every difference
    python scripts/sync_hf.py --write    # copy/delete files in the Space repo

--write only changes files on disk; it never stages, commits or pushes.
Line endings are ignored when comparing (the two checkouts differ in
.gitattributes); everything else must match byte for byte.
"""
import argparse
import fnmatch
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TARGET = os.path.join(os.path.dirname(ROOT), "hf-doculens-api")

# Tracked here, never deployed.
SOURCE_ONLY = [
    "*.ipynb",
    "*.md",
    "data/*",
    "perf/*",
    ".vscode/*",
    ".env.example",
    ".gitignore",
]
SOURCE_ONLY_EXCEPTIONS = ["README.md"]  # the Space reads its config from README.md

# Tracked only in the Space repo (or tracked in both but kept per repo).
TARGET_ONLY = [
    ".gitattributes",
    ".gitignore",
    ".vscode/*",
]


def _matches(path, patterns):
    return any(fnmatch.fnmatch(path, p) for p in patterns)


def _tracked(repo):
    out = subprocess.run(["git", "-C", repo, "ls-files", "-z"], capture_output=True, check=True).stdout
    return {p for p in out.decode("utf-8").split("\0") if p}


def synced_files(source_files):
    return sorted(p for p in source_files
                  if p in SOURCE_ONLY_EXCEPTIONS or not _matches(p, SOURCE_ONLY))


def _content(path):
    with open(path, "rb") as f:
        return f.read().replace(b"\r\n", b"\n")


def plan(source, target, source_files, target_files):
    """(to_copy, to_delete): paths whose target copy is missing or differs,
    and target files that are neither synced from source nor TARGET_ONLY."""
    wanted = synced_files(source_files)
    to_copy = [p for p in wanted
               if not os.path.exists(os.path.join(target, p))
               or _content(os.path.join(source, p)) != _content(os.path.join(target, p))]
    to_delete = sorted(p for p in target_files
                       if p not in wanted and not _matches(p, TARGET_ONLY)
                       and os.path.exists(os.path.join(target, p)))
    return to_copy, to_delete


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    parser.add_argument("--target", default=DEFAULT_TARGET)
    args = parser.parse_args()

    if not os.path.isdir(os.path.join(args.target, ".git")):
        raise SystemExit(f"not a git checkout: {args.target}")
    to_copy, to_delete = plan(ROOT, args.target, _tracked(ROOT), _tracked(args.target))

    for p in to_copy:
        state = "missing" if not os.path.exists(os.path.join(args.target, p)) else "differs"
        print(f"{'copy' if args.write else state:8} {p}")
        if args.write:
            dest = os.path.join(args.target, p)
            os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
            shutil.copyfile(os.path.join(ROOT, p), dest)
    for p in to_delete:
        print(f"{'delete' if args.write else 'extra':8} {p}")
        if args.write:
            os.remove(os.path.join(args.target, p))

    if not (to_copy or to_delete):
        print("hf-doculens-api matches pdf-reader")
        return
    if args.check:
        print(f"{len(to_copy) + len(to_delete)} difference(s); run with --write to sync")
        sys.exit(1)
    print(f"wrote {len(to_copy)}, deleted {len(to_delete)} in {args.target}; review and commit there")


if __name__ == "__main__":
    main()
