-- MS-557: existing folders remain at the root (NULL parent).
ALTER TABLE folders
    ADD COLUMN IF NOT EXISTS parent_folder_id TEXT
    REFERENCES folders(folder_id) ON DELETE RESTRICT;

CREATE INDEX IF NOT EXISTS idx_folders_parent ON folders (parent_folder_id);
